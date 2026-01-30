'''
wrappers of the decoder.py, specifically for supervised analysis
'''

import numpy as np
import pandas as pd
import xarray as xr
import jax.numpy as jnp

import poor_man_gplvm.decoder as decoder

def decode_naive_bayes(spk, tuning, tensor_pad_mask=None, flat_idx_to_coord=None, **kwargs):
    '''
    Wrapper around `poor_man_gplvm.decoder.get_naive_bayes_ma_chunk` with a simple supervised API.

    Inputs
    - spk: (n_time, n_neuron) or padded (n_trial, n_time, n_neuron)
    - tuning: (n_label_bin, n_neuron) (typically `tuning_res['tuning_flat']`)
    - tensor_pad_mask: (n_trial, n_time, 1) bool, True for valid time bins (from `trial_analysis.bin_spike_train_to_trial_based`)
    - flat_idx_to_coord: pd.DataFrame indexed by flat_idx, columns ['maze', ...label_dims...] from `get_tuning_supervised.get_tuning`

    Kwargs (forwarded / used internally)
    - n_time_per_chunk: int, default 10000
    - observation_model: 'poisson' (default) or 'gaussian'
    - noise_std: float, only used for gaussian
    - dt_l: scalar or (n_time,) for matrix / (n_valid_time,) after masking for tensor
    - ma_neuron: (n_neuron,) or broadcastable to spk; default all-ones
    - ma_latent: (n_label_bin,) mask; default all-ones

    Returns
    - dict. For matrix input, arrays are (n_time, n_label_bin). For tensor input, per-trial results are lists of arrays with variable n_time.

    Cluster Jupyter examples

```python
import poor_man_gplvm.supervised_analysis.decoder_supervised as dec_sup

# matrix decoding
res = dec_sup.decode_naive_bayes(spk_mat, tuning_res['tuning_flat'], n_time_per_chunk=20000)

# trial tensor decoding
res_t = dec_sup.decode_naive_bayes(spk_tensor, tuning_res['tuning_flat'], tensor_pad_mask=tensor_pad_mask)

# xarray MultiIndex wrapping (label bins)
res_xr = dec_sup.decode_naive_bayes(
    spk_mat,
    tuning_res['tuning_flat'],
    flat_idx_to_coord=tuning_res['flat_idx_to_coord'],
)
```
    '''
    spk = jnp.asarray(spk)
    tuning = jnp.asarray(tuning)

    if spk.ndim == 2:
        res = _decode_naive_bayes_matrix(spk, tuning, **kwargs)
        if flat_idx_to_coord is not None:
            res = _wrap_label_results_xr(res, flat_idx_to_coord=flat_idx_to_coord)
        return res

    if spk.ndim == 3:
        if tensor_pad_mask is None:
            raise ValueError('tensor input requires tensor_pad_mask (n_trial, n_time, 1).')
        res = _decode_naive_bayes_tensor(spk, tuning, tensor_pad_mask=tensor_pad_mask, **kwargs)
        if flat_idx_to_coord is not None:
            res = _wrap_label_results_xr(res, flat_idx_to_coord=flat_idx_to_coord)
        return res

    raise ValueError(f'Unsupported spk ndim={spk.ndim}; expected 2 or 3.')

def _decode_naive_bayes_matrix(spk, tuning, **kwargs):
    '''
    spk: (n_time, n_neuron)
    tuning: (n_label_bin, n_neuron)
    '''
    observation_model = kwargs.get('observation_model', 'poisson')
    n_time_per_chunk = int(kwargs.get('n_time_per_chunk', 10000))
    dt_l = kwargs.get('dt_l', 1)
    ma_neuron = kwargs.get('ma_neuron', jnp.ones(spk.shape[1]))
    ma_latent = kwargs.get('ma_latent', jnp.ones(tuning.shape[0]))
    noise_std = kwargs.get('noise_std', 1.0)

    if observation_model == 'gaussian':
        hyperparam = {'noise_std': noise_std}
    else:
        hyperparam = {}

    log_posterior, log_marginal_l, log_marginal, log_likelihood = decoder.get_naive_bayes_ma_chunk(
        spk,
        tuning,
        hyperparam,
        ma_neuron,
        ma_latent,
        dt_l=dt_l,
        n_time_per_chunk=n_time_per_chunk,
        observation_model=observation_model,
    )

    posterior = jnp.exp(log_posterior)

    res = {
        'log_likelihood': log_likelihood,
        'log_posterior': log_posterior,
        'posterior': posterior,
        'log_marginal_l': log_marginal_l,
        'log_marginal': log_marginal,
    }
    return res

def _decode_naive_bayes_tensor(spk_tensor, tuning, tensor_pad_mask, **kwargs):
    '''
    spk_tensor: (n_trial, n_time, n_neuron) padded with zeros
    tensor_pad_mask: (n_trial, n_time, 1) bool, True for valid bins

    Returns dict where label-related arrays are lists (length n_trial), each entry is (n_time_i, n_label_bin).
    Also returns:
    - log_marginal_l_per_trial: list of (n_time_i,)
    - log_marginal_per_trial: (n_trial,)
    '''
    mask = np.asarray(tensor_pad_mask)[..., 0].astype(bool)  # (n_trial, n_time)
    n_trial, n_time, n_neuron = spk_tensor.shape

    spk_flat = spk_tensor.reshape((n_trial * n_time, n_neuron))
    mask_flat = mask.reshape((n_trial * n_time,))
    spk_valid = spk_flat[mask_flat]

    dt_l = kwargs.get('dt_l', 1)
    if np.ndim(dt_l) > 0 and not np.isscalar(dt_l):
        dt_l = np.asarray(dt_l).reshape((n_trial * n_time,))[mask_flat]

    kwargs2 = dict(kwargs)
    kwargs2['dt_l'] = dt_l

    res_mat = _decode_naive_bayes_matrix(spk_valid, tuning, **kwargs2)

    n_time_per_trial = mask.sum(axis=1).astype(int)  # (n_trial,)
    ends = np.cumsum(n_time_per_trial)
    starts = np.concatenate([np.array([0], dtype=int), ends[:-1]])

    def _split_by_trial(arr_2d):
        return [arr_2d[s:e] for s, e in zip(starts, ends)]

    def _split_1d(arr_1d):
        return [arr_1d[s:e] for s, e in zip(starts, ends)]

    log_likelihood_per_trial = _split_by_trial(res_mat['log_likelihood'])
    log_posterior_per_trial = _split_by_trial(res_mat['log_posterior'])
    posterior_per_trial = _split_by_trial(res_mat['posterior'])
    log_marginal_l_per_trial = _split_1d(res_mat['log_marginal_l'])
    log_marginal_per_trial = jnp.asarray([jnp.sum(x) for x in log_marginal_l_per_trial])

    res = {
        'log_likelihood': log_likelihood_per_trial,
        'log_posterior': log_posterior_per_trial,
        'posterior': posterior_per_trial,
        'log_marginal_l': log_marginal_l_per_trial,
        'log_marginal': log_marginal_per_trial,
    }
    return res


def _wrap_label_results_xr(res, flat_idx_to_coord):
    '''
    Wrap label-related outputs into xr.DataArray with MultiIndex coord `label_bin`.
    - For matrix output: returns DataArray (time, label_bin)
    - For tensor output: returns list[DataArray], one per trial
    '''
    df = flat_idx_to_coord
    if not isinstance(df, pd.DataFrame):
        raise ValueError('flat_idx_to_coord must be a pandas DataFrame.')

    cols = ['maze'] + [c for c in df.columns if c != 'maze']
    midx = pd.MultiIndex.from_frame(df.loc[:, cols])

    def _wrap_2d(arr, time_len):
        return xr.DataArray(
            np.asarray(arr),
            dims=('time', 'label_bin'),
            coords={'time': np.arange(time_len), 'label_bin': midx},
        )

    def _wrap_list(arr_l):
        return [_wrap_2d(a, a.shape[0]) for a in arr_l]

    if isinstance(res.get('log_likelihood', None), list):
        res2 = dict(res)
        res2['log_likelihood'] = _wrap_list(res['log_likelihood'])
        res2['log_posterior'] = _wrap_list(res['log_posterior'])
        res2['posterior'] = _wrap_list(res['posterior'])
        return res2

    n_time = int(np.asarray(res['log_likelihood']).shape[0])
    res2 = dict(res)
    res2['log_likelihood'] = _wrap_2d(res['log_likelihood'], n_time)
    res2['log_posterior'] = _wrap_2d(res['log_posterior'], n_time)
    res2['posterior'] = _wrap_2d(res['posterior'], n_time)
    return res2
