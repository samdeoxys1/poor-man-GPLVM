'''
wrappers of the decoder.py, specifically for supervised analysis
'''

import numpy as np
import pandas as pd
import xarray as xr
import time
import jax.numpy as jnp

import pynapple
import poor_man_gplvm.decoder as decoder

def decode_naive_bayes(spk, tuning, tensor_pad_mask=None, flat_idx_to_coord=None, dt=1.0, gain=1.0, time_l=None, **kwargs):
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
    - dt: scalar, bin size (seconds). Used when dt_l is not provided.
    - gain: scalar, multiply tuning (Hz) by gain before converting to expected counts.
    - dt_l: scalar or (n_time,) for matrix / (n_valid_time,) after masking for tensor (seconds). If provided, dt is ignored.
    - ma_neuron: (n_neuron,) or broadcastable to spk; default all-ones
    - ma_latent: (n_label_bin,) mask; default all-ones
    - time_l: only for tensor input. (n_valid_time,) timestamps for valid bins (trial-major concat)

    Returns
    - dict. For matrix input, arrays are (n_time, n_label_bin). For tensor input, per-trial results are lists of arrays with variable n_time.

    Cluster Jupyter examples

```python
import poor_man_gplvm.supervised_analysis.decoder_supervised as dec_sup

# matrix decoding
res = dec_sup.decode_naive_bayes(spk_mat, tuning_res['tuning_flat'], n_time_per_chunk=20000)

# trial tensor decoding
res_t = dec_sup.decode_naive_bayes(spk_tensor, tuning_res['tuning_flat'], tensor_pad_mask=tensor_pad_mask)

# xarray wrapping (label bins)
res_xr = dec_sup.decode_naive_bayes(
    spk_mat,
    tuning_res['tuning_flat'],
    flat_idx_to_coord=tuning_res['flat_idx_to_coord'],
)
```
    '''
    time_coord = None
    if hasattr(spk, 't'):
        time_coord = np.asarray(spk.t)

    if time_l is None:
        time_l = kwargs.get('time_l', None)

    spk = jnp.asarray(spk)
    tuning = jnp.asarray(tuning)

    if spk.ndim == 2:
        res = _decode_naive_bayes_matrix(spk, tuning, dt=dt, gain=gain, **kwargs)
        res = _decode_res_to_numpy(res)
        if time_coord is None and time_l is not None:
            time_coord = np.asarray(time_l)

        if flat_idx_to_coord is not None:
            res = _wrap_label_results_xr_by_maze(res, flat_idx_to_coord=flat_idx_to_coord, time_coord=time_coord)
        else:
            if time_coord is not None:
                res = _wrap_decode_res_tsdframe_matrix(res, time_coord=time_coord)
        return res

    if spk.ndim == 3:
        if tensor_pad_mask is None:
            raise ValueError('tensor input requires tensor_pad_mask (n_trial, n_time, 1).')
        res = _decode_naive_bayes_tensor(spk, tuning, tensor_pad_mask=tensor_pad_mask, dt=dt, gain=gain, **kwargs)
        res = _decode_res_to_numpy(res)
        if time_l is not None:
            res['time_l'] = np.asarray(time_l)
        if flat_idx_to_coord is not None:
            res = _wrap_label_results_xr_by_maze(res, flat_idx_to_coord=flat_idx_to_coord, time_coord=time_l)
        else:
            if time_l is not None:
                res = _wrap_decode_res_tsdframe_tensor(res, time_l=np.asarray(time_l))
        return res

    raise ValueError(f'Unsupported spk ndim={spk.ndim}; expected 2 or 3.')

def _decode_res_to_numpy(res):
    '''
    Convert JAX arrays to numpy arrays (recursively for lists / dicts).
    '''
    if isinstance(res, dict):
        return {k: _decode_res_to_numpy(v) for k, v in res.items()}
    if isinstance(res, list):
        return [_decode_res_to_numpy(v) for v in res]
    try:
        return np.asarray(res)
    except Exception:
        return res

def _wrap_decode_res_tsdframe_matrix(res, time_coord):
    '''
    Wrap matrix decode outputs into pynapple time series containers when time is available.
    '''
    res2 = dict(res)
    for k in ['log_likelihood', 'log_posterior', 'posterior']:
        if k in res2 and np.ndim(res2[k]) == 2:
            arr = np.asarray(res2[k])
            cols = np.arange(arr.shape[1])
            res2[k] = pynapple.TsdFrame(d=arr, t=np.asarray(time_coord), columns=cols)
    if 'log_marginal_l' in res2 and np.ndim(res2['log_marginal_l']) == 1:
        res2['log_marginal_l'] = pynapple.Tsd(d=np.asarray(res2['log_marginal_l']), t=np.asarray(time_coord))
    return res2

def _wrap_decode_res_tsdframe_tensor(res, time_l):
    '''
    Wrap tensor decode outputs into TsdFrame/Tsd using time_l (valid-bin concat time).
    Also wraps per-trial outputs into list[TsdFrame]/list[Tsd] using starts/ends.
    '''
    res2 = dict(res)
    starts = np.asarray(res2.get('starts', []), dtype=int)
    ends = np.asarray(res2.get('ends', []), dtype=int)

    for k in ['log_likelihood_concat', 'log_posterior_concat', 'posterior_concat']:
        if k in res2 and np.ndim(res2[k]) == 2:
            arr = np.asarray(res2[k])
            cols = np.arange(arr.shape[1])
            res2[k] = pynapple.TsdFrame(d=arr, t=np.asarray(time_l), columns=cols)

    for k in ['log_likelihood', 'log_posterior', 'posterior']:
        if k in res2 and isinstance(res2[k], list) and len(res2[k]) and np.ndim(res2[k][0]) == 2:
            arr_l = [np.asarray(a) for a in res2[k]]
            tsdf_l = []
            for a, s, e in zip(arr_l, starts, ends):
                cols = np.arange(a.shape[1])
                tsdf_l.append(pynapple.TsdFrame(d=a, t=np.asarray(time_l)[s:e], columns=cols))
            res2[k] = tsdf_l

    if 'log_marginal_l_concat' in res2 and np.ndim(res2['log_marginal_l_concat']) == 1:
        res2['log_marginal_l_concat'] = pynapple.Tsd(d=np.asarray(res2['log_marginal_l_concat']), t=np.asarray(time_l))
    if 'log_marginal_l' in res2 and isinstance(res2['log_marginal_l'], list) and len(res2['log_marginal_l']):
        tsd_l = []
        for a, s, e in zip(res2['log_marginal_l'], starts, ends):
            tsd_l.append(pynapple.Tsd(d=np.asarray(a), t=np.asarray(time_l)[s:e]))
        res2['log_marginal_l'] = tsd_l

    return res2

def _decode_naive_bayes_matrix(spk, tuning, **kwargs):
    '''
    spk: (n_time, n_neuron)
    tuning: (n_label_bin, n_neuron)
    '''
    observation_model = kwargs.get('observation_model', 'poisson')
    n_time_per_chunk = int(kwargs.get('n_time_per_chunk', 10000))
    dt = float(kwargs.get('dt', 1.0))
    gain = float(kwargs.get('gain', 1.0))
    dt_l = kwargs.get('dt_l', None)
    if dt_l is None:
        dt_l = dt
    dt_l = dt_l * gain
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

    verbose = bool(kwargs.get('verbose', True))
    profile_chunks = bool(kwargs.get('profile_chunks', True))
    n_time_per_chunk = int(kwargs.get('n_time_per_chunk', 10000))

    if verbose:
        print(f"[_decode_naive_bayes_tensor] spk_tensor shape={tuple(spk_tensor.shape)}")
        print(f"[_decode_naive_bayes_tensor] spk_valid shape={tuple(spk_valid.shape)} (after mask)")

    dt = float(kwargs.get('dt', 1.0))
    gain = float(kwargs.get('gain', 1.0))
    dt_l = kwargs.get('dt_l', None)
    if dt_l is None:
        dt_l = dt
    if np.ndim(dt_l) > 0 and not np.isscalar(dt_l):
        dt_l = np.asarray(dt_l).reshape((n_trial * n_time,))[mask_flat]
    dt_l = dt_l * gain

    if not profile_chunks:
        kwargs2 = dict(kwargs)
        kwargs2['dt_l'] = dt_l
        res_mat = _decode_naive_bayes_matrix(spk_valid, tuning, **kwargs2)
    else:
        observation_model = kwargs.get('observation_model', 'poisson')
        ma_neuron = kwargs.get('ma_neuron', jnp.ones(n_neuron))
        ma_latent = kwargs.get('ma_latent', jnp.ones(tuning.shape[0]))
        noise_std = kwargs.get('noise_std', 1.0)

        if observation_model == 'gaussian':
            hyperparam = {'noise_std': noise_std}
        else:
            hyperparam = {}

        n_valid_time = int(spk_valid.shape[0])
        n_chunks = int(np.ceil(n_valid_time / n_time_per_chunk)) if n_valid_time else 0

        if verbose:
            print(f"[_decode_naive_bayes_tensor] chunking n_valid_time={n_valid_time} with n_time_per_chunk={n_time_per_chunk} => n_chunks={n_chunks}")

        ma_neuron_full = jnp.broadcast_to(ma_neuron, (n_valid_time, n_neuron))
        dt_l_full = dt_l
        if np.ndim(dt_l_full) == 0 or np.isscalar(dt_l_full):
            dt_l_full = jnp.broadcast_to(dt_l_full, (n_valid_time,))
        else:
            dt_l_full = jnp.asarray(dt_l_full)

        log_post_l = []
        log_marginal_l_l = []
        log_marginal_chunk_l = []
        ll_per_pos_l_l = []

        t0 = time.time()
        for chunk_i in range(n_chunks):
            sl = slice(chunk_i * n_time_per_chunk, (chunk_i + 1) * n_time_per_chunk)
            y_chunk = spk_valid[sl]
            ma_neuron_chunk = ma_neuron_full[sl]
            dt_l_chunk = dt_l_full[sl]

            tic = time.time()
            log_post, log_marginal_l_chunk, log_marginal, ll_per_pos_chunk = decoder.get_naive_bayes_ma(
                y_chunk,
                tuning,
                hyperparam,
                ma_neuron_chunk,
                ma_latent,
                dt_l=dt_l_chunk,
                observation_model=observation_model,
            )
            log_post.block_until_ready()
            toc = time.time()

            if verbose:
                chunk_len = int(y_chunk.shape[0])
                print(f"[_decode_naive_bayes_tensor] chunk {chunk_i+1}/{n_chunks}: time_idx={sl.start}:{min(sl.stop, n_valid_time)} n_time={chunk_len} dt={toc-tic:.3f}s")

            log_post_l.append(log_post)
            log_marginal_l_l.append(log_marginal_l_chunk)
            log_marginal_chunk_l.append(log_marginal)
            ll_per_pos_l_l.append(ll_per_pos_chunk)

        log_posterior = jnp.concatenate(log_post_l, axis=0) if log_post_l else jnp.zeros((0, tuning.shape[0]))
        log_marginal_l = jnp.concatenate(log_marginal_l_l, axis=0) if log_marginal_l_l else jnp.zeros((0,))
        log_marginal = jnp.sum(jnp.asarray(log_marginal_chunk_l)) if log_marginal_chunk_l else jnp.asarray(0.0)
        log_likelihood = jnp.concatenate(ll_per_pos_l_l, axis=0) if ll_per_pos_l_l else jnp.zeros((0, tuning.shape[0]))

        posterior = jnp.exp(log_posterior)

        res_mat = {
            'log_likelihood': log_likelihood,
            'log_posterior': log_posterior,
            'posterior': posterior,
            'log_marginal_l': log_marginal_l,
            'log_marginal': log_marginal,
        }

        if verbose:
            print(f"[_decode_naive_bayes_tensor] decode total dt={time.time()-t0:.3f}s")
            print(f"[_decode_naive_bayes_tensor] concat mat shape={tuple(res_mat['log_posterior'].shape)}")

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
        'log_likelihood_concat': res_mat['log_likelihood'],
        'log_posterior_concat': res_mat['log_posterior'],
        'posterior_concat': res_mat['posterior'],
        'log_marginal_l_concat': res_mat['log_marginal_l'],
        'log_marginal_concat': res_mat['log_marginal'],
        'n_time_per_trial': n_time_per_trial,
        'starts': starts,
        'ends': ends,
    }
    return res


def _wrap_label_results_xr_by_maze(res, flat_idx_to_coord, time_coord=None):
    '''
    Wrap label-related outputs into dict[str, xr.DataArray] keyed by maze, so each xr only
    contains label bins for that maze (no "other maze" bins).

    - Matrix output: returns dict[maze] -> DataArray (time, label_bin)
    - Tensor output: returns dict[maze] -> list[DataArray], one per trial (views from concat)
      Also keeps concatenated DataArrays under *_concat keys as dict[maze] -> DataArray.
    '''
    df = flat_idx_to_coord
    if not isinstance(df, pd.DataFrame):
        raise ValueError('flat_idx_to_coord must be a pandas DataFrame.')

    if 'maze' not in df.columns:
        raise ValueError("flat_idx_to_coord must include column 'maze'.")

    maze_l = pd.unique(df['maze'])
    label_cols_all = [c for c in df.columns if c != 'maze']

    def _default_time(n_time):
        return np.arange(int(n_time))

    def _make_label_coord(pos_idx, maze):
        if len(label_cols_all) == 0:
            return np.arange(len(pos_idx))

        df_maze = df.iloc[pos_idx]
        label_cols = [c for c in label_cols_all if df_maze[c].notna().any()]
        if len(label_cols) == 0:
            return np.arange(len(pos_idx))

        df_lab = df_maze.loc[:, label_cols].copy()
        for c in label_cols:
            if df_lab[c].dtype.kind in 'if':
                df_lab[c] = df_lab[c].fillna(-1)
            else:
                df_lab[c] = df_lab[c].fillna('_nan')
        return pd.MultiIndex.from_frame(df_lab)

    def _split_cols_by_maze(arr_2d):
        out = {}
        arr_2d = np.asarray(arr_2d)
        for maze in maze_l:
            pos_idx = np.where(np.asarray(df['maze']) == maze)[0]
            out[maze] = arr_2d[:, pos_idx]
        return out

    if 'log_likelihood_concat' in res:
        n_time = int(np.asarray(res['log_likelihood_concat']).shape[0])
        if time_coord is None:
            time_coord = _default_time(n_time)
        else:
            time_coord = np.asarray(time_coord)

        ll_by_maze = _split_cols_by_maze(res['log_likelihood_concat'])
        lp_by_maze = _split_cols_by_maze(res['log_posterior_concat'])
        post_by_maze = _split_cols_by_maze(res['posterior_concat'])

        starts = np.asarray(res['starts']).astype(int)
        ends = np.asarray(res['ends']).astype(int)

        res2 = dict(res)
        res2['maze_l'] = maze_l

        res2['log_likelihood_concat'] = {}
        res2['log_posterior_concat'] = {}
        res2['posterior_concat'] = {}
        res2['log_likelihood'] = {}
        res2['log_posterior'] = {}
        res2['posterior'] = {}

        for maze in maze_l:
            pos_idx = np.where(np.asarray(df['maze']) == maze)[0]
            label_coord = _make_label_coord(pos_idx, maze)

            ll_c = xr.DataArray(
                ll_by_maze[maze],
                dims=('time', 'label_bin'),
                coords={'time': time_coord, 'label_bin': label_coord},
            )
            lp_c = xr.DataArray(
                lp_by_maze[maze],
                dims=('time', 'label_bin'),
                coords={'time': time_coord, 'label_bin': label_coord},
            )
            post_c = xr.DataArray(
                post_by_maze[maze],
                dims=('time', 'label_bin'),
                coords={'time': time_coord, 'label_bin': label_coord},
            )

            res2['log_likelihood_concat'][maze] = ll_c
            res2['log_posterior_concat'][maze] = lp_c
            res2['posterior_concat'][maze] = post_c
            res2['log_likelihood'][maze] = [ll_c.isel(time=slice(s, e)) for s, e in zip(starts, ends)]
            res2['log_posterior'][maze] = [lp_c.isel(time=slice(s, e)) for s, e in zip(starts, ends)]
            res2['posterior'][maze] = [post_c.isel(time=slice(s, e)) for s, e in zip(starts, ends)]

        return res2

    n_time = int(np.asarray(res['log_likelihood']).shape[0])
    if time_coord is None:
        time_coord = _default_time(n_time)
    else:
        time_coord = np.asarray(time_coord)

    res2 = dict(res)
    ll_by_maze = _split_cols_by_maze(res['log_likelihood'])
    lp_by_maze = _split_cols_by_maze(res['log_posterior'])
    post_by_maze = _split_cols_by_maze(res['posterior'])

    res2['maze_l'] = maze_l
    res2['log_likelihood'] = {}
    res2['log_posterior'] = {}
    res2['posterior'] = {}
    for maze in maze_l:
        pos_idx = np.where(np.asarray(df['maze']) == maze)[0]
        label_coord = _make_label_coord(pos_idx, maze)
        res2['log_likelihood'][maze] = xr.DataArray(
            ll_by_maze[maze],
            dims=('time', 'label_bin'),
            coords={'time': time_coord, 'label_bin': label_coord},
        )
        res2['log_posterior'][maze] = xr.DataArray(
            lp_by_maze[maze],
            dims=('time', 'label_bin'),
            coords={'time': time_coord, 'label_bin': label_coord},
        )
        res2['posterior'][maze] = xr.DataArray(
            post_by_maze[maze],
            dims=('time', 'label_bin'),
            coords={'time': time_coord, 'label_bin': label_coord},
        )
    return res2

