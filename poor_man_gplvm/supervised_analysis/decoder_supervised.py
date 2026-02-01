'''
wrappers of the decoder.py, specifically for supervised analysis
'''

import numpy as np
import pandas as pd
import xarray as xr
import time
import jax
import jax.numpy as jnp
import jax.scipy as jscipy

import pynapple as nap
import poor_man_gplvm.decoder as decoder
import poor_man_gplvm.decoder_trial as decoder_trial
import poor_man_gplvm.supervised_analysis.get_tuning_supervised as get_tuning_supervised

def _parse_event_index_per_bin(event_index_per_bin):
    '''
    Parse an (n_time,) event index vector into contiguous event segments.

    Returns dict with:
    - event_l: (n_event,) event ids in appearance order
    - starts: (n_event,) start indices into time-concat arrays
    - ends: (n_event,) end indices (exclusive)
    - n_time_per_event: (n_event,) segment lengths
    '''
    e = np.asarray(event_index_per_bin)
    if e.size == 0:
        return {
            'event_l': np.asarray([], dtype=e.dtype),
            'starts': np.asarray([], dtype=int),
            'ends': np.asarray([], dtype=int),
            'n_time_per_event': np.asarray([], dtype=int),
        }

    is_new = np.empty((e.size,), dtype=bool)
    is_new[0] = True
    is_new[1:] = e[1:] != e[:-1]
    starts = np.where(is_new)[0].astype(int)
    ends = np.concatenate([starts[1:], np.asarray([e.size], dtype=int)])
    event_l = e[starts]
    n_time_per_event = (ends - starts).astype(int)
    return {
        'event_l': np.asarray(event_l),
        'starts': starts,
        'ends': ends,
        'n_time_per_event': n_time_per_event,
    }

def decode_naive_bayes(
    spk,
    tuning,
    tensor_pad_mask=None,
    flat_idx_to_coord=None,
    event_index_per_bin=None,
    return_per_event=False,
    dt=1.0,
    gain=1.0,
    time_l=None,
    **kwargs
):
    '''
    Wrapper around `poor_man_gplvm.decoder.get_naive_bayes_ma_chunk` with a simple supervised API.

    Inputs
    - spk: (n_time, n_neuron) or padded (n_trial, n_time, n_neuron)
    - tuning: (n_label_bin, n_neuron) (typically `tuning_res['tuning_flat']`)
    - tensor_pad_mask: (n_trial, n_time, 1) bool, True for valid time bins (from `trial_analysis.bin_spike_train_to_trial_based`)
    - flat_idx_to_coord: pd.DataFrame indexed by flat_idx, columns ['maze', ...label_dims...] from `get_tuning_supervised.get_tuning`
    - event_index_per_bin: optional (n_time,) int-like. Only for matrix spk input; indicates which event each time bin
      belongs to (typically from `trial_analysis.bin_spike_train_to_trial_based` for concatenated events).
    - return_per_event: bool. Only for matrix spk input + event_index_per_bin. If False (default), only returns
      `log_marginal_per_event` (plus parsing meta). If True, also returns per-event arrays for posterior/likelihood.

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

# matrix decoding with event grouping -> xarray + per-event outputs
res_ev = dec_sup.decode_naive_bayes(
    spk_mat,
    tuning_res['tuning_flat'],
    event_index_per_bin=event_index_per_bin,
    return_per_event=True,
    time_l=time_l,  # optional, used when spk_mat has no .t
)

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

        if event_index_per_bin is not None:
            event_index_per_bin = np.asarray(event_index_per_bin)
            res['event_index_per_bin'] = event_index_per_bin
            parsing_res = _parse_event_index_per_bin(event_index_per_bin)
            res.update(parsing_res)
            res['return_per_event'] = bool(return_per_event)

            starts = np.asarray(res['starts']).astype(int)
            ends = np.asarray(res['ends']).astype(int)
            if 'log_marginal_l' in res and np.ndim(res['log_marginal_l']) == 1:
                arr = np.asarray(res['log_marginal_l'])
                res['log_marginal_per_event'] = np.asarray([np.sum(arr[s:e]) for s, e in zip(starts, ends)])

                if bool(return_per_event):
                    res['log_marginal_l_per_event'] = [arr[s:e] for s, e in zip(starts, ends)]

            if bool(return_per_event):
                for k in ['log_likelihood', 'log_posterior', 'posterior']:
                    if k in res and np.ndim(res[k]) == 2:
                        arr = np.asarray(res[k])
                        res[f'{k}_per_event'] = [arr[s:e] for s, e in zip(starts, ends)]

        if flat_idx_to_coord is not None:
            res = _wrap_label_results_xr_by_maze(
                res,
                flat_idx_to_coord=flat_idx_to_coord,
                time_coord=time_coord,
                event_index_per_bin=event_index_per_bin,
                return_per_event=bool(return_per_event),
            )
        else:
            if event_index_per_bin is not None:
                res = _wrap_decode_res_xr_matrix(
                    res,
                    time_coord=time_coord,
                    event_index_per_bin=event_index_per_bin,
                    return_per_event=bool(return_per_event),
                )
            elif time_coord is not None:
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
    if isinstance(res, jax.Array):
        return np.asarray(jax.device_get(res))
    if isinstance(res, (np.ndarray, np.number)):
        return np.asarray(res)
    if isinstance(res, (int, float, bool, str, type(None))):
        return res
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
            res2[k] = nap.TsdFrame(d=arr, t=np.asarray(time_coord), columns=cols)
    if 'log_marginal_l' in res2 and np.ndim(res2['log_marginal_l']) == 1:
        res2['log_marginal_l'] = nap.Tsd(d=np.asarray(res2['log_marginal_l']), t=np.asarray(time_coord))
    return res2

def _wrap_decode_res_xr_matrix(res, time_coord, event_index_per_bin=None, return_per_event=False):
    '''
    Wrap matrix decode outputs into xarray DataArray(s), optionally with an event coord.
    '''
    res2 = dict(res)
    time_coord = np.asarray(time_coord) if time_coord is not None else np.arange(int(np.asarray(res2['log_likelihood']).shape[0]))
    coords_time = {'time': time_coord}
    if event_index_per_bin is not None:
        coords_time['event_index_per_bin'] = ('time', np.asarray(event_index_per_bin))

    for k in ['log_likelihood', 'log_posterior', 'posterior']:
        if k in res2 and np.ndim(res2[k]) == 2:
            arr = np.asarray(res2[k])
            res2[k] = xr.DataArray(
                arr,
                dims=('time', 'label_bin'),
                coords=dict(coords_time, label_bin=np.arange(arr.shape[1])),
            )

    if 'log_marginal_l' in res2 and np.ndim(res2['log_marginal_l']) == 1:
        res2['log_marginal_l'] = xr.DataArray(
            np.asarray(res2['log_marginal_l']),
            dims=('time',),
            coords=coords_time,
        )

    if bool(return_per_event) and ('starts' in res2) and ('ends' in res2):
        starts = np.asarray(res2['starts']).astype(int)
        ends = np.asarray(res2['ends']).astype(int)
        for k in ['log_likelihood', 'log_posterior', 'posterior']:
            if k in res2 and isinstance(res2[k], xr.DataArray):
                res2[f'{k}_per_event'] = [res2[k].isel(time=slice(s, e)) for s, e in zip(starts, ends)]
        if 'log_marginal_l' in res2 and isinstance(res2['log_marginal_l'], xr.DataArray):
            res2['log_marginal_l_per_event'] = [res2['log_marginal_l'].isel(time=slice(s, e)) for s, e in zip(starts, ends)]
    return res2

def _wrap_decode_res_tsdframe_tensor(res, time_l):
    '''
    Wrap tensor decode outputs into TsdFrame/Tsd using time_l (valid-bin concat time).
    Also wraps per-trial outputs into list[TsdFrame]/list[Tsd] using starts/ends.
    '''
    res2 = dict(res)
    starts = np.asarray(res2.get('starts', []), dtype=int)
    ends = np.asarray(res2.get('ends', []), dtype=int)

    for k in ['log_likelihood', 'log_posterior', 'posterior']:
        if k in res2 and np.ndim(res2[k]) == 2:
            arr = np.asarray(res2[k])
            cols = np.arange(arr.shape[1])
            res2[k] = nap.TsdFrame(d=arr, t=np.asarray(time_l), columns=cols)

    for k in ['log_likelihood_per_event', 'log_posterior_per_event', 'posterior_per_event']:
        if k in res2 and isinstance(res2[k], list) and len(res2[k]) and np.ndim(res2[k][0]) == 2:
            arr_l = [np.asarray(a) for a in res2[k]]
            tsdf_l = []
            for a, s, e in zip(arr_l, starts, ends):
                cols = np.arange(a.shape[1])
                tsdf_l.append(nap.TsdFrame(d=a, t=np.asarray(time_l)[s:e], columns=cols))
            res2[k] = tsdf_l

    if 'log_marginal_l' in res2 and np.ndim(res2['log_marginal_l']) == 1:
        res2['log_marginal_l'] = nap.Tsd(d=np.asarray(res2['log_marginal_l']), t=np.asarray(time_l))
    if 'log_marginal_l_per_event' in res2 and isinstance(res2['log_marginal_l_per_event'], list) and len(res2['log_marginal_l_per_event']):
        tsd_l = []
        for a, s, e in zip(res2['log_marginal_l_per_event'], starts, ends):
            tsd_l.append(nap.Tsd(d=np.asarray(a), t=np.asarray(time_l)[s:e]))
        res2['log_marginal_l_per_event'] = tsd_l

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

    Returns a dict with:
    - time-concat arrays under keys:
      `log_likelihood`, `log_posterior`, `posterior`, `log_marginal_l`, `log_marginal`
    - per-event (per-trial) lists under keys:
      `log_likelihood_per_event`, `log_posterior_per_event`, `posterior_per_event`, `log_marginal_l_per_event`
    - parsing meta:
      `n_time_per_event`, `starts`, `ends`
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

    log_likelihood_per_event = _split_by_trial(res_mat['log_likelihood'])
    log_posterior_per_event = _split_by_trial(res_mat['log_posterior'])
    posterior_per_event = _split_by_trial(res_mat['posterior'])
    log_marginal_l_per_event = _split_1d(res_mat['log_marginal_l'])
    log_marginal_per_event = jnp.asarray([jnp.sum(x) for x in log_marginal_l_per_event])

    res = {
        # time-concat outputs
        'log_likelihood': res_mat['log_likelihood'],
        'log_posterior': res_mat['log_posterior'],
        'posterior': res_mat['posterior'],
        'log_marginal_l': res_mat['log_marginal_l'],
        'log_marginal': res_mat['log_marginal'],
        # per-event outputs
        'log_likelihood_per_event': log_likelihood_per_event,
        'log_posterior_per_event': log_posterior_per_event,
        'posterior_per_event': posterior_per_event,
        'log_marginal_l_per_event': log_marginal_l_per_event,
        'log_marginal_per_event': log_marginal_per_event,
        # parsing meta
        'n_time_per_event': n_time_per_trial,
        'starts': starts,
        'ends': ends,
    }
    return res


def _wrap_label_results_xr_by_maze(res, flat_idx_to_coord, time_coord=None, event_index_per_bin=None, return_per_event=False):
    '''
    Wrap label-related outputs into dict[str, xr.DataArray] keyed by maze, so each xr only
    contains label bins for that maze (no "other maze" bins).

    - Matrix output: returns dict[maze] -> DataArray (time, label_bin)
    - Tensor output: returns dict[maze] -> DataArray (time concat) and dict[maze] -> list[DataArray] under *_per_event.
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
    if bool(return_per_event) and ('starts' in res2) and ('ends' in res2):
        res2['log_likelihood_per_event'] = {}
        res2['log_posterior_per_event'] = {}
        res2['posterior_per_event'] = {}

    coords_time = {'time': time_coord}
    if event_index_per_bin is not None:
        coords_time['event_index_per_bin'] = ('time', np.asarray(event_index_per_bin))

    for maze in maze_l:
        pos_idx = np.where(np.asarray(df['maze']) == maze)[0]
        label_coord = _make_label_coord(pos_idx, maze)
        ll_c = xr.DataArray(
            ll_by_maze[maze],
            dims=('time', 'label_bin'),
            coords=dict(coords_time, label_bin=label_coord),
        )
        lp_c = xr.DataArray(
            lp_by_maze[maze],
            dims=('time', 'label_bin'),
            coords=dict(coords_time, label_bin=label_coord),
        )
        post_c = xr.DataArray(
            post_by_maze[maze],
            dims=('time', 'label_bin'),
            coords=dict(coords_time, label_bin=label_coord),
        )
        res2['log_likelihood'][maze] = ll_c
        res2['log_posterior'][maze] = lp_c
        res2['posterior'][maze] = post_c

        if bool(return_per_event) and ('starts' in res2) and ('ends' in res2):
            starts = np.asarray(res2['starts']).astype(int)
            ends = np.asarray(res2['ends']).astype(int)
            res2['log_likelihood_per_event'][maze] = [ll_c.isel(time=slice(s, e)) for s, e in zip(starts, ends)]
            res2['log_posterior_per_event'][maze] = [lp_c.isel(time=slice(s, e)) for s, e in zip(starts, ends)]
            res2['posterior_per_event'][maze] = [post_c.isel(time=slice(s, e)) for s, e in zip(starts, ends)]
    return res2


def _wrap_label_results_xr_by_maze_dynamics(res, flat_idx_to_coord, time_coord=None, event_index_per_bin=None, return_per_event=False):
    '''
    Maze-aware xarray wrapping for dynamics decoding outputs.

    - Splits latent dimension by maze (from flat_idx_to_coord['maze'])
    - Wraps arrays with latent dim into DataArray using a MultiIndex label_bin coord (same as decode_naive_bayes wrapper)

    Expected keys (if present):
    - (time, n_latent): 'log_likelihood', 'log_posterior_latent_marg', 'posterior_latent_marg'
    - (time, n_dyn, n_latent): 'log_posterior_all', 'posterior_all'
    - per-event lists under *_per_event (if return_per_event True in matrix mode, or tensor mode)
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

    def _make_label_coord(pos_idx):
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

    def _split_latent(arr):
        out = {}
        arr = np.asarray(arr)
        for maze in maze_l:
            pos_idx = np.where(np.asarray(df['maze']) == maze)[0]
            out[maze] = arr[:, pos_idx]
        return out

    def _split_dyn_latent(arr):
        out = {}
        arr = np.asarray(arr)
        for maze in maze_l:
            pos_idx = np.where(np.asarray(df['maze']) == maze)[0]
            out[maze] = arr[:, :, pos_idx]
        return out

    # time coordinate
    n_time = None
    for k in ['log_likelihood', 'posterior_latent_marg', 'log_posterior_all', 'posterior_all']:
        if k in res and np.ndim(res[k]) >= 1:
            n_time = int(np.asarray(res[k]).shape[0])
            break
    if n_time is None:
        return dict(res)

    if time_coord is None:
        time_coord = _default_time(n_time)
    else:
        time_coord = np.asarray(time_coord)

    coords_time = {'time': time_coord}
    if event_index_per_bin is not None:
        coords_time['event_index_per_bin'] = ('time', np.asarray(event_index_per_bin))

    res2 = dict(res)

    # initialize dict outputs
    for k in ['log_likelihood', 'log_posterior_latent_marg', 'posterior_latent_marg', 'log_posterior_all', 'posterior_all']:
        if k in res2:
            res2[k] = {}
    if bool(return_per_event):
        for k in ['log_likelihood_per_event', 'log_posterior_latent_marg_per_event', 'posterior_latent_marg_per_event',
                  'log_posterior_all_per_event', 'posterior_all_per_event']:
            if k in res2:
                res2[k] = {}

    # wrap
    if 'log_likelihood' in res:
        ll_by_maze = _split_latent(res['log_likelihood'])
    if 'log_posterior_latent_marg' in res:
        lpm_by_maze = _split_latent(res['log_posterior_latent_marg'])
    if 'posterior_latent_marg' in res:
        pm_by_maze = _split_latent(res['posterior_latent_marg'])
    if 'log_posterior_all' in res:
        lpa_by_maze = _split_dyn_latent(res['log_posterior_all'])
    if 'posterior_all' in res:
        pa_by_maze = _split_dyn_latent(res['posterior_all'])

    for maze in maze_l:
        pos_idx = np.where(np.asarray(df['maze']) == maze)[0]
        label_coord = _make_label_coord(pos_idx)

        if 'log_likelihood' in res:
            res2['log_likelihood'][maze] = xr.DataArray(
                ll_by_maze[maze],
                dims=('time', 'label_bin'),
                coords=dict(coords_time, label_bin=label_coord),
            )
        if 'log_posterior_latent_marg' in res:
            res2['log_posterior_latent_marg'][maze] = xr.DataArray(
                lpm_by_maze[maze],
                dims=('time', 'label_bin'),
                coords=dict(coords_time, label_bin=label_coord),
            )
        if 'posterior_latent_marg' in res:
            res2['posterior_latent_marg'][maze] = xr.DataArray(
                pm_by_maze[maze],
                dims=('time', 'label_bin'),
                coords=dict(coords_time, label_bin=label_coord),
            )
        if 'log_posterior_all' in res:
            res2['log_posterior_all'][maze] = xr.DataArray(
                lpa_by_maze[maze],
                dims=('time', 'dyn', 'label_bin'),
                coords=dict(coords_time, dyn=np.arange(np.asarray(lpa_by_maze[maze]).shape[1]), label_bin=label_coord),
            )
        if 'posterior_all' in res:
            res2['posterior_all'][maze] = xr.DataArray(
                pa_by_maze[maze],
                dims=('time', 'dyn', 'label_bin'),
                coords=dict(coords_time, dyn=np.arange(np.asarray(pa_by_maze[maze]).shape[1]), label_bin=label_coord),
            )

        if bool(return_per_event) and ('starts' in res2) and ('ends' in res2):
            starts = np.asarray(res2['starts']).astype(int)
            ends = np.asarray(res2['ends']).astype(int)
            if 'log_likelihood_per_event' in res2 and 'log_likelihood' in res2:
                res2['log_likelihood_per_event'][maze] = [res2['log_likelihood'][maze].isel(time=slice(s, e)) for s, e in zip(starts, ends)]
            if 'log_posterior_latent_marg_per_event' in res2 and 'log_posterior_latent_marg' in res2:
                res2['log_posterior_latent_marg_per_event'][maze] = [res2['log_posterior_latent_marg'][maze].isel(time=slice(s, e)) for s, e in zip(starts, ends)]
            if 'posterior_latent_marg_per_event' in res2 and 'posterior_latent_marg' in res2:
                res2['posterior_latent_marg_per_event'][maze] = [res2['posterior_latent_marg'][maze].isel(time=slice(s, e)) for s, e in zip(starts, ends)]
            if 'log_posterior_all_per_event' in res2 and 'log_posterior_all' in res2:
                res2['log_posterior_all_per_event'][maze] = [res2['log_posterior_all'][maze].isel(time=slice(s, e)) for s, e in zip(starts, ends)]
            if 'posterior_all_per_event' in res2 and 'posterior_all' in res2:
                res2['posterior_all_per_event'][maze] = [res2['posterior_all'][maze].isel(time=slice(s, e)) for s, e in zip(starts, ends)]

    res2['maze_l'] = maze_l

    # dynamics marginal is not maze-specific; keep it as pynapple time series for convenience
    # (create once, then slice for per-event to avoid repeated TsdFrame construction)
    dyn_tsdf = None
    if 'posterior_dynamics_marg' in res2 and np.ndim(res2['posterior_dynamics_marg']) == 2:
        arr = np.asarray(res2['posterior_dynamics_marg'])
        n_dyn = int(arr.shape[1])
        cols = ['move', 'jump'] if n_dyn == 2 else np.arange(n_dyn)
        dyn_tsdf = nap.TsdFrame(d=arr, t=np.asarray(time_coord), columns=cols)
        res2['posterior_dynamics_marg'] = dyn_tsdf

    # if per-event requested upstream, prefer slicing the full pynapple object (fast) over re-wrapping numpy chunks
    if dyn_tsdf is not None and ('starts' in res2) and ('ends' in res2) and (bool(return_per_event) or ('posterior_dynamics_marg_per_event' in res)):
        starts = np.asarray(res2.get('starts', []), dtype=int)
        ends = np.asarray(res2.get('ends', []), dtype=int)
        if starts.size and ends.size:
            res2['posterior_dynamics_marg_per_event'] = [
                dyn_tsdf[s:e]
                for s, e in zip(starts, ends)
                if e > s
            ]

    return res2


def _wrap_decode_res_xr_matrix_dynamics(res, time_coord, event_index_per_bin=None, return_per_event=False):
    '''
    Wrap matrix dynamics outputs into xarray DataArray(s), optionally with an event coord.
    Uses a simple integer label_bin coord (0..n_latent-1).
    '''
    res2 = dict(res)
    time_coord = np.asarray(time_coord) if time_coord is not None else np.arange(int(np.asarray(res2['posterior_latent_marg']).shape[0]))
    coords_time = {'time': time_coord}
    if event_index_per_bin is not None:
        coords_time['event_index_per_bin'] = ('time', np.asarray(event_index_per_bin))

    if 'log_likelihood' in res2 and np.ndim(res2['log_likelihood']) == 2:
        arr = np.asarray(res2['log_likelihood'])
        res2['log_likelihood'] = xr.DataArray(arr, dims=('time', 'label_bin'), coords=dict(coords_time, label_bin=np.arange(arr.shape[1])))
    for k in ['log_posterior_latent_marg', 'posterior_latent_marg']:
        if k in res2 and np.ndim(res2[k]) == 2:
            arr = np.asarray(res2[k])
            res2[k] = xr.DataArray(arr, dims=('time', 'label_bin'), coords=dict(coords_time, label_bin=np.arange(arr.shape[1])))
    for k in ['log_posterior_all', 'posterior_all']:
        if k in res2 and np.ndim(res2[k]) == 3:
            arr = np.asarray(res2[k])
            res2[k] = xr.DataArray(arr, dims=('time', 'dyn', 'label_bin'),
                                   coords=dict(coords_time, dyn=np.arange(arr.shape[1]), label_bin=np.arange(arr.shape[2])))

    if 'log_marginal_l' in res2 and np.ndim(res2['log_marginal_l']) == 1:
        res2['log_marginal_l'] = xr.DataArray(np.asarray(res2['log_marginal_l']), dims=('time',), coords=coords_time)

    if bool(return_per_event) and ('starts' in res2) and ('ends' in res2):
        starts = np.asarray(res2['starts']).astype(int)
        ends = np.asarray(res2['ends']).astype(int)
        for k in ['log_likelihood', 'log_posterior_latent_marg', 'posterior_latent_marg', 'log_posterior_all', 'posterior_all', 'log_marginal_l']:
            if k in res2 and isinstance(res2[k], xr.DataArray):
                res2[f'{k}_per_event'] = [res2[k].isel(time=slice(s, e)) for s, e in zip(starts, ends)]
    return res2


def _wrap_decode_res_tsdframe_matrix_dynamics(res, time_coord):
    '''
    Wrap matrix dynamics outputs into pynapple time series containers when time is available.
    '''
    res2 = dict(res)
    t = np.asarray(time_coord)
    for k in ['log_likelihood', 'log_posterior_latent_marg', 'posterior_latent_marg']:
        if k in res2 and np.ndim(res2[k]) == 2:
            arr = np.asarray(res2[k])
            cols = np.arange(arr.shape[1])
            res2[k] = nap.TsdFrame(d=arr, t=t, columns=cols)
    if 'posterior_dynamics_marg' in res2 and np.ndim(res2['posterior_dynamics_marg']) == 2:
        arr = np.asarray(res2['posterior_dynamics_marg'])
        n_dyn = int(arr.shape[1])
        cols = ['move', 'jump'] if n_dyn == 2 else np.arange(n_dyn)
        res2['posterior_dynamics_marg'] = nap.TsdFrame(d=arr, t=t, columns=cols)
    if 'log_marginal_l' in res2 and np.ndim(res2['log_marginal_l']) == 1:
        res2['log_marginal_l'] = nap.Tsd(d=np.asarray(res2['log_marginal_l']), t=t)
    return res2


def _wrap_decode_res_tsdframe_tensor_dynamics(res, time_l):
    '''
    Wrap tensor (time-concat) dynamics outputs into TsdFrame/Tsd using time_l.
    Also wraps per-event outputs into list[TsdFrame]/list[Tsd] using starts/ends.
    '''
    res2 = dict(res)
    starts = np.asarray(res2.get('starts', []), dtype=int)
    ends = np.asarray(res2.get('ends', []), dtype=int)
    t = np.asarray(time_l)

    def _wrap_2d(arr_2d):
        arr_2d = np.asarray(arr_2d)
        cols = np.arange(arr_2d.shape[1])
        return nap.TsdFrame(d=arr_2d, t=t, columns=cols)

    def _wrap_1d(arr_1d):
        return nap.Tsd(d=np.asarray(arr_1d), t=t)

    # time-concat arrays
    for k in ['log_likelihood', 'log_posterior_latent_marg', 'posterior_latent_marg', 'posterior_dynamics_marg']:
        if k in res2 and np.ndim(res2[k]) == 2:
            res2[k] = _wrap_2d(res2[k])

    # per-event lists (trial-based): slice the full TsdFrame (avoid rebuilding TsdFrames in a loop)
    for k_full, k_per in [
        ('log_likelihood', 'log_likelihood_per_event'),
        ('log_posterior_latent_marg', 'log_posterior_latent_marg_per_event'),
        ('posterior_latent_marg', 'posterior_latent_marg_per_event'),
        ('posterior_dynamics_marg', 'posterior_dynamics_marg_per_event'),
    ]:
        if k_per in res2 and k_full in res2 and isinstance(res2[k_full], nap.TsdFrame) and starts.size and ends.size:
            tsdf = res2[k_full]
            res2[k_per] = [
                tsdf[s:e]
                for s, e in zip(starts, ends)
                if e > s
            ]

    # 3D arrays (dyn, latent): leave as numpy for now (harder to represent cleanly in nap)
    # but still wrap per-event lists if desired in analysis code.
    return res2


def _safe_row_normalize(mat):
    mat = np.asarray(mat, dtype=float)
    if mat.size == 0:
        return mat
    row_sums = mat.sum(axis=1, keepdims=True)
    zero_row = (row_sums[:, 0] == 0)
    row_sums[zero_row, :] = 1.0
    mat = mat / row_sums
    if np.any(zero_row):
        ii = np.where(zero_row)[0]
        mat[ii, :] = 0.0
        mat[ii, ii] = 1.0
    return mat


def _safe_log_prob(mat):
    mat = np.asarray(mat, dtype=float)
    with np.errstate(divide='ignore', invalid='ignore'):
        logp = np.log(mat)
    logp = np.where(np.isfinite(logp), logp, -1e20)
    return logp


def _get_log_transition_matrices_supervised(
    *,
    n_latent,
    coord_to_flat_idx=None,
    continuous_transition_movement_variance=1.0,
    p_move_to_jump=0.02,
    p_jump_to_move=0.02,
    custom_continuous_transition_kernel=None,
):
    '''
    Build log transition kernels for supervised JumpLVM-style decoding.

    Returns dict with:
    - latent_transition_kernel_l: (2, n_latent, n_latent) [move, jump]
    - log_latent_transition_kernel_l: (2, n_latent, n_latent)
    - dynamics_transition_kernel: (2, 2)
    - log_dynamics_transition_kernel: (2, 2)
    '''
    n_latent = int(n_latent)
    if n_latent < 0:
        raise ValueError(f'n_latent must be >=0, got {n_latent}')

    if custom_continuous_transition_kernel is not None:
        K_move = np.asarray(custom_continuous_transition_kernel, dtype=float)
    else:
        if coord_to_flat_idx is None:
            raise ValueError('coord_to_flat_idx is required when custom_continuous_transition_kernel is None.')
        K_move = get_latent_transition_kernel_multi_maze(
            coord_to_flat_idx,
            continuous_transition_movement_variance=continuous_transition_movement_variance,
        )

    if K_move.shape != (n_latent, n_latent):
        raise ValueError(f'continuous transition kernel shape {K_move.shape} != (n_latent, n_latent)=({n_latent}, {n_latent})')

    K_move = _safe_row_normalize(K_move)

    if n_latent == 0:
        K_jump = np.zeros((0, 0), dtype=float)
    else:
        K_jump = np.full((n_latent, n_latent), 1.0 / float(n_latent), dtype=float)

    latent_transition_kernel_l = np.stack([K_move, K_jump], axis=0)
    log_latent_transition_kernel_l = np.stack([_safe_log_prob(K_move), _safe_log_prob(K_jump)], axis=0)

    dyn_mat = np.array(
        [
            [1.0 - float(p_move_to_jump), float(p_move_to_jump)],
            [float(p_jump_to_move), 1.0 - float(p_jump_to_move)],
        ],
        dtype=float,
    )
    dyn_mat = _safe_row_normalize(dyn_mat)
    log_dyn_mat = _safe_log_prob(dyn_mat)

    return {
        'latent_transition_kernel_l': latent_transition_kernel_l,
        'log_latent_transition_kernel_l': log_latent_transition_kernel_l,
        'dynamics_transition_kernel': dyn_mat,
        'log_dynamics_transition_kernel': log_dyn_mat,
    }


def decode_with_dynamics(
    spk,
    tuning,
    tensor_pad_mask=None,
    coord_to_flat_idx=None,
    flat_idx_to_coord=None,
    event_index_per_bin=None,
    return_per_event=False,
    dt=1.0,
    gain=1.0,
    time_l=None,
    continuous_transition_movement_variance=1., # scalar, array, or {maze:}
    p_move_to_jump=0.02,
    p_jump_to_move=0.02,
    custom_continuous_transition_kernel=None,
    **kwargs
):
    '''
    State-space decoder (latent + dynamics) as in JumpLVM core, but with supervised tuning.

    Inputs
    - spk: (n_time, n_neuron) or padded tensor (n_trial, T_max, n_neuron)
    - tuning: (n_latent, n_neuron) firing rate in Hz (typically `tuning_res['tuning_flat']`)
    - tensor_pad_mask: required for tensor input, (n_trial, T_max, 1) bool
    - coord_to_flat_idx: (optional but usually needed) from `get_tuning_supervised.get_tuning` output.
      - single-maze: pd.Series mapping coord -> flat idx
      - multi-maze: dict[maze] -> pd.Series mapping coord -> global flat idx (concatenated tuning order)
      Needed to build the continuous (move) transition kernel unless `custom_continuous_transition_kernel` is provided.
    - flat_idx_to_coord: optional pd.DataFrame (index=flat_idx, includes column 'maze') for maze-aware xarray wrapping.
    - event_index_per_bin: matrix-mode only, (n_time,) event id per time bin (for per-event slicing).
    - return_per_event: matrix-mode only; if True, also returns *_per_event lists for time-indexed arrays.

    Transition params
    - continuous_transition_movement_variance: controls Gaussian smoothing width (variance in label units^2) for the move kernel
    - p_move_to_jump / p_jump_to_move: define the 2x2 dynamics transition matrix
    - custom_continuous_transition_kernel: optional global (n_latent, n_latent) transition matrix (row-normalized internally)

    Notes
    - Matrix mode uses `decoder.smooth_all_step_combined_ma_chunk` (supports poisson/gaussian).
    - Tensor mode uses `decoder_trial.decode_trials_padded_vmapped` (poisson only).
    - Tensor mode masking: supports `tensor_pad_mask` (valid bins) + `ma_latent`, and only a global per-neuron mask
      (no spatiotemporal neuron mask per trial/time bin).

    Cluster Jupyter examples

```python
import poor_man_gplvm.supervised_analysis.decoder_supervised as dec_sup

# matrix mode
res = dec_sup.decode_with_dynamics(
    spk_mat,
    tuning_res['tuning_flat'],
    coord_to_flat_idx=tuning_res['coord_to_flat_idx'],
    flat_idx_to_coord=tuning_res.get('flat_idx_to_coord', None),
    dt=0.02,
    gain=1.0,
    n_time_per_chunk=20000,
    likelihood_scale=1.0,
)

# tensor mode
res_t = dec_sup.decode_with_dynamics(
    spk_tensor,
    tuning_res['tuning_flat'],
    tensor_pad_mask=tensor_pad_mask,
    coord_to_flat_idx=tuning_res['coord_to_flat_idx'],
    dt=0.02,
    gain=1.0,
    n_trial_per_chunk=200,
    prior_magnifier=1.0,
    time_l=time_l,  # optional, valid-bin concat time
)
```
    '''
    time_coord = None
    if hasattr(spk, 't'):
        time_coord = np.asarray(spk.t)

    if time_l is None:
        time_l = kwargs.get('time_l', None)

    # allow passing coord_to_flat_idx via kwargs for backwards compatibility
    if coord_to_flat_idx is None:
        coord_to_flat_idx = kwargs.get('coord_to_flat_idx', None)

    spk_j = jnp.asarray(spk)
    tuning_j = jnp.asarray(tuning)

    observation_model = kwargs.get('observation_model', 'poisson')
    noise_std = float(kwargs.get('noise_std', 1.0))
    likelihood_scale = float(kwargs.get('likelihood_scale', 1.0))
    n_time_per_chunk = int(kwargs.get('n_time_per_chunk', 10000))

    # transitions
    n_latent = int(tuning_j.shape[0])
    trans = _get_log_transition_matrices_supervised(
        n_latent=n_latent,
        coord_to_flat_idx=coord_to_flat_idx,
        continuous_transition_movement_variance=continuous_transition_movement_variance,
        p_move_to_jump=p_move_to_jump,
        p_jump_to_move=p_jump_to_move,
        custom_continuous_transition_kernel=custom_continuous_transition_kernel,
    )
    log_latent_transition_kernel_l = jnp.asarray(trans['log_latent_transition_kernel_l'])
    log_dynamics_transition_kernel = jnp.asarray(trans['log_dynamics_transition_kernel'])

    # masks
    ma_neuron = kwargs.get('ma_neuron', None)
    if ma_neuron is None:
        ma_neuron = jnp.ones((spk_j.shape[-1],), dtype=jnp.float32)
    else:
        ma_neuron = jnp.asarray(ma_neuron)
    ma_latent = kwargs.get('ma_latent', None)
    if ma_latent is None:
        ma_latent = jnp.ones((n_latent,), dtype=bool)
    else:
        ma_latent = jnp.asarray(ma_latent).astype(bool)

    if spk_j.ndim == 2:
        # matrix mode: use decoder.smooth_all_step_combined_ma_chunk
        if observation_model == 'gaussian':
            hyperparam = {'noise_std': noise_std}
        else:
            hyperparam = {}

        # decoder.py assumes dt=1 inside likelihood; supply expected counts/bin directly
        tuning_eff = tuning_j * jnp.asarray(dt, dtype=jnp.float32) * jnp.asarray(gain, dtype=jnp.float32)

        (log_posterior_all,
         log_marginal_final,
         log_causal_posterior_all,
         log_one_step_predictive_marginals_all,
         log_accumulated_joint_total,
         log_likelihood_all) = decoder.smooth_all_step_combined_ma_chunk(
            spk_j,
            tuning_eff,
            hyperparam,
            log_latent_transition_kernel_l,
            log_dynamics_transition_kernel,
            ma_neuron,
            ma_latent=ma_latent,
            likelihood_scale=likelihood_scale,
            n_time_per_chunk=n_time_per_chunk,
            observation_model=observation_model,
        )

        posterior_all = jnp.exp(log_posterior_all)
        posterior_latent_marg = jnp.sum(posterior_all, axis=1)
        posterior_dynamics_marg = jnp.sum(posterior_all, axis=2)
        log_posterior_latent_marg = jscipy.special.logsumexp(log_posterior_all, axis=1)

        res = {
            'log_likelihood': log_likelihood_all,
            'log_posterior_all': log_posterior_all,
            'posterior_all': posterior_all,
            'log_posterior_latent_marg': log_posterior_latent_marg,
            'posterior_latent_marg': posterior_latent_marg,
            'posterior_dynamics_marg': posterior_dynamics_marg,
            'log_marginal_l': log_one_step_predictive_marginals_all,
            'log_marginal': log_marginal_final,
            'log_causal_posterior_all': log_causal_posterior_all,
            'log_latent_transition_kernel_l': log_latent_transition_kernel_l,
            'log_dynamics_transition_kernel': log_dynamics_transition_kernel,
        }

        if log_accumulated_joint_total is not None:
            res.update(decoder.compute_transition_posterior_prob(log_accumulated_joint_total))

        res = _decode_res_to_numpy(res)

        if time_coord is None and time_l is not None:
            time_coord = np.asarray(time_l)

        # event grouping (matrix only)
        if event_index_per_bin is not None:
            event_index_per_bin = np.asarray(event_index_per_bin)
            res['event_index_per_bin'] = event_index_per_bin
            parsing_res = _parse_event_index_per_bin(event_index_per_bin)
            res.update(parsing_res)
            res['return_per_event'] = bool(return_per_event)

            starts = np.asarray(res['starts']).astype(int)
            ends = np.asarray(res['ends']).astype(int)
            if 'log_marginal_l' in res and np.ndim(res['log_marginal_l']) == 1:
                arr = np.asarray(res['log_marginal_l'])
                res['log_marginal_per_event'] = np.asarray([np.sum(arr[s:e]) for s, e in zip(starts, ends)])
                if bool(return_per_event):
                    res['log_marginal_l_per_event'] = [arr[s:e] for s, e in zip(starts, ends)]

            if bool(return_per_event):
                n_time = int(np.asarray(spk_j).shape[0])
                for k in [
                    'log_likelihood',
                    'log_posterior_all',
                    'posterior_all',
                    'log_posterior_latent_marg',
                    'posterior_latent_marg',
                    'posterior_dynamics_marg',
                ]:
                    if k in res and isinstance(res[k], np.ndarray) and res[k].ndim >= 1 and res[k].shape[0] == n_time:
                        arr = np.asarray(res[k])
                        res[f'{k}_per_event'] = [arr[s:e] for s, e in zip(starts, ends)]

        # wrapping
        if flat_idx_to_coord is not None:
            res = _wrap_label_results_xr_by_maze_dynamics(
                res,
                flat_idx_to_coord=flat_idx_to_coord,
                time_coord=time_coord,
                event_index_per_bin=event_index_per_bin,
                return_per_event=bool(return_per_event),
            )
        else:
            if event_index_per_bin is not None:
                res = _wrap_decode_res_xr_matrix_dynamics(
                    res,
                    time_coord=time_coord,
                    event_index_per_bin=event_index_per_bin,
                    return_per_event=bool(return_per_event),
                )
            elif time_coord is not None:
                res = _wrap_decode_res_tsdframe_matrix_dynamics(res, time_coord=time_coord)
        return res

    if spk_j.ndim == 3:
        # tensor mode: use decoder_trial.decode_trials_padded_vmapped
        if tensor_pad_mask is None:
            raise ValueError('tensor input requires tensor_pad_mask (n_trial, n_time, 1).')
        if observation_model != 'poisson':
            raise ValueError('tensor dynamics decoder currently supports observation_model="poisson" only (decoder_trial backend).')

        prior_magnifier = float(kwargs.get('prior_magnifier', kwargs.get('likelihood_scale', 1.0)))
        n_trial_per_chunk = int(kwargs.get('n_trial_per_chunk', 400))

        # reduce ma_neuron to (n_neuron,) for decoder_trial
        ma_neuron_np = np.asarray(_decode_res_to_numpy(ma_neuron))
        if ma_neuron_np.ndim > 1:
            ma_neuron_np = np.any(ma_neuron_np, axis=tuple(range(ma_neuron_np.ndim - 1)))
        neuron_mask = ma_neuron_np.astype(bool)

        res_trial = decoder_trial.decode_trials_padded_vmapped(
            spk_j,
            tuning_j,
            log_latent_transition_kernel_l,
            log_dynamics_transition_kernel,
            tensor_pad_mask=tensor_pad_mask,
            dt=float(dt),
            gain=float(gain),
            neuron_mask=neuron_mask,
            ma_latent=ma_latent.astype(bool),
            n_trial_per_chunk=n_trial_per_chunk,
            prior_magnifier=prior_magnifier,
            return_numpy=False,
        )

        # time-concat (valid bins) arrays
        tensor_pad_mask_np = np.asarray(_decode_res_to_numpy(tensor_pad_mask)).astype(bool)
        valid_mask = tensor_pad_mask_np[..., 0]
        n_trial, t_max = valid_mask.shape
        trial_lengths = valid_mask.sum(axis=1).astype(int)
        ends = np.cumsum(trial_lengths)
        starts = np.concatenate([np.array([0], dtype=int), ends[:-1]])

        log_post_padded = np.asarray(_decode_res_to_numpy(res_trial['log_post_padded']))  # (n_trial, T, n_dyn, n_latent)
        n_dyn = int(log_post_padded.shape[2])
        log_post_flat = log_post_padded.reshape((n_trial * t_max, n_dyn, n_latent))
        mask_flat = valid_mask.reshape((n_trial * t_max,))
        log_posterior_all = log_post_flat[mask_flat]
        posterior_all = np.exp(log_posterior_all)
        posterior_latent_marg = posterior_all.sum(axis=1)
        posterior_dynamics_marg = posterior_all.sum(axis=2)
        log_posterior_latent_marg = np.asarray(jax.device_get(jscipy.special.logsumexp(jnp.asarray(log_posterior_all), axis=1)))

        # log likelihood for valid bins (match matrix semantics: use tuning_eff as expected counts/bin)
        tuning_eff = tuning_j * jnp.asarray(dt, dtype=jnp.float32) * jnp.asarray(gain, dtype=jnp.float32)
        spk_flat = np.asarray(_decode_res_to_numpy(spk_j)).reshape((n_trial * t_max, int(spk_j.shape[-1])))
        spk_valid = jnp.asarray(spk_flat[mask_flat])
        ma_neuron_valid = jnp.ones_like(spk_valid) * jnp.asarray(neuron_mask, dtype=jnp.float32)[None, :]
        hyperparam = {}
        log_likelihood = np.asarray(jax.device_get(decoder.get_loglikelihood_ma_all(
            spk_valid,
            tuning_eff,
            hyperparam,
            ma_neuron_valid,
            ma_latent,
            observation_model='poisson',
        )))

        log_marginal_per_event = np.asarray(_decode_res_to_numpy(res_trial['log_marginal']))
        log_marginal = float(np.sum(log_marginal_per_event))

        def _split_by_trial(arr):
            return [arr[s:e] for s, e in zip(starts, ends)]

        res = {
            'log_likelihood': log_likelihood,
            'log_posterior_all': log_posterior_all,
            'posterior_all': posterior_all,
            'log_posterior_latent_marg': log_posterior_latent_marg,
            'posterior_latent_marg': posterior_latent_marg,
            'posterior_dynamics_marg': posterior_dynamics_marg,
            'log_marginal_per_event': log_marginal_per_event,
            'log_marginal': log_marginal,
            'n_time_per_event': trial_lengths,
            'starts': starts,
            'ends': ends,
            'tensor_pad_mask': tensor_pad_mask_np,
            'log_latent_transition_kernel_l': np.asarray(_decode_res_to_numpy(log_latent_transition_kernel_l)),
            'log_dynamics_transition_kernel': np.asarray(_decode_res_to_numpy(log_dynamics_transition_kernel)),
        }

        # per-event lists (trial-based)
        res['log_likelihood_per_event'] = _split_by_trial(res['log_likelihood'])
        res['log_posterior_all_per_event'] = _split_by_trial(res['log_posterior_all'])
        res['posterior_all_per_event'] = _split_by_trial(res['posterior_all'])
        res['log_posterior_latent_marg_per_event'] = _split_by_trial(res['log_posterior_latent_marg'])
        res['posterior_latent_marg_per_event'] = _split_by_trial(res['posterior_latent_marg'])
        res['posterior_dynamics_marg_per_event'] = _split_by_trial(res['posterior_dynamics_marg'])

        if time_l is not None:
            res['time_l'] = np.asarray(time_l)

        if flat_idx_to_coord is not None:
            res = _wrap_label_results_xr_by_maze_dynamics(res, flat_idx_to_coord=flat_idx_to_coord, time_coord=time_l)
        else:
            if time_l is not None:
                res = _wrap_decode_res_tsdframe_tensor_dynamics(res, time_l=np.asarray(time_l))
        return res

    raise ValueError(f'Unsupported spk ndim={spk_j.ndim}; expected 2 or 3.')

def get_latent_transition_kernel_multi_maze(coord_to_flat_idx,continuous_transition_movement_variance=1.):
    '''
    for creating transition kernel for the supervised analysis, especially can deal with multiple mazes
    '''
    def _as_dict(v):
        if isinstance(v, dict):
            return v
        return None

    def _to_maze_dict(v, maze_l):
        vd = _as_dict(v)
        if vd is not None:
            return vd
        return {k: v for k in maze_l}

    def _variance_to_smooth_std(var, n_dim):
        var = np.asarray(var) if not np.isscalar(var) else float(var)
        if np.ndim(var) == 0:
            std = np.sqrt(float(var))
            return np.repeat(std, n_dim)
        var = np.asarray(var).astype(float).ravel()
        if var.size == 1:
            return np.repeat(np.sqrt(var[0]), n_dim)
        if var.size != n_dim:
            raise ValueError(f'continuous_transition_movement_variance size={var.size} != n_dim={n_dim}')
        return np.sqrt(var)

    def _coords_to_full_flat_index(coord_tuples, centers_l):
        # coord_tuples: list[tuple] length n_keep, each tuple length n_dim
        # centers_l: list[np.ndarray] per dim, sorted unique centers
        n_dim = len(centers_l)
        shape = [len(c) for c in centers_l]
        stride = np.ones((n_dim,), dtype=int)
        for d in range(n_dim - 2, -1, -1):
            stride[d] = stride[d + 1] * shape[d + 1]

        flat_idx = np.zeros((len(coord_tuples),), dtype=int)
        for i, ct in enumerate(coord_tuples):
            s = 0
            for d in range(n_dim):
                c = centers_l[d]
                j = int(np.searchsorted(c, ct[d]))
                if j >= len(c) or c[j] != ct[d]:
                    raise ValueError(f'coord value not found in centers: dim={d} value={ct[d]}')
                s += j * stride[d]
            flat_idx[i] = s
        return flat_idx

    # accept single-maze Series or multi-maze dict[maze]->Series
    if isinstance(coord_to_flat_idx, pd.Series):
        coord_to_flat_idx = {'single': coord_to_flat_idx}
    if not isinstance(coord_to_flat_idx, dict):
        raise ValueError('coord_to_flat_idx must be a pd.Series or dict[str, pd.Series].')

    maze_l = list(coord_to_flat_idx.keys())
    if len(maze_l) == 0:
        return np.zeros((0, 0))

    # build per-maze transition kernels on the *valid* bins, then place into a global block-diagonal matrix
    var_d = _to_maze_dict(continuous_transition_movement_variance, maze_l)
    n_total = 0
    for k in maze_l:
        s = coord_to_flat_idx[k]
        if not isinstance(s, pd.Series):
            raise ValueError(f'coord_to_flat_idx[{k}] must be a pd.Series.')
        if s.size:
            n_total = max(n_total, int(np.max(np.asarray(s.values))) + 1)

    K_global = np.zeros((n_total, n_total), dtype=float)

    for k in maze_l:
        s = coord_to_flat_idx[k]
        if s.size == 0:
            continue

        if not isinstance(s.index, pd.MultiIndex):
            # single-dim: represent as 1-level MultiIndex for uniform handling
            s = pd.Series(s.values, index=pd.MultiIndex.from_arrays([s.index], names=[s.index.name]))

        # order states by global flat index (to match tuning_flat concatenation)
        order = np.argsort(np.asarray(s.values).astype(int))
        coord_tuples = [tuple(s.index[i]) for i in order]
        global_idx = np.asarray(s.values, dtype=int)[order]

        n_dim = int(s.index.nlevels)
        centers_l = [np.sort(np.unique(np.asarray(s.index.get_level_values(d), dtype=float))) for d in range(n_dim)]
        grid_shape = tuple(len(c) for c in centers_l)

        smooth_std = _variance_to_smooth_std(var_d[k], n_dim=n_dim)
        S_full = get_tuning_supervised.get_smoothing_matrix(
            bin_centers_l=centers_l,
            grid_shape=grid_shape,
            smooth_std=smooth_std,
        )

        keep_full_flat = _coords_to_full_flat_index(coord_tuples, centers_l)
        S_sub = S_full[np.ix_(keep_full_flat, keep_full_flat)]

        # row-normalize after subsetting (holes / dropped states break row sums)
        row_sums = S_sub.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1.0
        S_sub = S_sub / row_sums

        K_global[np.ix_(global_idx, global_idx)] = S_sub

    # final row-normalize (should already be ok, but keeps invariants)
    if K_global.size:
        row_sums = K_global.sum(axis=1, keepdims=True)
        zero_row = (row_sums[:, 0] == 0)
        row_sums[zero_row, :] = 1.0
        K_global = K_global / row_sums
        if np.any(zero_row):
            ii = np.where(zero_row)[0]
            K_global[ii, ii] = 1.0

    return K_global