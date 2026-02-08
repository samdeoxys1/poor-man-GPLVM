'''
Shuffle test for whether the decoding of some event is "significant"
Currently using neuron-id shuffle and circular shuffle
Support both the supervised and unsupervised decoding (both use tuning, only difference is where the tuning is from), 
Currently do the test on naive bayes marginal likelihood, since the state-space has a "built-in" test for dynamics
'''

import numpy as np
import tqdm
import os
import pandas as pd
import xarray as xr

import poor_man_gplvm.supervised_analysis.decoder_supervised as decoder_supervised


def _parse_event_index_per_bin(event_index_per_bin):
    e = np.asarray(event_index_per_bin)
    if e.size == 0:
        return {
            'event_l': np.asarray([], dtype=e.dtype),
            'starts': np.asarray([], dtype=int),
            'ends': np.asarray([], dtype=int),
        }

    is_new = np.empty((e.size,), dtype=bool)
    is_new[0] = True
    is_new[1:] = e[1:] != e[:-1]
    starts = np.where(is_new)[0].astype(int)
    ends = np.concatenate([starts[1:], np.asarray([e.size], dtype=int)])
    event_l = e[starts]
    return {
        'event_l': np.asarray(event_l),
        'starts': starts,
        'ends': ends,
    }

def _save_npz_dict(save_path, d):
    d2 = {}
    for k, v in d.items():
        if isinstance(v, dict):
            d2[k] = np.asarray(v, dtype=object)
        else:
            d2[k] = v
    np.savez(str(save_path), **d2)
    print(f'saved: {save_path}')

def _save_pickle(save_path, d):
    pd.to_pickle(d, str(save_path))
    print(f'saved: {save_path}')


def _load_pickle(save_path):
    out = pd.read_pickle(str(save_path))
    print(f'loaded: {save_path}')
    return out


def _resolve_save_paths(save_dir, save_fn):
    save_dir = str(save_dir)
    save_fn = str(save_fn)
    base_path = os.path.join(save_dir, save_fn)

    if base_path.endswith('.npz'):
        pkl_path = base_path[:-4] + '.pkl'
        npz_path = base_path
    elif base_path.endswith('.pkl'):
        pkl_path = base_path
        npz_path = base_path[:-4] + '.npz'
    else:
        pkl_path = base_path + '.pkl'
        npz_path = base_path + '.npz'

    return {
        'pkl_path': pkl_path,
        'npz_path': npz_path,
    }


def _load_npz_dict(save_path):
    with np.load(str(save_path), allow_pickle=True) as z:
        out = {k: z[k] for k in z.files}
    for k, v in list(out.items()):
        if isinstance(v, np.ndarray) and v.dtype == object and v.shape == ():
            out[k] = v.item()
    print(f'loaded: {save_path}')
    return out


def create_neuron_id_shuffle_one_event(spk_mat, n_shuffle=100, seed=0):
    '''
    given spk_mat (n_time, n_neuron), create neuron-id shuffled spk_mat (n_shuffle, n_time, n_neuron);
    each shuffle uses ONE random permutation of neuron IDs, applied to all time bins.

    Cluster Jupyter example

```python
import poor_man_gplvm.post_fit_workflow.shuffle_test_decoding as shuf

spk_shuf = shuf.create_neuron_id_shuffle_one_event(spk_event, n_shuffle=200, seed=0)
print(spk_shuf.shape)
```
    '''
    spk = np.asarray(spk_mat)
    rng = np.random.default_rng(seed)
    n_time, n_neuron = spk.shape

    perm = np.asarray([rng.permutation(n_neuron) for _ in range(int(n_shuffle))], dtype=int)  # (n_shuffle, n_neuron)
    spk_t = spk.T  # (n_neuron, n_time)
    out = spk_t[perm].transpose(0, 2, 1)  # (n_shuffle, n_time, n_neuron)
    return out


def create_circular_shuffle_one_event(spk_mat, n_shuffle=100, seed=0):
    '''
    given spk_mat (n_time, n_neuron), create circular shuffled spk_mat (n_shuffle, n_time, n_neuron):
    each neuron's time series is circularly shifted independently (within-event).

    Cluster Jupyter example

```python
import poor_man_gplvm.post_fit_workflow.shuffle_test_decoding as shuf

spk_shuf = shuf.create_circular_shuffle_one_event(spk_event, n_shuffle=200, seed=0)
print(spk_shuf.shape)
```
    '''
    spk = np.asarray(spk_mat)
    rng = np.random.default_rng(seed)
    n_time, n_neuron = spk.shape
    if n_time <= 1:
        return np.broadcast_to(spk[None, :, :], (int(n_shuffle), n_time, n_neuron)).copy()

    offsets = rng.integers(0, n_time, size=(int(n_shuffle), n_neuron), dtype=int)
    base = np.arange(n_time, dtype=int)[None, None, :]
    idx = (base + offsets[:, :, None]) % n_time  # (n_shuffle, n_neuron, n_time)

    spk_t = spk.T  # (n_neuron, n_time)
    spk_t_b = np.broadcast_to(spk_t[None, :, :], (int(n_shuffle), n_neuron, n_time))
    out_t = np.take_along_axis(spk_t_b, idx, axis=2)  # (n_shuffle, n_neuron, n_time)
    out = out_t.transpose(0, 2, 1)
    return out


def create_neuron_id_shuffle_all_events_concat(spk_mat, event_index_per_bin, n_shuffle=100, seed=0, block_size=16):
    '''
    given spk_mat (n_time,n_neuron), and event_index_per_bin (n_time,) (from trial_analysis.bin_spike_train_to_trial_based), 
    create neuron-id shuffled spk_mat (n_shuffle, n_time, n_neuron); each spike count is assigned a random neuron-id
    '''
    spk = np.asarray(spk_mat)
    event_index_per_bin = np.asarray(event_index_per_bin)

    keep = event_index_per_bin >= 0
    spk = spk[keep]
    event_index_per_bin = event_index_per_bin[keep]

    parsing = _parse_event_index_per_bin(event_index_per_bin)
    starts = parsing['starts']
    ends = parsing['ends']
    n_time, n_neuron = spk.shape
    n_shuffle = int(n_shuffle)

    rng = np.random.default_rng(seed)
    out = np.empty((n_shuffle, n_time, n_neuron), dtype=spk.dtype)

    n_block = int(np.ceil(n_shuffle / int(block_size))) if n_shuffle else 0
    for bi in range(n_block):
        s0 = bi * int(block_size)
        s1 = min((bi + 1) * int(block_size), n_shuffle)
        bsz = int(s1 - s0)
        out_b = np.empty((bsz, n_time, n_neuron), dtype=spk.dtype)

        for st, en in zip(starts, ends):
            spk_ev = spk[st:en]  # (t_ev, n_neuron)
            spk_ev_t = spk_ev.T  # (n_neuron, t_ev)
            perm = np.asarray([rng.permutation(n_neuron) for _ in range(bsz)], dtype=int)  # (bsz, n_neuron)
            out_b[:, st:en, :] = spk_ev_t[perm].transpose(0, 2, 1)

        out[s0:s1] = out_b
    return out


def create_circular_shuffle_all_events_concat(spk_mat, event_index_per_bin, n_shuffle=100, seed=0, block_size=16):
    '''
    given spk_mat (n_time,n_neuron), and event_index_per_bin (n_time,) (from trial_analysis.bin_spike_train_to_trial_based), 
    create circular shuffled spk_mat (n_shuffle, n_time, n_neuron): each neuron's spike count is circularly shuffled independently
    '''
    spk = np.asarray(spk_mat)
    event_index_per_bin = np.asarray(event_index_per_bin)

    keep = event_index_per_bin >= 0
    spk = spk[keep]
    event_index_per_bin = event_index_per_bin[keep]

    parsing = _parse_event_index_per_bin(event_index_per_bin)
    starts = parsing['starts']
    ends = parsing['ends']

    n_time, n_neuron = spk.shape
    n_shuffle = int(n_shuffle)
    rng = np.random.default_rng(seed)
    out = np.empty((n_shuffle, n_time, n_neuron), dtype=spk.dtype)

    n_block = int(np.ceil(n_shuffle / int(block_size))) if n_shuffle else 0
    for bi in range(n_block):
        s0 = bi * int(block_size)
        s1 = min((bi + 1) * int(block_size), n_shuffle)
        bsz = int(s1 - s0)
        out_b = np.empty((bsz, n_time, n_neuron), dtype=spk.dtype)

        for st, en in zip(starts, ends):
            spk_ev = spk[st:en]  # (t_ev, n_neuron)
            t_ev = int(spk_ev.shape[0])
            if t_ev <= 1:
                out_b[:, st:en, :] = spk_ev[None, :, :]
                continue

            offsets = rng.integers(0, t_ev, size=(bsz, n_neuron), dtype=int)
            base = np.arange(t_ev, dtype=int)[None, None, :]
            idx = (base + offsets[:, :, None]) % t_ev  # (bsz, n_neuron, t_ev)

            spk_ev_t = spk_ev.T  # (n_neuron, t_ev)
            spk_t_b = np.broadcast_to(spk_ev_t[None, :, :], (bsz, n_neuron, t_ev))
            out_ev_t = np.take_along_axis(spk_t_b, idx, axis=2)  # (bsz, n_neuron, t_ev)
            out_b[:, st:en, :] = out_ev_t.transpose(0, 2, 1)

        out[s0:s1] = out_b
    return out


def shuffle_test_naive_bayes_marginal_l(
    spk_mat,
    event_index_per_bin,
    tuning,
    n_shuffle=100,
    sig_thresh=0.95,
    q_l=None,
    seed=0,
    dt=0.02,
    gain=1.0,
    model_fit_dt=0.1,
    tuning_is_count_per_bin=False,
    decoding_kwargs=None,
    dosave=False,
    force_reload=False,
    save_dir=None,
    save_fn='shuffle_test_naive_bayes_marginal_l.npz',
    return_shuffle=False,
):
    '''
    create neuron-id and circular shuffle (within-event); decode true + shuffles; get per-event log marginal sum quantiles

    - Test statistic: log_marginal_per_event = sum_t log_marginal_l[t] within each event
    - Significance: true > quantile(sig_thresh) of shuffle distribution (per event)

    Notes
    - dt / gain are taken out of decoding_kwargs (like `transition_analysis.decode_compare_transition_kernels`).
    - model_fit_dt is only for scaling LVM tuning (counts/bin) into Hz; for supervised tuning (already Hz), keep
      tuning_is_count_per_bin=False and model_fit_dt is ignored.

    Cluster Jupyter example

```python
import poor_man_gplvm.post_fit_workflow.shuffle_test_decoding as shuf

# spk_mat, event_index_per_bin, time_l typically from trial_analysis.bin_spike_train_to_trial_based(...)
# tuning_res = get_tuning_supervised.get_tuning(...)

res = shuf.shuffle_test_naive_bayes_marginal_l(
    spk_mat,
    event_index_per_bin,
    tuning=tuning_res['tuning_flat'],
    n_shuffle=200,
    sig_thresh=0.95,
    seed=0,
    dt=0.02,
    gain=1.0,
    decoding_kwargs={'n_time_per_chunk': 20000},
    dosave=True,
    force_reload=False,
    save_dir='/mnt/home/szheng/ceph/<project_name>/data/<session>/shuffle_test',
    save_fn='shuffle_test_naive_bayes_marginal_l.npz',
)
print(res['is_sig_overall'].mean())
```
    '''
    if bool(dosave):
        if save_dir is None:
            raise ValueError('dosave=True requires save_dir.')
        save_paths = _resolve_save_paths(save_dir, save_fn)
        if (not bool(force_reload)) and os.path.exists(save_paths['pkl_path']):
            return _load_pickle(save_paths['pkl_path'])
        if (not bool(force_reload)) and os.path.exists(save_paths['npz_path']):
            return _load_npz_dict(save_paths['npz_path'])

    spk = np.asarray(spk_mat)
    event_index_per_bin = np.asarray(event_index_per_bin)
    tuning = np.asarray(tuning)
    n_shuffle = int(n_shuffle)
    decoding_kwargs = {} if decoding_kwargs is None else dict(decoding_kwargs)
    if 'dt' in decoding_kwargs:
        dt = decoding_kwargs.pop('dt')
    if 'gain' in decoding_kwargs:
        gain = decoding_kwargs.pop('gain')
    q_l = np.linspace(0, 1, 41) if q_l is None else np.asarray(q_l)

    keep = event_index_per_bin >= 0
    spk = spk[keep]
    event_index_per_bin = event_index_per_bin[keep]

    parsing = _parse_event_index_per_bin(event_index_per_bin)
    starts = parsing['starts']
    ends = parsing['ends']
    event_l = parsing['event_l']
    n_event = int(len(event_l))
    n_time, n_neuron = spk.shape

    if bool(tuning_is_count_per_bin):
        tuning_hz = tuning / float(model_fit_dt)
    else:
        tuning_hz = tuning

    print('[shuffle_test_naive_bayes_marginal_l] chunk: decode true')
    res_true = decoder_supervised.decode_naive_bayes(
        spk,
        tuning_hz,
        event_index_per_bin=event_index_per_bin,
        return_per_event=False,
        dt=float(dt),
        gain=float(gain),
        **decoding_kwargs,
    )
    log_marginal_per_event_true = np.asarray(res_true['log_marginal_per_event'])

    print('[shuffle_test_naive_bayes_marginal_l] chunk: init shuffle stats')
    log_marginal_per_event_id_shuffle = np.empty((n_shuffle, n_event), dtype=float)
    log_marginal_per_event_circular_shuffle = np.empty((n_shuffle, n_event), dtype=float)

    rng_id = np.random.default_rng(int(seed))
    rng_circ = np.random.default_rng(int(seed) + 1)

    print('[shuffle_test_naive_bayes_marginal_l] chunk: decode shuffles (tqdm)')
    spk_id = np.empty((n_time, n_neuron), dtype=spk.dtype)
    spk_circ = np.empty((n_time, n_neuron), dtype=spk.dtype)
    for si in tqdm.tqdm(range(n_shuffle), total=n_shuffle):
        for st, en in zip(starts, ends):
            spk_ev = spk[st:en]  # (t_ev, n_neuron)
            t_ev = int(spk_ev.shape[0])

            perm = rng_id.permutation(n_neuron)
            spk_id[st:en, :] = spk_ev[:, perm]

            if t_ev <= 1:
                spk_circ[st:en, :] = spk_ev
            else:
                offsets = rng_circ.integers(0, t_ev, size=(n_neuron,), dtype=int)
                base = np.arange(t_ev, dtype=int)[:, None]
                idx = (base + offsets[None, :]) % t_ev  # (t_ev, n_neuron)
                spk_circ[st:en, :] = np.take_along_axis(spk_ev, idx, axis=0)

        res_id = decoder_supervised.decode_naive_bayes(
            spk_id,
            tuning_hz,
            event_index_per_bin=event_index_per_bin,
            return_per_event=False,
            dt=float(dt),
            gain=float(gain),
            **decoding_kwargs,
        )
        res_c = decoder_supervised.decode_naive_bayes(
            spk_circ,
            tuning_hz,
            event_index_per_bin=event_index_per_bin,
            return_per_event=False,
            dt=float(dt),
            gain=float(gain),
            **decoding_kwargs,
        )
        log_marginal_per_event_id_shuffle[si] = np.asarray(res_id['log_marginal_per_event'])
        log_marginal_per_event_circular_shuffle[si] = np.asarray(res_c['log_marginal_per_event'])

    print('[shuffle_test_naive_bayes_marginal_l] chunk: quantiles + significance')
    log_marginal_per_event_id_shuffle_q = np.quantile(log_marginal_per_event_id_shuffle, q_l, axis=0)
    log_marginal_per_event_circular_shuffle_q = np.quantile(log_marginal_per_event_circular_shuffle, q_l, axis=0)
    id_thresh = np.quantile(log_marginal_per_event_id_shuffle, float(sig_thresh), axis=0)
    circ_thresh = np.quantile(log_marginal_per_event_circular_shuffle, float(sig_thresh), axis=0)

    is_sig_id_shuffle = log_marginal_per_event_true > id_thresh
    is_sig_circular_shuffle = log_marginal_per_event_true > circ_thresh
    is_sig_overall = np.logical_and(is_sig_id_shuffle, is_sig_circular_shuffle)

    event_df = pd.DataFrame(
        {
            'event_i': event_l,
            'start_index': starts,
            'end_index': ends,
            'log_marginal_per_event_true': log_marginal_per_event_true,
            'id_shuffle_thresh': id_thresh,
            'circular_shuffle_thresh': circ_thresh,
            'is_sig_id_shuffle': is_sig_id_shuffle,
            'is_sig_circular_shuffle': is_sig_circular_shuffle,
            'is_sig_overall': is_sig_overall,
        },
    )

    is_sig_frac = pd.Series(
        {
            'id': float(np.mean(np.asarray(event_df['is_sig_id_shuffle'].to_numpy(dtype=bool)))) if n_event else float('nan'),
            'circular': float(np.mean(np.asarray(event_df['is_sig_circular_shuffle'].to_numpy(dtype=bool)))) if n_event else float('nan'),
            'overall': float(np.mean(np.asarray(event_df['is_sig_overall'].to_numpy(dtype=bool)))) if n_event else float('nan'),
        },
        name='is_sig_frac',
    )

    print('[shuffle_test_naive_bayes_marginal_l] chunk: expand is_sig_overall to time bins')
    event_id_l = np.asarray(event_df['event_i'].to_numpy())
    sig_event_l = np.asarray(event_df['is_sig_overall'].to_numpy(dtype=bool))
    if event_id_l.size:
        max_event_id = int(np.max(event_id_l))
        if max_event_id <= 50_000_000:
            lut = np.zeros((max_event_id + 1,), dtype=bool)
            lut[event_id_l.astype(int)] = sig_event_l
            is_sig_mask_time = lut[event_index_per_bin.astype(int)]
        else:
            order = np.argsort(event_id_l.astype(int))
            event_id_sorted = event_id_l.astype(int)[order]
            sig_sorted = sig_event_l[order]
            idx = np.searchsorted(event_id_sorted, event_index_per_bin.astype(int))
            is_sig_mask_time = np.zeros_like(event_index_per_bin, dtype=bool)
            ok = (idx >= 0) & (idx < event_id_sorted.size) & (event_id_sorted[idx] == event_index_per_bin.astype(int))
            is_sig_mask_time[ok] = sig_sorted[idx[ok]]
    else:
        is_sig_mask_time = np.zeros_like(event_index_per_bin, dtype=bool)

    q_cols = pd.Index(np.asarray(q_l), name='q')
    log_marginal_id_q_df = pd.DataFrame(
        log_marginal_per_event_id_shuffle_q.T,
        index=pd.Index(event_l, name='event_l'),
        columns=q_cols,
    )
    log_marginal_circular_q_df = pd.DataFrame(
        log_marginal_per_event_circular_shuffle_q.T,
        index=pd.Index(event_l, name='event_l'),
        columns=q_cols,
    )

    out = {
        'event_df': event_df,
        'log_marginal_id_q_df': log_marginal_id_q_df,
        'log_marginal_circular_q_df': log_marginal_circular_q_df,
        'decode_true': res_true,
        'is_sig_mask_time': is_sig_mask_time,
        'is_sig_frac': is_sig_frac,
        'meta': {
            'n_time': int(n_time),
            'n_neuron': int(n_neuron),
            'n_event': int(n_event),
            'n_shuffle': int(n_shuffle),
            'seed': int(seed),
            'dt': float(dt),
            'gain': float(gain),
            'model_fit_dt': float(model_fit_dt),
            'tuning_is_count_per_bin': bool(tuning_is_count_per_bin),
            'decoding_kwargs': decoding_kwargs,
            'return_shuffle': bool(return_shuffle),
            'sig_thresh': float(sig_thresh),
        },
    }

    if bool(return_shuffle):
        out['event_l'] = event_l
        out['starts'] = starts
        out['ends'] = ends
        out['event_index_per_bin'] = event_index_per_bin
        out['q_l'] = q_l
        out['log_marginal_per_event_true'] = log_marginal_per_event_true
        out['log_marginal_per_event_id_shuffle'] = log_marginal_per_event_id_shuffle
        out['log_marginal_per_event_circular_shuffle'] = log_marginal_per_event_circular_shuffle
        out['log_marginal_per_event_id_shuffle_q'] = log_marginal_per_event_id_shuffle_q
        out['log_marginal_per_event_circular_shuffle_q'] = log_marginal_per_event_circular_shuffle_q
        out['is_sig_id_shuffle'] = is_sig_id_shuffle
        out['is_sig_circular_shuffle'] = is_sig_circular_shuffle
        out['is_sig_overall'] = is_sig_overall

    if bool(dosave):
        os.makedirs(str(save_dir), exist_ok=True)
        save_paths = _resolve_save_paths(save_dir, save_fn)
        _save_pickle(save_paths['pkl_path'], out)
        if bool(return_shuffle):
            _save_npz_dict(save_paths['npz_path'], {
                'event_l': event_l,
                'starts': starts,
                'ends': ends,
                'event_index_per_bin': event_index_per_bin,
                'q_l': q_l,
                'log_marginal_per_event_true': log_marginal_per_event_true,
                'log_marginal_per_event_id_shuffle': log_marginal_per_event_id_shuffle,
                'log_marginal_per_event_circular_shuffle': log_marginal_per_event_circular_shuffle,
                'log_marginal_per_event_id_shuffle_q': log_marginal_per_event_id_shuffle_q,
                'log_marginal_per_event_circular_shuffle_q': log_marginal_per_event_circular_shuffle_q,
                'is_sig_id_shuffle': is_sig_id_shuffle,
                'is_sig_circular_shuffle': is_sig_circular_shuffle,
                'is_sig_overall': is_sig_overall,
            })
    return out


def sweep_gain_shuffle_test_naive_bayes_marginal_l(
    spk_mat,
    event_index_per_bin,
    tuning,
    *,
    min_gain=1,
    max_gain=10,
    gain_step=2,
    train_frac=0.5,
    train_seed=0,
    n_shuffle=100,
    sig_thresh=0.95,
    q_l=None,
    seed=0,
    dt=0.02,
    model_fit_dt=0.1,
    tuning_is_count_per_bin=False,
    decoding_kwargs=None,
    dosave=False,
    force_reload=False,
    save_dir=None,
    save_fn='sweep_gain_shuffle_test_naive_bayes_marginal_l.pkl',
    return_full_res=False,
):
    '''
    Sweep gain for shuffle_test_naive_bayes_marginal_l with an event-based train/test split.

    Stop rule: stop at max_gain or when test frac_sig_overall decreased for the past two gains
    (unless total gains to try < 7, then run all).

    Cluster Jupyter example

```python
import poor_man_gplvm.post_fit_workflow.shuffle_test_decoding as shuf

res = shuf.sweep_gain_shuffle_test_naive_bayes_marginal_l(
    spk_mat,
    event_index_per_bin,
    tuning=tuning_res['tuning_flat'],  # supervised tuning (Hz)
    dt=0.02,
    min_gain=0.5,
    max_gain=3.0,
    gain_step=0.25,
    train_frac=0.5,
    train_seed=0,
    n_shuffle=200,
    seed=0,
    decoding_kwargs={'n_time_per_chunk': 20000},
    dosave=False,
)
print(res['best_gain'], res['best_test_frac_sig'])
```
    '''
    if bool(dosave):
        if save_dir is None:
            raise ValueError('dosave=True requires save_dir.')
        save_paths = _resolve_save_paths(save_dir, save_fn)
        if (not bool(force_reload)) and os.path.exists(save_paths['pkl_path']):
            return _load_pickle(save_paths['pkl_path'])
        if (not bool(force_reload)) and os.path.exists(save_paths['npz_path']):
            return _load_npz_dict(save_paths['npz_path'])

    decoding_kwargs = {} if decoding_kwargs is None else dict(decoding_kwargs)
    if 'dt' in decoding_kwargs:
        dt = decoding_kwargs.pop('dt')
    if 'gain' in decoding_kwargs:
        decoding_kwargs.pop('gain')

    spk = np.asarray(spk_mat)
    event_index_per_bin = np.asarray(event_index_per_bin)
    keep = event_index_per_bin >= 0
    event_index_keep = event_index_per_bin[keep]
    parsing = _parse_event_index_per_bin(event_index_keep)
    event_l = np.asarray(parsing['event_l'])
    n_event = int(event_l.size)

    rng = np.random.default_rng(int(train_seed))
    n_train = int(np.round(float(train_frac) * n_event))
    n_train = max(0, min(n_event, n_train))
    train_idx = np.sort(rng.choice(n_event, size=n_train, replace=False)) if n_train else np.asarray([], dtype=int)
    train_event_l = event_l[train_idx]
    is_train = np.zeros((n_event,), dtype=bool)
    is_train[train_idx] = True
    test_event_l = event_l[~is_train]
    print(f'[sweep_gain_shuffle_test_naive_bayes_marginal_l] event split: n_event={n_event} n_train={int(train_event_l.size)} n_test={int(test_event_l.size)}')

    gain_l = np.arange(float(min_gain), float(max_gain) + 1e-12, float(gain_step), dtype=float)
    if gain_l.size == 0:
        gain_l = np.asarray([float(min_gain)], dtype=float)

    early_stop = bool(gain_l.size >= 7)
    gain_tried_l = []
    train_frac_sig_l = []
    test_frac_sig_l = []
    shuffle_res_l = [] if bool(return_full_res) else None
    best_res = None
    best_gain = float('nan')
    best_test_frac = -np.inf

    for gi, gain in enumerate(gain_l):
        print(f'[sweep_gain_shuffle_test_naive_bayes_marginal_l] gain {gi+1}/{int(gain_l.size)}: {float(gain):.6g}')
        res = shuffle_test_naive_bayes_marginal_l(
            spk,
            event_index_per_bin,
            tuning=tuning,
            n_shuffle=int(n_shuffle),
            sig_thresh=float(sig_thresh),
            q_l=q_l,
            seed=int(seed),
            dt=float(dt),
            gain=float(gain),
            model_fit_dt=float(model_fit_dt),
            tuning_is_count_per_bin=bool(tuning_is_count_per_bin),
            decoding_kwargs=decoding_kwargs,
            dosave=False,
            force_reload=False,
            return_shuffle=False,
        )

        event_df = res['event_df']
        ev_id = np.asarray(event_df['event_i'].to_numpy())
        sig = np.asarray(event_df['is_sig_overall'].to_numpy(dtype=bool))
        m_train = np.isin(ev_id, train_event_l)
        m_test = np.isin(ev_id, test_event_l)
        frac_train = float(np.mean(sig[m_train])) if np.any(m_train) else float('nan')
        frac_test = float(np.mean(sig[m_test])) if np.any(m_test) else float('nan')

        gain_tried_l.append(float(gain))
        train_frac_sig_l.append(frac_train)
        test_frac_sig_l.append(frac_test)
        if shuffle_res_l is not None:
            shuffle_res_l.append(res)

        frac_test_cmp = float(frac_test) if np.isfinite(frac_test) else -np.inf
        if (best_res is None) or (frac_test_cmp > float(best_test_frac)):
            best_res = res
            best_gain = float(gain)
            best_test_frac = float(frac_test_cmp)

        if early_stop and len(test_frac_sig_l) >= 3:
            if (test_frac_sig_l[-1] < test_frac_sig_l[-2]) and (test_frac_sig_l[-2] < test_frac_sig_l[-3]):
                print('[sweep_gain_shuffle_test_naive_bayes_marginal_l] early stop: test frac_sig decreased for past two gains')
                break

    gain_tried_l = np.asarray(gain_tried_l, dtype=float)
    train_frac_sig_l = np.asarray(train_frac_sig_l, dtype=float)
    test_frac_sig_l = np.asarray(test_frac_sig_l, dtype=float)
    if best_res is None and gain_tried_l.size:
        # fallback (shouldn't happen)
        best_res = shuffle_test_naive_bayes_marginal_l(
            spk,
            event_index_per_bin,
            tuning=tuning,
            n_shuffle=int(n_shuffle),
            sig_thresh=float(sig_thresh),
            q_l=q_l,
            seed=int(seed),
            dt=float(dt),
            gain=float(gain_tried_l[0]),
            model_fit_dt=float(model_fit_dt),
            tuning_is_count_per_bin=bool(tuning_is_count_per_bin),
            decoding_kwargs=decoding_kwargs,
            dosave=False,
            force_reload=False,
            return_shuffle=False,
        )
        best_gain = float(gain_tried_l[0])
        best_test_frac = float(test_frac_sig_l[0]) if test_frac_sig_l.size else float('nan')

    out = {
        'gain_l': gain_tried_l,
        'train_frac_sig_overall_l': train_frac_sig_l,
        'test_frac_sig_overall_l': test_frac_sig_l,
        'best_gain': float(best_gain),
        'best_test_frac_sig': float(best_test_frac) if np.isfinite(best_test_frac) else float('nan'),
        'best_shuffle_test_res': best_res,
        'train_event_l': np.asarray(train_event_l),
        'test_event_l': np.asarray(test_event_l),
        'meta': {
            'min_gain': float(min_gain),
            'max_gain': float(max_gain),
            'gain_step': float(gain_step),
            'train_frac': float(train_frac),
            'train_seed': int(train_seed),
            'n_shuffle': int(n_shuffle),
            'sig_thresh': float(sig_thresh),
            'seed': int(seed),
            'dt': float(dt),
            'model_fit_dt': float(model_fit_dt),
            'tuning_is_count_per_bin': bool(tuning_is_count_per_bin),
            'decoding_kwargs': decoding_kwargs,
            'early_stop_enabled': bool(early_stop),
            'return_full_res': bool(return_full_res),
        },
    }
    if shuffle_res_l is not None:
        out['shuffle_res_l'] = np.asarray(shuffle_res_l, dtype=object)

    if bool(dosave):
        os.makedirs(str(save_dir), exist_ok=True)
        save_paths = _resolve_save_paths(save_dir, save_fn)
        _save_pickle(save_paths['pkl_path'], out)
        _save_npz_dict(save_paths['npz_path'], {
            'gain_l': gain_tried_l,
            'train_frac_sig_overall_l': train_frac_sig_l,
            'test_frac_sig_overall_l': test_frac_sig_l,
            'best_gain': np.asarray(out['best_gain']),
            'best_test_frac_sig': np.asarray(out['best_test_frac_sig']),
            'train_event_l': np.asarray(train_event_l),
            'test_event_l': np.asarray(test_event_l),
            'meta': np.asarray(out['meta'], dtype=object),
        })

    return out
