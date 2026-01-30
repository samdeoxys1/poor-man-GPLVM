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
        elif isinstance(v, pd.DataFrame):
            d2[k] = np.asarray(v, dtype=object)
        else:
            d2[k] = v
    np.savez(str(save_path), **d2)
    print(f'saved: {save_path}')


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
    decoding_kwargs={'dt': 0.02, 'gain': 1.0, 'n_time_per_chunk': 20000},
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
        save_path = os.path.join(str(save_dir), str(save_fn))
        if (not bool(force_reload)) and os.path.exists(save_path):
            return _load_npz_dict(save_path)

    spk = np.asarray(spk_mat)
    event_index_per_bin = np.asarray(event_index_per_bin)
    tuning = np.asarray(tuning)
    n_shuffle = int(n_shuffle)
    decoding_kwargs = {} if decoding_kwargs is None else dict(decoding_kwargs)
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

    print('[shuffle_test_naive_bayes_marginal_l] chunk: decode true')
    res_true = decoder_supervised.decode_naive_bayes(
        spk,
        tuning,
        event_index_per_bin=event_index_per_bin,
        return_per_event=False,
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
            tuning,
            event_index_per_bin=event_index_per_bin,
            return_per_event=False,
            **decoding_kwargs,
        )
        res_c = decoder_supervised.decode_naive_bayes(
            spk_circ,
            tuning,
            event_index_per_bin=event_index_per_bin,
            return_per_event=False,
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
            'starts': starts,
            'ends': ends,
            'log_marginal_per_event_true': log_marginal_per_event_true,
            'id_shuffle_thresh': id_thresh,
            'circular_shuffle_thresh': circ_thresh,
            'is_sig_id_shuffle': is_sig_id_shuffle,
            'is_sig_circular_shuffle': is_sig_circular_shuffle,
            'is_sig_overall': is_sig_overall,
        },
        index=pd.Index(event_l, name='event_l'),
    )

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
        'meta': {
            'n_time': int(n_time),
            'n_neuron': int(n_neuron),
            'n_event': int(n_event),
            'n_shuffle': int(n_shuffle),
            'seed': int(seed),
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
        _save_npz_dict(os.path.join(str(save_dir), str(save_fn)), out)
    return out

