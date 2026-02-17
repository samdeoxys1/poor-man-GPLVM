'''
After fitting the model, we can decode latent with transitions different from the 
prior to make it closer to the posterior; or use any custom transition to define "continuous" and then "jump" is the deviation from that
or compute posterior transition from different epochs and do model comparison in ripples
'''

"""
Functions for:
A) Estimating latent transition matrices from posterior marginals within behavioral epochs,
   respecting episode boundaries (no cross-episode transitions).
B) Decoding spk_list under multiple custom transition kernels to compare marginal likelihoods.

Example usage:
    # A) Transition estimation
    epoch_dict = {"loc": loc_iset, "imm": imm_iset}
    epoch_trans = transition_by_epoch(post_lat_tsd, epoch_dict)

    # B) Kernel comparison
    kernels = [kernel_A, kernel_B]
    cmp = decode_compare_transition_kernels(spk_list, model, kernels, kernel_names=["A","B"])
"""

import numpy as np
import pynapple as nap
from typing import Optional, List, Dict, Any, Union
import scipy.spatial
import jax
import jax.numpy as jnp
import pandas as pd
import poor_man_gplvm.decoder_trial as pdt
import poor_man_gplvm.gp_kernel as gpk
import pickle

# =============================================================================
# A) Transition estimation within epochs (latent-only, no forward-backward)
# =============================================================================

def transition_counts_from_posterior_latent(
    post_lat: np.ndarray,
    *,
    episode_start_mask: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    """
    Compute expected latent transition counts from posterior marginals.

    Method (simple, no forward-backward):
        C[i,j] = sum_t p(z_{t-1}=i) * p(z_t=j)  for valid t
        P[j|i] = C[i,j] / sum_j C[i,j]

    Parameters
    ----------
    post_lat : (T, K) array
        Posterior marginal over latent states at each time bin. Rows should sum to 1.
    episode_start_mask : (T,) bool array, optional
        If episode_start_mask[t] is True, skip transition into t (i.e. t-1 -> t is invalid).
        Default: only t=0 is an episode start.

    Returns
    -------
    dict with:
        counts : (K, K) expected joint counts
        trans_mat : (K, K) row-normalized transition matrix (rows may be 0 if no mass)
        row_mass : (K,) total outgoing mass per source state
        n_steps_used : int, number of valid transitions counted
    """
    T, K = post_lat.shape
    if T < 2:
        return {
            "counts": np.zeros((K, K)),
            "trans_mat": np.zeros((K, K)),
            "row_mass": np.zeros(K),
            "n_steps_used": 0,
        }

    if episode_start_mask is None:
        episode_start_mask = np.zeros(T, dtype=bool)
        episode_start_mask[0] = True

    # valid transitions: t where episode_start_mask[t] is False and t >= 1
    valid_t = np.where(~episode_start_mask)[0]
    valid_t = valid_t[valid_t >= 1]

    if len(valid_t) == 0:
        return {
            "counts": np.zeros((K, K)),
            "trans_mat": np.zeros((K, K)),
            "row_mass": np.zeros(K),
            "n_steps_used": 0,
        }

    # C[i,j] = sum over valid t of post_lat[t-1, i] * post_lat[t, j]
    prev = post_lat[valid_t - 1]  # (n_valid, K)
    curr = post_lat[valid_t]       # (n_valid, K)
    counts = prev.T @ curr         # (K, K)

    row_mass = counts.sum(axis=1)
    trans_mat = np.zeros_like(counts)
    nonzero = row_mass > 0
    trans_mat[nonzero] = counts[nonzero] / row_mass[nonzero, None]

    return {
        "counts": counts,
        "trans_mat": trans_mat,
        "row_mass": row_mass,
        "n_steps_used": len(valid_t),
    }


def transition_from_posterior_in_intervalset(
    post_lat_tsd: nap.TsdFrame,
    epoch_iset: nap.IntervalSet,
    *,
    episode_is_interval: bool = True,
) -> Dict[str, Any]:
    """
    Compute latent transition matrix from posterior within an IntervalSet.

    Each interval in epoch_iset is treated as one episode; transitions do not cross episodes.

    Parameters
    ----------
    post_lat_tsd : nap.TsdFrame
        Posterior marginal over latent states, shape (T_total, K).
    epoch_iset : nap.IntervalSet
        Intervals defining the epoch. Each interval = one episode.
    episode_is_interval : bool
        If True (default), each interval is a separate episode (no cross-interval transitions).

    Returns
    -------
    dict with aggregated counts, trans_mat, row_mass, n_steps_used across all episodes.
    """
    K = post_lat_tsd.shape[1]
    total_counts = np.zeros((K, K))
    total_n_steps = 0

    for i in range(len(epoch_iset)):
        start, end = epoch_iset[i, 0], epoch_iset[i, 1]
        restricted = post_lat_tsd.restrict(nap.IntervalSet(start, end))
        if len(restricted) < 2:
            continue
        post_arr = np.asarray(restricted.d)
        # within one episode, only t=0 is an episode start
        ep_mask = np.zeros(len(post_arr), dtype=bool)
        ep_mask[0] = True
        res = transition_counts_from_posterior_latent(post_arr, episode_start_mask=ep_mask)
        total_counts += res["counts"]
        total_n_steps += res["n_steps_used"]

    row_mass = total_counts.sum(axis=1)
    trans_mat = np.zeros_like(total_counts)
    nonzero = row_mass > 0
    trans_mat[nonzero] = total_counts[nonzero] / row_mass[nonzero, None]

    return {
        "counts": total_counts,
        "trans_mat": trans_mat,
        "row_mass": row_mass,
        "n_steps_used": total_n_steps,
    }


def transition_by_epoch(
    post_lat_tsd: nap.TsdFrame,
    epoch_dict: Dict[str, nap.IntervalSet],
) -> Dict[str, Dict[str, Any]]:
    """
    Compute latent transitions for multiple behavioral epochs.

    Parameters
    ----------
    post_lat_tsd : nap.TsdFrame
        Posterior marginal over latent states.
    epoch_dict : dict
        Keys are epoch names (e.g. "loc", "imm", "headscan"),
        values are nap.IntervalSet for each epoch.

    Returns
    -------
    dict of dicts with hierarchy `{key: {epoch_name: value}}`, where `key` is one of
    the outputs of `transition_from_posterior_in_intervalset` (e.g. "counts", "trans_mat",
    "row_mass", "n_steps_used").
    """
    by_epoch = {
        name: transition_from_posterior_in_intervalset(post_lat_tsd, iset)
        for name, iset in epoch_dict.items()
    }
    if not by_epoch:
        return {}

    # flip to {key: {epoch: ...}} to avoid confusion downstream
    keys = list(next(iter(by_epoch.values())).keys())
    out = {}
    for k in keys:
        out[k] = {epoch: by_epoch[epoch][k] for epoch in by_epoch.keys()}
    return out

def transition_from_tuning_distance(tuning_fit,inverse_temperature=1.,metric='cosine'):
    '''
    turn tuning similarity into transition matrix
    distance (i.e. dis-similarity) as energy/cost; Gibbs distribution for transition
    tuning_fit: n_latent_bin x n_neuron
    inverse_temperature: inverse of temperature, lower temperature means more sharp transition
    '''
    
    tuning_distance = scipy.spatial.distance.pdist(tuning_fit, metric=metric) 
    tuning_distance = scipy.spatial.distance.squareform(tuning_distance) # n_latent_bin x n_latent_bin
    
    transition_matrix = np.exp(-tuning_distance * inverse_temperature)
    transition_matrix = transition_matrix / transition_matrix.sum(axis=1,keepdims=True)
    return transition_matrix,tuning_distance

def select_inverse_temperature(tuning_fit,p_trans_target,inverse_temperature_l = np.arange(5,20),metric='cosine'):
    '''
    select inverse_temperature for transition_matrix, by matching the transition matrix to the target transition matrix
    the best target is the posterior transition probability only under the continuous dynamics, i.e. p_transition_full[0,0] / p_transition_full[0,0].sum(axis=1,keepdims=True)
    '''
    transition_matrix,tuning_distance = transition_from_tuning_distance(tuning_fit,inverse_temperature_l[0],metric=metric)
    loss_l = {}
    for inverse_temperature in inverse_temperature_l:
        transition_matrix,tuning_distance = transition_from_tuning_distance(tuning_fit,inverse_temperature,metric=metric)
        loss_l[inverse_temperature] = rowwise_cross_entropy_loss(inverse_temperature,tuning_distance,p_trans_target).item()
    loss_l =pd.Series(loss_l)
    best_inverse_temperature = loss_l.idxmin()
    transition_matrix,tuning_distance = transition_from_tuning_distance(tuning_fit,best_inverse_temperature,metric=metric)
    return transition_matrix,best_inverse_temperature,loss_l


def _weighted_quantile_1d(x, w, q):
    """
    Weighted quantile for 1D arrays.
    x, w: 1D arrays of same length, w >= 0
    q: float in (0, 1]
    returns: scalar threshold t such that cumulative weight of x<=t is q.
    """
    x = np.asarray(x)
    w = np.asarray(w)
    finite = np.isfinite(x) & np.isfinite(w)
    x = x[finite]
    w = w[finite]
    order = np.argsort(x)
    x_sorted = x[order]
    w_sorted = w[order]
    cw = np.cumsum(w_sorted)
    tot = cw[-1]
    if tot <= 0:
        raise ValueError("[_weighted_quantile_1d] sum(w) must be > 0")
    target = q * tot
    idx = int(np.searchsorted(cw, target, side="left"))
    idx = min(max(idx, 0), len(x_sorted) - 1)
    return float(x_sorted[idx])


def select_inverse_temperature_match_step(
    tuning_fit,
    p_joint_latent,
    p_latent,
    *,
    tail_quantile=0.9,
    inverse_temperature_l=np.arange(5, 20),
    metric="cosine",
):
    """
    Select inverse_temperature by matching a *bulk* step statistic (trimmed by tail_quantile).

    Given posterior latent pair weights P_post(i,j) proportional to expected joint occupancy
    across adjacent time bins (e.g. from model_fit.decode_latent), define a step metric g_ij
    from tuning_fit (pairwise
    distances under `metric`). Let tau be the weighted quantile of {g_ij} under P_post, then
    compute the bulk conditional mean:

        m* = E[g | g <= tau] under weights P_post.

    For each beta in inverse_temperature_l, define a Gibbs transition kernel
        K_beta(j|i) ∝ exp(-beta * g_ij),
    then row-normalize K_beta, and form a proposed joint with the provided stationary dist:
        Q_beta(i,j) = pi_i * K_beta(j|i),  where pi = p_latent.

    Choose beta that best matches m(beta) to m* under the same bulk truncation (g<=tau).

    Returns dict with:
        p_trans_latent_clean : (L,L) Gibbs transition matrix at best beta (rows normalized)
        best_inverse_temperature : scalar
        loss_l : pd.Series indexed by beta
        metric_mat : (L,L) pairwise distance matrix g_ij
        metric_at_quantile : scalar tau
    """
    metric_vec = scipy.spatial.distance.pdist(tuning_fit, metric=metric)
    metric_mat = scipy.spatial.distance.squareform(metric_vec)  # (L,L)

    P = np.asarray(p_joint_latent, dtype=float)
    P = np.nan_to_num(P, nan=0.0, posinf=0.0, neginf=0.0)
    P[P < 0] = 0.0
    Z = P.sum()
    if Z <= 0:
        raise ValueError("[select_inverse_temperature_match_step] p_joint_latent has zero total mass")
    p_joint_latent_clean = P / Z  # (L,L), sum=1

    pi = np.asarray(p_latent, dtype=float)
    pi = np.nan_to_num(pi, nan=0.0, posinf=0.0, neginf=0.0)
    pi[pi < 0] = 0.0
    pi_Z = pi.sum()
    if pi_Z <= 0:
        raise ValueError("[select_inverse_temperature_match_step] p_latent has zero total mass")
    p_latent_clean = pi / pi_Z

    # "step" should exclude self-transitions (diagonal), otherwise tau is often 0
    offdiag = ~np.eye(metric_mat.shape[0], dtype=bool)
    g = metric_mat[offdiag].ravel()
    w = p_joint_latent_clean[offdiag].ravel()
    metric_at_quantile = _weighted_quantile_1d(g, w, tail_quantile)
    bulk_mask = (metric_mat <= metric_at_quantile) & offdiag

    denom_star = float((p_joint_latent_clean * bulk_mask).sum())
    if denom_star <= 0:
        raise ValueError("[select_inverse_temperature_match_step] no mass in bulk (check tail_quantile / metric)")
    m_star = float((p_joint_latent_clean * metric_mat * bulk_mask).sum() / denom_star)

    loss_d = {}
    for beta in inverse_temperature_l:
        K_beta = np.exp(-metric_mat * float(beta))
        K_beta = K_beta / K_beta.sum(axis=1, keepdims=True)  # transition matrix, row-normalized
        Q_beta = p_latent_clean[:, None] * K_beta            # proposed joint
        Q_beta = Q_beta / Q_beta.sum()                       # normalize joint (safety)
        denom = float((Q_beta * bulk_mask).sum())
        if denom <= 0:
            loss_d[beta] = np.nan
            continue
        m_beta = float((Q_beta * metric_mat * bulk_mask).sum() / denom)
        loss_d[beta] = (m_beta - m_star) ** 2

    loss_l = pd.Series(loss_d)
    best_inverse_temperature = loss_l.idxmin()

    # final "clean" transition = Gibbs transition at best beta
    p_trans_latent_clean = np.exp(-metric_mat * float(best_inverse_temperature))
    p_trans_latent_clean = p_trans_latent_clean / p_trans_latent_clean.sum(axis=1, keepdims=True)

    return {
        "p_trans_latent_clean": p_trans_latent_clean,
        "best_inverse_temperature": best_inverse_temperature,
        "loss_l": loss_l,
        "metric_mat": metric_mat,
        "metric_at_quantile": metric_at_quantile,
    }


def select_inverse_temperature_match_step_continuous(
    tuning_fit,
    p_joint_full,
    p_latent,
    *,
    continuous_ind=0,
    inverse_temperature_l=np.arange(5, 20),
    metric="cosine",
):
    """
    Like select_inverse_temperature_match_step, but define the target step statistic using only
    the "continuous" regime of a full (regime x regime x L x L) joint.

    This extracts p_joint_full[continuous_ind, continuous_ind], renormalizes it into an (L,L)
    p_joint_latent_, then reuses select_inverse_temperature_match_step with tail_quantile=1.0.
    """
    p_joint_latent_ = np.asarray(p_joint_full)[continuous_ind, continuous_ind]
    p_joint_latent_ = np.nan_to_num(p_joint_latent_, nan=0.0, posinf=0.0, neginf=0.0)
    p_joint_latent_[p_joint_latent_ < 0] = 0.0
    Z = float(np.sum(p_joint_latent_))
    if Z <= 0:
        raise ValueError(
            "[select_inverse_temperature_match_step_continuous] p_joint_full[continuous_ind,continuous_ind] has zero total mass"
        )
    p_joint_latent_ = p_joint_latent_ / Z

    return select_inverse_temperature_match_step(
        tuning_fit,
        p_joint_latent_,
        p_latent,
        tail_quantile=1.0,
        inverse_temperature_l=inverse_temperature_l,
        metric=metric,
    )

#== helper for selecting inverse_temperature for transition_matrix from tuning_distance ==
def rowwise_cross_entropy_loss(beta, S, P_star, w=None):
    """
    beta: scalar
    S: (K, K) distance matrix (lower => more likely)
    P_star: (K, K) target transition probs (rows sum to 1)
    w: optional (K,) row weights (e.g., stationary dist or row counts)
    returns: scalar loss = sum_i w_i * H(P_star[i], P_beta[i])
    """
    logP_beta = jax.nn.log_softmax(-beta * S, axis=-1)  # (K, K)
    row_ce = -jnp.sum(P_star * logP_beta, axis=-1)     # (K,)

    if w is None:
        return jnp.sum(row_ce)
    w = w / jnp.sum(w)
    return jnp.sum(w * row_ce)


# =============================================================================
# B) Decode under multiple transition kernels + compare marginal likelihood
# =============================================================================

def decode_compare_transition_kernels(
    spk_tensor,
    model,
    continuous_transition_mat_d: Union[Dict[str, np.ndarray], List[np.ndarray]],
    tensor_pad_mask=None,
    *,
    dt: float = 0.02,
    gain: float = 1.0,
    model_fit_dt: float = 0.1,
    time_l=None,
    kernel_names: Optional[List[str]] = None,
    neuron_mask=None,
    ma_latent=None,
    n_trial_per_chunk: int = 400,
    prior_magnifier: float = 1.0,
    return_numpy: bool = True,
) -> Dict[str, Any]:
    """
    Decode trial-tensor spike counts under multiple *continuous/move* transition kernels
    and compare marginal likelihoods.

    The provided kernel only affects the continuous/move latent transition.
    Jump latent transition stays uniform, and move/jump switching is governed
    by p_move_to_jump / p_jump_to_move (unchanged, from the model).

    Parameters
    ----------
    spk_tensor : (n_trial, T_max, n_neuron) array
        Spike counts per trial/event.
    model : GPLVM jump model
        Trained model. The function will **copy** it first, then set
        `model_copy.custom_transition_kernel = kernel` for each kernel (this also refreshes
        cached log-transition matrices).
    continuous_transition_mat_d : dict or list
        Either:
        - dict: name -> (K, K) kernel
        - list: list of (K, K) kernels (names auto-generated unless kernel_names is provided)
    tensor_pad_mask : (n_trial, T_max, 1) bool, optional
        Trial padding mask from `trial_analysis.bin_spike_train_to_trial_based`.
        If None, treat all bins as valid (i.e. no padding).
    dt : float
        Decode bin size (seconds). Passed to `decoder_trial.decode_trials_padded_vmapped`.
    gain : float
        Multiplicative gain on tuning (Hz) before converting to expected counts/bin.
    model_fit_dt : float
        Bin size (seconds) used when fitting `model.tuning`. If `model.tuning` is in
        units of counts/bin, then tuning is converted to Hz by dividing by model_fit_dt.
        Default 0.1.
    time_l : array-like, optional
        If provided, must be a 1D array of timestamps for **valid bins only**,
        in trial-major concatenation order (same convention as
        `poor_man_gplvm.supervised_analysis.decoder_supervised` tensor mode).
        Length must equal the number of valid bins across all events.
        When provided, the function wraps time-concat posteriors into `nap.TsdFrame`
        and returns per-event posteriors by slicing those `TsdFrame`s using
        `start_index` / `end_index`.
    kernel_names : list of str, optional
        If provided, must match number of kernels and sets output ordering.

    Returns
    -------
    dict with:
        kernel_names : list of str
        log_marginal : dict, kernel_name -> (n_trial,) array of log marginal likelihoods
        log_marginal_total : dict, kernel_name -> sum of log_marginal across trials
        posterior_latent_marg : dict, kernel_name -> (n_valid, n_latent) array or nap.TsdFrame (if time_l provided)
        posterior_dynamics_marg : dict, kernel_name -> (n_valid, n_dyn) array or nap.TsdFrame (if time_l provided)
        posterior_latent_marg_per_event : dict, kernel_name -> list of per-event arrays/TsdFrames
        posterior_dynamics_marg_per_event : dict, kernel_name -> list of per-event arrays/TsdFrames
        log_marginal_category_per_event_df : pd.DataFrame (n_event x n_category)
        prob_category_per_event_df : pd.DataFrame (n_event x n_category)
        n_time_per_event : (n_event,) array of valid-bin counts per event
        start_index : (n_event,) start indices into valid-bin concat arrays
        end_index : (n_event,) end indices (exclusive) into valid-bin concat arrays
    """
    if isinstance(continuous_transition_mat_d, dict):
        if kernel_names is None:
            kernel_names = list(continuous_transition_mat_d.keys())
        kernels = [continuous_transition_mat_d[k] for k in kernel_names]
    else:
        kernels = list(continuous_transition_mat_d)
        if kernel_names is None:
            kernel_names = [f"kernel_{i}" for i in range(len(kernels))]
    if len(kernel_names) != len(kernels):
        raise ValueError("kernel_names must have same length as kernels")

    model_copy = pickle.loads(pickle.dumps(model))

    tensor_pad_mask_ = tensor_pad_mask
    if tensor_pad_mask_ is None:
        n_trial, t_max = int(spk_tensor.shape[0]), int(spk_tensor.shape[1])
        tensor_pad_mask_ = np.ones((n_trial, t_max, 1), dtype=bool)
    valid_mask_all = np.asarray(tensor_pad_mask_)[:, :, 0].astype(bool)  # (n_trial, T_max)
    trial_lengths = valid_mask_all.sum(axis=1).astype(int)
    ends = np.cumsum(trial_lengths).astype(int)
    starts = np.concatenate([np.array([0], dtype=int), ends[:-1]])
    start_index = starts
    end_index = ends
    n_time_per_event = trial_lengths

    if neuron_mask is None:
        neuron_mask = getattr(model_copy, "neuron_mask", None)
    if ma_latent is None:
        ma_latent = getattr(model_copy, "ma_latent", None)

    tuning_hz = np.asarray(model_copy.tuning) / float(model_fit_dt)

    orig_kernel = getattr(model_copy, "custom_transition_kernel", None)

    log_marginal_d = {}
    log_marginal_total_d = {}
    posterior_latent_marg_d = {}
    posterior_dynamics_marg_d = {}
    posterior_latent_marg_per_event_d = {}
    posterior_dynamics_marg_per_event_d = {}

    time_l_ = None
    if time_l is not None:
        time_l_ = np.asarray(time_l)
        if time_l_.ndim != 1:
            raise ValueError("time_l must be 1D (valid-bin concatenated time; trial-major).")
        total_valid = int(ends[-1]) if ends.size else 0
        if int(time_l_.shape[0]) != total_valid:
            raise ValueError("time_l length must equal total number of valid bins across events.")

    try:
        for name, kernel in zip(kernel_names, kernels):
            model_copy.custom_transition_kernel = kernel
            res = pdt.decode_trials_padded_vmapped(
                spk_tensor,
                tuning_hz,
                model_copy.log_latent_transition_kernel_l,
                model_copy.log_dynamics_transition_kernel,
                tensor_pad_mask=tensor_pad_mask_,
                dt=dt,
                gain=gain,
                neuron_mask=neuron_mask,
                ma_latent=ma_latent,
                n_trial_per_chunk=n_trial_per_chunk,
                prior_magnifier=prior_magnifier,
                return_numpy=return_numpy,
            )
            lml = res["log_marginal"]
            log_marginal_d[name] = lml
            log_marginal_total_d[name] = float(np.sum(lml))

            # Match `decoder_supervised` tensor-mode convention:
            # flatten (trial, time) -> valid-bin concat, then marginalize from posterior_all.
            log_post_padded = np.asarray(res["log_post_padded"])  # (n_trial, T_max, n_dyn, n_latent), NaN on padded
            n_trial, t_max, n_dyn, n_latent = log_post_padded.shape
            log_post_flat = log_post_padded.reshape((n_trial * t_max, n_dyn, n_latent))
            mask_flat = valid_mask_all.reshape((n_trial * t_max,))
            log_posterior_all = log_post_flat[mask_flat]
            posterior_all = np.exp(log_posterior_all)  # (n_valid, n_dyn, n_latent)
            posterior_latent_marg = posterior_all.sum(axis=1)    # (n_valid, n_latent)
            posterior_dynamics_marg = posterior_all.sum(axis=2)  # (n_valid, n_dyn)

            if time_l_ is not None:
                posterior_latent_marg_d[name] = nap.TsdFrame(
                    t=time_l_,
                    d=posterior_latent_marg,
                    columns=np.arange(n_latent),
                )
                dyn_cols = ['continuous', 'jump'] if int(n_dyn) == 2 else np.arange(n_dyn)
                posterior_dynamics_marg_d[name] = nap.TsdFrame(
                    t=time_l_,
                    d=posterior_dynamics_marg,
                    columns=dyn_cols,
                )
                # per-event: slice the big TsdFrame (decoder_supervised convention)
                lat_tsdf = posterior_latent_marg_d[name]
                dyn_tsdf = posterior_dynamics_marg_d[name]
                posterior_latent_marg_per_event_d[name] = [lat_tsdf[s:e] for s, e in zip(starts, ends) if e > s]
                posterior_dynamics_marg_per_event_d[name] = [dyn_tsdf[s:e] for s, e in zip(starts, ends) if e > s]
            else:
                posterior_latent_marg_d[name] = posterior_latent_marg
                posterior_dynamics_marg_d[name] = posterior_dynamics_marg
                posterior_latent_marg_per_event_d[name] = [posterior_latent_marg[s:e] for s, e in zip(starts, ends)]
                posterior_dynamics_marg_per_event_d[name] = [posterior_dynamics_marg[s:e] for s, e in zip(starts, ends)]

            print(f"[decode_compare_transition_kernels] {name}: log_marginal_total={log_marginal_total_d[name]:.2f}")
    finally:
        # restore original
        model_copy.custom_transition_kernel = orig_kernel

    log_marginal_category_per_event_df = pd.DataFrame({k: np.asarray(v) for k, v in log_marginal_d.items()})
    # posterior over categories per event
    log_marginal_mat = log_marginal_category_per_event_df.to_numpy()
    row_max = np.max(log_marginal_mat, axis=1, keepdims=True)
    prob_mat = np.exp(log_marginal_mat - row_max)
    prob_mat = prob_mat / np.sum(prob_mat, axis=1, keepdims=True)
    prob_category_per_event_df = pd.DataFrame(prob_mat, columns=log_marginal_category_per_event_df.columns, index=log_marginal_category_per_event_df.index)

    out = {
        "kernel_names": kernel_names,
        "log_marginal": log_marginal_d,
        "log_marginal_total": log_marginal_total_d,
        "posterior_latent_marg": posterior_latent_marg_d,
        "posterior_dynamics_marg": posterior_dynamics_marg_d,
        "posterior_latent_marg_per_event": posterior_latent_marg_per_event_d,
        "posterior_dynamics_marg_per_event": posterior_dynamics_marg_per_event_d,
        "log_marginal_category_per_event_df": log_marginal_category_per_event_df,
        "prob_category_per_event_df": prob_category_per_event_df,
        "n_time_per_event": n_time_per_event,
        "start_index": start_index,
        "end_index": end_index,
    }
    if time_l_ is not None:
        out["time_l"] = time_l_
    return out


def decode_with_multiple_transition_kernels(
    spk_tensor,
    model,
    continuous_transition_mat_d: Union[Dict[str, np.ndarray], List[np.ndarray]],
    tensor_pad_mask=None,
    *,
    dt: float = 0.02,
    gain: float = 1.0,
    model_fit_dt: float = 0.1,
    time_l=None,
    kernel_names: Optional[List[str]] = None,
    neuron_mask=None,
    ma_latent=None,
    n_trial_per_chunk: int = 400,
    prior_magnifier: float = 1.0,
    return_numpy: bool = True,
    p_stay=None,
    p_dynamics_transmat=None,
) -> Dict[str, Any]:
    """
    Decode trial-tensor spike counts under one multi-dynamics model (list of latent
    transition matrices + uniform fragmented). Returns posterior over dynamics; per-event
    summary is mean-in-time of posterior_dynamics_marg (no model-comparison hierarchy).

    Parameters
    ----------
    spk_tensor, model, tensor_pad_mask, dt, gain, model_fit_dt, time_l, kernel_names,
    neuron_mask, ma_latent, n_trial_per_chunk, prior_magnifier, return_numpy
        Same as decode_compare_transition_kernels.
    continuous_transition_mat_d : dict or list
        Dict name -> (L,L) or list of (L,L). Latent transition per behavior; last dynamics = fragmented (uniform).
    p_stay : float, optional
        Passed to create_transition_prob_from_transmat_list. Default from model if neither p_stay nor p_dynamics_transmat.
    p_dynamics_transmat : (K+1,K+1) array, optional
        Full dynamics transition matrix; overrides p_stay.

    Returns
    -------
    kernel_names (list, includes "fragmented"), posterior_latent_marg, posterior_dynamics_marg,
    posterior_latent_marg_per_event, posterior_dynamics_marg_per_event,
    mean_prob_category_per_event_df (mean in time per event), n_time_per_event, start_index, end_index,
    time_l (if provided), log_marginal.
    """
    if isinstance(continuous_transition_mat_d, dict):
        if kernel_names is None:
            kernel_names = list(continuous_transition_mat_d.keys())
        transmat_list = [continuous_transition_mat_d[k] for k in kernel_names]
    else:
        transmat_list = list(continuous_transition_mat_d)
        if kernel_names is None:
            kernel_names = [f"kernel_{i}" for i in range(len(transmat_list))]
    if len(kernel_names) != len(transmat_list):
        raise ValueError("kernel_names must have same length as transmat list")
    dyn_names = list(kernel_names) + ["fragmented"]

    tensor_pad_mask_ = tensor_pad_mask
    if tensor_pad_mask_ is None:
        n_trial, t_max = int(spk_tensor.shape[0]), int(spk_tensor.shape[1])
        tensor_pad_mask_ = np.ones((n_trial, t_max, 1), dtype=bool)
    valid_mask_all = np.asarray(tensor_pad_mask_)[:, :, 0].astype(bool)
    trial_lengths = valid_mask_all.sum(axis=1).astype(int)
    ends = np.cumsum(trial_lengths).astype(int)
    starts = np.concatenate([np.array([0], dtype=int), ends[:-1]])
    start_index = starts
    end_index = ends
    n_time_per_event = trial_lengths

    if neuron_mask is None:
        neuron_mask = getattr(model, "neuron_mask", None)
    if ma_latent is None:
        ma_latent = getattr(model, "ma_latent", None)

    tuning_hz = np.asarray(model.tuning) / float(model_fit_dt)
    possible_latent_bin = model.possible_latent_bin

    if p_dynamics_transmat is not None:
        latent_transition_kernel_l, log_latent_transition_kernel_l, dynamics_transition_kernel, log_dynamics_transition_kernel = gpk.create_transition_prob_from_transmat_list(
            possible_latent_bin, transmat_list, p_dynamics_transmat=p_dynamics_transmat)
    else:
        if p_stay is None:
            p_stay = max(1.0 - model.p_jump_to_move, 1.0 - model.p_move_to_jump)
        latent_transition_kernel_l, log_latent_transition_kernel_l, dynamics_transition_kernel, log_dynamics_transition_kernel = gpk.create_transition_prob_from_transmat_list(
            possible_latent_bin, transmat_list, p_stay=p_stay)

    res = pdt.decode_trials_padded_vmapped(
        spk_tensor,
        tuning_hz,
        np.asarray(log_latent_transition_kernel_l),
        np.asarray(log_dynamics_transition_kernel),
        tensor_pad_mask=tensor_pad_mask_,
        dt=dt,
        gain=gain,
        neuron_mask=neuron_mask,
        ma_latent=ma_latent,
        n_trial_per_chunk=n_trial_per_chunk,
        prior_magnifier=prior_magnifier,
        return_numpy=return_numpy,
    )
    log_post_padded = np.asarray(res["log_post_padded"])
    n_trial, t_max, n_dyn, n_latent = log_post_padded.shape
    log_post_flat = log_post_padded.reshape((n_trial * t_max, n_dyn, n_latent))
    mask_flat = valid_mask_all.reshape((n_trial * t_max,))
    log_posterior_all = log_post_flat[mask_flat]
    posterior_all = np.exp(log_posterior_all)
    posterior_latent_marg = posterior_all.sum(axis=1)
    posterior_dynamics_marg = posterior_all.sum(axis=2)

    time_l_ = None
    if time_l is not None:
        time_l_ = np.asarray(time_l)
        if time_l_.ndim != 1:
            raise ValueError("time_l must be 1D")
        total_valid = int(ends[-1]) if ends.size else 0
        if int(time_l_.shape[0]) != total_valid:
            raise ValueError("time_l length must equal total valid bins")

    posterior_latent_marg_per_event = [posterior_latent_marg[s:e] for s, e in zip(starts, ends)]
    posterior_dynamics_marg_per_event = [posterior_dynamics_marg[s:e] for s, e in zip(starts, ends)]

    mean_posterior_per_event = np.array([p.mean(axis=0) for p in posterior_dynamics_marg_per_event])
    mean_prob_category_per_event_df = pd.DataFrame(mean_posterior_per_event, columns=dyn_names)

    if time_l_ is not None:
        posterior_latent_marg_tsdf = nap.TsdFrame(t=time_l_, d=posterior_latent_marg, columns=np.arange(n_latent))
        dyn_cols = np.arange(n_dyn)
        posterior_dynamics_marg_tsdf = nap.TsdFrame(t=time_l_, d=posterior_dynamics_marg, columns=dyn_cols)
        posterior_latent_marg_per_event = [posterior_latent_marg_tsdf[s:e] for s, e in zip(starts, ends)]
        posterior_dynamics_marg_per_event = [posterior_dynamics_marg_tsdf[s:e] for s, e in zip(starts, ends)]

    out = {
        "kernel_names": dyn_names,
        "posterior_latent_marg": posterior_latent_marg,
        "posterior_dynamics_marg": posterior_dynamics_marg,
        "posterior_latent_marg_per_event": posterior_latent_marg_per_event,
        "posterior_dynamics_marg_per_event": posterior_dynamics_marg_per_event,
        "mean_prob_category_per_event_df": mean_prob_category_per_event_df,
        "n_time_per_event": n_time_per_event,
        "start_index": start_index,
        "end_index": end_index,
        "log_marginal": res["log_marginal"],
    }
    if time_l_ is not None:
        out["time_l"] = time_l_
    return out


