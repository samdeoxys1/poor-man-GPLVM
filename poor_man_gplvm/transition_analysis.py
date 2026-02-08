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
    dict keyed by epoch name, each value is transition summary dict from
    transition_from_posterior_in_intervalset.
    """
    return {
        name: transition_from_posterior_in_intervalset(post_lat_tsd, iset)
        for name, iset in epoch_dict.items()
    }

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
    kernel_names: Optional[List[str]] = None,
    keep_posteriors: bool = False,
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
    kernel_names : list of str, optional
        If provided, must match number of kernels and sets output ordering.
    keep_posteriors : bool
        If True, store posterior_latent_marginal_list for each kernel.

    Returns
    -------
    dict with:
        kernel_names : list of str
        log_marginal : dict, kernel_name -> (n_trial,) array of log marginal likelihoods
        log_marginal_total : dict, kernel_name -> sum of log_marginal across trials
        posteriors (optional) : dict, kernel_name -> posterior_latent_marginal_list
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

    if neuron_mask is None:
        neuron_mask = getattr(model_copy, "neuron_mask", None)
    if ma_latent is None:
        ma_latent = getattr(model_copy, "ma_latent", None)

    tuning_hz = np.asarray(model_copy.tuning) / float(model_fit_dt)

    orig_kernel = getattr(model_copy, "custom_transition_kernel", None)

    log_marginal_d = {}
    log_marginal_total_d = {}
    posteriors_d = {}

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
            if keep_posteriors:
                posteriors_d[name] = res["posterior_latent_marginal_list"]
            print(f"[decode_compare_transition_kernels] {name}: log_marginal_total={log_marginal_total_d[name]:.2f}")
    finally:
        # restore original
        model_copy.custom_transition_kernel = orig_kernel

    out = {
        "kernel_names": kernel_names,
        "log_marginal": log_marginal_d,
        "log_marginal_total": log_marginal_total_d,
        
    }
    out.update(posteriors_d)
    return out


