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
# import jlvm_trial_decoding as jtd


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
    return best_inverse_temperature,loss_l

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

# def decode_compare_transition_kernels(
#     spk_list: Union[List[np.ndarray], np.ndarray],
#     model,
#     kernels: List[np.ndarray],
#     *,
#     kernel_names: Optional[List[str]] = None,
#     decode_kwargs: Optional[Dict[str, Any]] = None,
#     keep_posteriors: bool = False,
# ) -> Dict[str, Any]:
#     """
#     Decode spk_list under multiple custom transition kernels and compare marginal likelihoods.

#     The custom kernel only affects the random-walk (non-jump) latent transition;
#     the jump latent transition stays uniform, and move/jump switching is governed
#     by p_move_to_jump / p_jump_to_move (unchanged).

#     Parameters
#     ----------
#     spk_list : list of (T_i, n_neuron) arrays or single (T, n_neuron) array
#         Spike counts per trial/event.
#     model : PoissonGPLVMJump1D
#         Trained model (will temporarily modify model.custom_transition_kernel).
#     kernels : list of (K, K) arrays
#         Custom transition kernels to compare. Will be normalized by gp_kernel helper.
#     kernel_names : list of str, optional
#         Names for each kernel; defaults to ["kernel_0", "kernel_1", ...].
#     decode_kwargs : dict, optional
#         Additional kwargs passed to decode_trials_padded_vmapped_poor_gplvm_backend_posteriors
#         (e.g. model_fit_binsize, decode_binsize, gain, hyperparam, neuron_mask).
#     keep_posteriors : bool
#         If True, store posterior_latent_marginal_list for each kernel.

#     Returns
#     -------
#     dict with:
#         kernel_names : list of str
#         log_marginal : dict, kernel_name -> (n_trial,) array of log marginal likelihoods
#         log_marginal_total : dict, kernel_name -> sum of log_marginal across trials
#         posteriors : dict (only if keep_posteriors=True), kernel_name -> posterior_latent_marginal_list
#     """
#     if kernel_names is None:
#         kernel_names = [f"kernel_{i}" for i in range(len(kernels))]
#     if len(kernel_names) != len(kernels):
#         raise ValueError("kernel_names must have same length as kernels")

#     if decode_kwargs is None:
#         decode_kwargs = {}

#     # save original kernel
#     orig_kernel = getattr(model, "custom_transition_kernel", None)

#     log_marginal_d = {}
#     log_marginal_total_d = {}
#     posteriors_d = {}

#     try:
#         for name, kernel in zip(kernel_names, kernels):
#             model.custom_transition_kernel = kernel
#             res = jtd.decode_trials_padded_vmapped_poor_gplvm_backend_posteriors(
#                 spk_list, model, **decode_kwargs
#             )
#             lml = res["log_marginal"]
#             log_marginal_d[name] = lml
#             log_marginal_total_d[name] = float(np.sum(lml))
#             if keep_posteriors:
#                 posteriors_d[name] = res["posterior_latent_marginal_list"]
#             print(f"[decode_compare_transition_kernels] {name}: log_marginal_total={log_marginal_total_d[name]:.2f}")
#     finally:
#         # restore original
#         model.custom_transition_kernel = orig_kernel

#     out = {
#         "kernel_names": kernel_names,
#         "log_marginal": log_marginal_d,
#         "log_marginal_total": log_marginal_total_d,
#     }
#     if keep_posteriors:
#         out["posteriors"] = posteriors_d
#     return out


