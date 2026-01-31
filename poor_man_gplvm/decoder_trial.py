"""
state-space decoder for trial-based data (trial tensor inputs). Different from decoder.py, it's vmapped over trials, accepts tensor shaped spike counts (n_trial, T_max, n_neuron) and a mask for valid bins (n_trial, T_max, 1), both from trial_analysis.bin_spike_train_to_trial_based

Main entrypoint:
- decode_trials_padded_vmapped

Notes
- Tuning is assumed to be in Hz; we convert to expected counts/bin via `dt`.
"""

import numpy as np

import jax
import jax.numpy as jnp
import jax.scipy as jscipy

import functools


def _to_numpy(x):
    if isinstance(x, dict):
        return {k: _to_numpy(v) for k, v in x.items()}
    if isinstance(x, list):
        return [_to_numpy(v) for v in x]
    if isinstance(x, jax.Array):
        return np.asarray(jax.device_get(x))
    return x


def _logsumexp(x, axis=None, keepdims=False):
    return jscipy.special.logsumexp(x, axis=axis, keepdims=keepdims)


def _init_log_posterior(n_dyn, ma_latent):
    """
    ma_latent: (n_latent,) bool
    returns: (n_dyn, n_latent) log posterior init
    """
    ma_latent = jnp.asarray(ma_latent).astype(bool)
    n_latent_eff = jnp.maximum(jnp.sum(ma_latent).astype(jnp.float32), 1.0)
    logp = -jnp.log(jnp.asarray(n_dyn, dtype=jnp.float32) * n_latent_eff)
    log_post = jnp.where(ma_latent[None, :], logp, -1e20)
    return log_post


@functools.partial(jax.jit, static_argnames=("prior_magnifier",))
def _filter_one_step_masked(
    carry,
    x,
    *,
    log_latent_transition_kernel_l,
    log_dynamics_transition_kernel,
    prior_magnifier=1.0,
):
    """
    carry: (log_post_prev, log_marginal_prev)
      - log_post_prev: (n_dyn, n_latent)
    x: (ll_curr, is_valid)
      - ll_curr: (n_latent,)
      - is_valid: bool
    """
    log_post_prev, log_marg_prev = carry
    ll_curr, is_valid = x

    # dynamics transition: prev_dyn -> curr_dyn
    tmp = log_post_prev[:, None, :] + log_dynamics_transition_kernel[:, :, None]  # (n_dyn_prev, n_dyn_curr, n_latent_prev)
    log_p_tuning_prev_nontuning_curr_prior = _logsumexp(tmp, axis=0)  # (n_dyn_curr, n_latent_prev)

    # latent transition conditioned on curr_dyn: prev_latent -> curr_latent
    tmp2 = log_p_tuning_prev_nontuning_curr_prior[:, :, None] + log_latent_transition_kernel_l  # (n_dyn, n_latent_prev, n_latent_curr)
    log_prior_curr = _logsumexp(tmp2, axis=1)  # (n_dyn, n_latent)

    log_post_curr_ = log_prior_curr * prior_magnifier + ll_curr[None, :]
    log_marg_ratio = _logsumexp(log_post_curr_)
    log_post_curr = log_post_curr_ - log_marg_ratio
    log_marg_curr = log_marg_prev + log_marg_ratio

    # Skip padded bins: no update, no marginal change.
    log_post_next = jnp.where(is_valid, log_post_curr, log_post_prev)
    log_marg_next = jnp.where(is_valid, log_marg_curr, log_marg_prev)
    log_prior_out = jnp.where(is_valid, log_prior_curr, log_post_prev)
    carry_next = (log_post_next, log_marg_next)
    return carry_next, (log_post_next, log_prior_out)


@functools.partial(jax.jit, static_argnames=("prior_magnifier",))
def _filter_all_step_masked(
    ll_all,
    valid_mask,
    *,
    log_latent_transition_kernel_l,
    log_dynamics_transition_kernel,
    log_post_init,
    prior_magnifier=1.0,
):
    """
    ll_all: (T, n_latent)
    valid_mask: (T,) bool
    returns:
      - log_post_all: (T, n_dyn, n_latent)
      - log_marginal_final: scalar
      - log_prior_all: (T, n_dyn, n_latent) (prior before likelihood)
    """
    carry_init = (log_post_init, jnp.asarray(0.0, dtype=jnp.float32))
    xs = (ll_all, valid_mask)
    f = functools.partial(
        _filter_one_step_masked,
        log_latent_transition_kernel_l=log_latent_transition_kernel_l,
        log_dynamics_transition_kernel=log_dynamics_transition_kernel,
        prior_magnifier=prior_magnifier,
    )
    carry_final, (log_post_all, log_prior_all) = jax.lax.scan(f, carry_init, xs=xs)
    log_marginal_final = carry_final[1]
    return log_post_all, log_marginal_final, log_prior_all


@functools.partial(jax.jit, static_argnames=())
def _smooth_one_step_masked(
    carry,
    x,
    *,
    log_latent_transition_kernel_l,
    log_dynamics_transition_kernel,
):
    """
    carry: log_acausal_next (n_dyn, n_latent)
    x: (log_causal_curr, log_causal_prior_next, do_update)
    """
    log_acausal_next = carry
    log_causal_curr, log_causal_prior_next, do_update = x

    # If skipping, keep carry.
    def _update(args):
        log_acausal_next_, log_causal_curr_, log_causal_prior_next_ = args
        x_next_given_x_curr_I_next = log_latent_transition_kernel_l[None, :, :, :]  # (n_dyn_curr, n_dyn_next?, n_latent_curr, n_latent_next) after broadcast
        I_next_given_I_curr = log_dynamics_transition_kernel[:, :, None, None]
        post_prior_diff = log_acausal_next_ - log_causal_prior_next_
        post_prior_diff = post_prior_diff[None, :, None, :]
        inside_integral = (
            x_next_given_x_curr_I_next
            + I_next_given_I_curr
            + post_prior_diff
            + log_causal_curr_[:, None, :, None]
        )
        log_acausal_curr = _logsumexp(inside_integral, axis=(1, 3))
        return log_acausal_curr

    log_acausal_curr = jax.lax.cond(
        do_update,
        _update,
        lambda args: log_acausal_next,
        (log_acausal_next, log_causal_curr, log_causal_prior_next),
    )
    return log_acausal_curr, log_acausal_curr


@functools.partial(jax.jit, static_argnames=())
def _smooth_all_step_masked(
    log_causal_post_all,
    log_causal_prior_all,
    valid_mask,
    *,
    log_latent_transition_kernel_l,
    log_dynamics_transition_kernel,
):
    """
    log_causal_post_all: (T, n_dyn, n_latent)
    log_causal_prior_all: (T, n_dyn, n_latent) from filter (prior at each time)
    valid_mask: (T,) bool

    returns:
      - log_acausal_post_all: (T, n_dyn, n_latent)
    """
    # last time step is the init (even if padded, filter keeps it equal to last valid)
    carry_init = log_causal_post_all[-1]

    log_causal_curr_all = log_causal_post_all[:-1]
    log_causal_prior_next_all = log_causal_prior_all[1:]
    valid_curr = valid_mask[:-1]
    valid_next = valid_mask[1:]
    do_update = jnp.logical_and(valid_curr, valid_next)

    xs = (log_causal_curr_all, log_causal_prior_next_all, do_update)
    f = functools.partial(
        _smooth_one_step_masked,
        log_latent_transition_kernel_l=log_latent_transition_kernel_l,
        log_dynamics_transition_kernel=log_dynamics_transition_kernel,
    )
    _, log_acausal_prev_all = jax.lax.scan(f, carry_init, xs=xs, reverse=True)
    log_acausal_all = jnp.concatenate([log_acausal_prev_all, carry_init[None, :, :]], axis=0)
    return log_acausal_all


@functools.partial(jax.jit, static_argnames=("prior_magnifier",))
def _decode_one_trial_masked(
    ll_all,
    valid_mask,
    *,
    log_latent_transition_kernel_l,
    log_dynamics_transition_kernel,
    log_post_init,
    prior_magnifier=1.0,
):
    log_causal_post, log_marginal, log_causal_prior = _filter_all_step_masked(
        ll_all,
        valid_mask,
        log_latent_transition_kernel_l=log_latent_transition_kernel_l,
        log_dynamics_transition_kernel=log_dynamics_transition_kernel,
        log_post_init=log_post_init,
        prior_magnifier=prior_magnifier,
    )
    log_acausal_post = _smooth_all_step_masked(
        log_causal_post,
        log_causal_prior,
        valid_mask,
        log_latent_transition_kernel_l=log_latent_transition_kernel_l,
        log_dynamics_transition_kernel=log_dynamics_transition_kernel,
    )
    return log_acausal_post, log_marginal


def decode_trials_padded_vmapped(
    spk_tensor,
    tuning,
    log_latent_transition_kernel_l,
    log_dynamics_transition_kernel,
    *,
    tensor_pad_mask,
    dt=0.02,
    gain=1.0,
    neuron_mask=None,
    ma_latent=None,
    n_trial_per_chunk=400,
    prior_magnifier=1.0,
    return_numpy=True,
):
    """
    Trial-tensor decoding with a latent+dynamics HMM backend (JAX).

    Inputs
    - spk_tensor: (n_trial, T_max, n_neuron) binned spike counts (padded with 0).
    - tensor_pad_mask: (n_trial, T_max, 1) bool, True for valid bins.
      (from `poor_man_gplvm.trial_analysis.bin_spike_train_to_trial_based`)
    - tuning: (n_latent, n_neuron) firing rate in Hz.
    - log_latent_transition_kernel_l: (n_dyn, n_latent, n_latent) log transition for latent conditioned on dyn (prev_latent -> curr_latent).
    - log_dynamics_transition_kernel: (n_dyn, n_dyn) log transition for dyn (prev_dyn -> curr_dyn).

    Kwargs
    - dt: decode bin size (seconds).
    - gain: scalar multiply tuning (Hz) before converting to expected counts.
    - neuron_mask: (n_neuron,) bool/0-1. Default all True.
    - ma_latent: (n_latent,) bool. Default all True.
    - n_trial_per_chunk: chunk trials to limit peak memory.
    - prior_magnifier: tempering factor multiplying log-prior before adding likelihood (like other decoders in this repo).
    - return_numpy: return numpy arrays if True; else return JAX arrays.

    Returns (dict)
    - log_post_padded: (n_trial, T_max, n_dyn, n_latent) smoothed log posterior; padded bins are NaN.
    - log_marginal: (n_trial,) marginal log-likelihood per trial (only valid bins contribute).
    - trial_lengths: (n_trial,) number of valid bins.
    - tensor_pad_mask: passthrough.

    Cluster Jupyter example

```python
import numpy as np
import poor_man_gplvm.decoder_trial as decoder_trial

# spk_tensor: (n_trial, T_max, n_neuron)
# tensor_pad_mask: (n_trial, T_max, 1)
# tuning: (n_latent, n_neuron) in Hz
# log_latent_transition_kernel_l: (n_dyn, n_latent, n_latent)
# log_dynamics_transition_kernel: (n_dyn, n_dyn)

res = decoder_trial.decode_trials_padded_vmapped(
    spk_tensor,
    tuning,
    log_latent_transition_kernel_l,
    log_dynamics_transition_kernel,
    tensor_pad_mask=tensor_pad_mask,
    dt=0.02,
    gain=1.0,
    n_trial_per_chunk=200,
    prior_magnifier=1.0,
)

log_post = res["log_post_padded"]  # (n_trial, T_max, n_dyn, n_latent), NaN where padded
lml = res["log_marginal"]          # (n_trial,)
```
    """
    spk_tensor = jnp.asarray(spk_tensor)
    tensor_pad_mask = jnp.asarray(tensor_pad_mask).astype(bool)
    tuning = jnp.asarray(tuning)
    log_latent_transition_kernel_l = jnp.asarray(log_latent_transition_kernel_l)
    log_dynamics_transition_kernel = jnp.asarray(log_dynamics_transition_kernel)

    if neuron_mask is None:
        neuron_mask = jnp.ones((spk_tensor.shape[-1],), dtype=bool)
    else:
        neuron_mask = jnp.asarray(neuron_mask).astype(bool)

    if ma_latent is None:
        ma_latent = jnp.ones((tuning.shape[0],), dtype=bool)
    else:
        ma_latent = jnp.asarray(ma_latent).astype(bool)

    n_trial, t_max, n_neuron = spk_tensor.shape
    n_latent = tuning.shape[0]
    n_dyn = log_dynamics_transition_kernel.shape[0]

    # expected counts/bin
    lam = tuning * jnp.asarray(dt, dtype=jnp.float32) * jnp.asarray(gain, dtype=jnp.float32) + 1e-20  # (n_latent, n_neuron)
    log_lam = jnp.log(lam)

    valid_mask_all = tensor_pad_mask[:, :, 0]
    trial_lengths = jnp.sum(valid_mask_all, axis=1).astype(int)

    log_post_init = _init_log_posterior(n_dyn, ma_latent)

    # vmap over trials in a chunk
    decode_vmapped = jax.jit(
        jax.vmap(
            functools.partial(
                _decode_one_trial_masked,
                log_latent_transition_kernel_l=log_latent_transition_kernel_l,
                log_dynamics_transition_kernel=log_dynamics_transition_kernel,
                log_post_init=log_post_init,
                prior_magnifier=prior_magnifier,
            ),
            in_axes=(0, 0),
        )
    )

    log_post_out = []
    log_marg_out = []

    for st in range(0, int(n_trial), int(n_trial_per_chunk)):
        ed = min(int(n_trial), st + int(n_trial_per_chunk))
        y_chunk = spk_tensor[st:ed]  # (b, T, n_neuron)
        valid_mask = valid_mask_all[st:ed]  # (b, T)

        m = valid_mask[:, :, None] & neuron_mask[None, None, :]
        m = m.astype(jnp.float32)

        # Efficient Poisson ll:
        # sum_n m*(y*log(lam) - lam - gammaln(y+1))
        y_masked = y_chunk * m
        term1 = jnp.matmul(y_masked, log_lam.T)  # (b, T, n_latent)
        term2 = jnp.matmul(m, lam.T)  # (b, T, n_latent)
        term3 = jnp.sum(jscipy.special.gammaln(y_chunk + 1.0) * m, axis=-1)  # (b, T)
        ll = term1 - term2 - term3[:, :, None]
        ll = jnp.where(ma_latent[None, None, :], ll, -1e20)

        log_post_chunk, log_marg_chunk = decode_vmapped(ll, valid_mask)
        log_post_out.append(log_post_chunk)
        log_marg_out.append(log_marg_chunk)

    log_post_padded = jnp.concatenate(log_post_out, axis=0)  # (n_trial, T, n_dyn, n_latent)
    log_marginal = jnp.concatenate(log_marg_out, axis=0)  # (n_trial,)

    # mask padded bins in output
    log_post_padded = jnp.where(valid_mask_all[:, :, None, None], log_post_padded, jnp.nan)

    out = {
        "log_post_padded": log_post_padded,
        "log_marginal": log_marginal,
        "trial_lengths": trial_lengths,
        "tensor_pad_mask": tensor_pad_mask,
    }
    if bool(return_numpy):
        return _to_numpy(out)
    return out

