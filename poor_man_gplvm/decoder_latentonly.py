'''
Decoder functions for latent-only GPLVM models (no dynamics dimension)
Simplified versions of decoder.py functions with latent-only posteriors
'''

import jax.numpy as jnp
import jax
import jax.random as jr
import jax.scipy as jscipy
from jax.scipy.special import logsumexp
from jax import jit, vmap
from functools import partial
from jax.lax import scan
import tqdm

# Import likelihood functions from main decoder
from poor_man_gplvm.decoder import (
    get_loglikelihood_ma_poisson, 
    get_loglikelihood_ma_gaussian,
    get_loglikelihood_ma_all,
    get_loglikelihood_ma_all_changing_dt,
    get_naive_bayes_ma,
    get_naive_bayes_ma_chunk
)

'''
Latent-only decoder functions:
- Posterior shapes: n_time x n_latent (no dynamics dimension)
- Transition kernels: n_latent x n_latent (no dynamics)
- Simplified broadcasting without dynamics
'''

@jit
def filter_one_step_latent(carry, ll_curr, log_latent_transition_kernel, likelihood_scale=1):
    '''
    Simplified filter step for latent-only model
    carry: (log_posterior_prev, log_marginal_tillprev)
    ll_curr: n_latent - log likelihood for current timepoint
    log_latent_transition_kernel: n_latent x n_latent
    '''
    log_posterior_prev, log_marginal_tillprev = carry
    
    # Compute prior: p(x_t | x_{t-1}) * p(x_{t-1} | y_{1:t-1})
    log_prior_curr = logsumexp(
        log_posterior_prev[:, None] + log_latent_transition_kernel, 
        axis=0
    )
    
    # Compute posterior: p(x_t | y_{1:t}) ∝ p(y_t | x_t) * p(x_t | y_{1:t-1})
    log_post_curr_unnorm = log_prior_curr + likelihood_scale * ll_curr
    log_marginal_ratio_curr = logsumexp(log_post_curr_unnorm)
    log_post_curr = log_post_curr_unnorm - log_marginal_ratio_curr
    log_marginal_tillcurr = log_marginal_tillprev + log_marginal_ratio_curr
    
    carry_next = (log_post_curr, log_marginal_tillcurr)
    return carry_next, (log_post_curr, log_prior_curr, log_marginal_ratio_curr)

def filter_all_step_latent(ll_all, log_latent_transition_kernel, carry_init=None, likelihood_scale=1):
    '''
    Run causal filter for latent-only model
    ll_all: n_time x n_latent
    log_latent_transition_kernel: n_latent x n_latent
    '''
    n_latent = log_latent_transition_kernel.shape[0]
    
    if carry_init is None:
        log_posterior_init = jnp.log(jnp.ones(n_latent) / n_latent)
        carry_init = (log_posterior_init, jnp.array(0.))
    
    f = partial(
        filter_one_step_latent,
        log_latent_transition_kernel=log_latent_transition_kernel,
        likelihood_scale=likelihood_scale
    )
    carry_final, (log_posterior_all, log_prior_all, log_one_step_predictive_marginals) = scan(
        f, carry_init, xs=ll_all
    )
    log_marginal_final = carry_final[1]
    
    return log_posterior_all, log_marginal_final, log_prior_all, log_one_step_predictive_marginals

@partial(jit, static_argnames=['observation_model'])
def filter_all_step_combined_ma_latent(y, tuning, hyperparam, log_latent_transition_kernel, 
                                     ma_neuron, ma_latent, carry_init=None, likelihood_scale=1, 
                                     observation_model='poisson'):
    '''
    Combined filter step with likelihood computation for latent-only model
    '''
    ll_all = get_loglikelihood_ma_all(y, tuning, hyperparam, ma_neuron, ma_latent, 
                                    observation_model=observation_model)
    log_posterior_all, log_marginal_final, log_prior_all, log_one_step_predictive_marginals = filter_all_step_latent(
        ll_all, log_latent_transition_kernel, carry_init=carry_init, likelihood_scale=likelihood_scale
    )
    return log_posterior_all, log_marginal_final, log_prior_all, log_one_step_predictive_marginals

@jit  
def smooth_one_step_latent(carry, x, log_latent_transition_kernel):
    '''
    Simplified smoother step for latent-only model
    carry: (log_acausal_posterior_next, log_accumulated_joint)
    x: (log_causal_posterior_curr, log_causal_prior_next)
    '''
    log_acausal_posterior_next, log_accumulated_joint = carry
    log_causal_posterior_curr, log_causal_prior_next = x
    
    # Compute smoothed posterior: p(x_t | y_{1:T})
    # Broadcasting: (n_latent_curr, n_latent_next)
    post_prior_diff = log_acausal_posterior_next - log_causal_prior_next
    inside_integral = (log_latent_transition_kernel + 
                      post_prior_diff[None, :] + 
                      log_causal_posterior_curr[:, None])
    
    # Joint distribution for transition probabilities
    log_curr_next_joint = inside_integral  # n_latent_curr x n_latent_next
    
    # Marginal posterior at current time
    log_acausal_posterior_curr = logsumexp(inside_integral, axis=1)
    
    # Accumulate joint for transition probability computation
    log_accumulated_joint_new = jnp.logaddexp(log_accumulated_joint, log_curr_next_joint)
    
    carry_new = (log_acausal_posterior_curr, log_accumulated_joint_new)
    return carry_new, log_acausal_posterior_curr

@jit
def smooth_all_step_latent(log_causal_posterior_all, log_causal_prior_all, 
                         log_latent_transition_kernel, carry_init=None):
    '''
    Run backward smoother for latent-only model
    '''
    if carry_init is None:
        do_concat = True
        n_latent = log_latent_transition_kernel.shape[0]
        # Initialize accumulated joint with very negative values
        carry_init = (log_causal_posterior_all[-1], 
                     jnp.ones((n_latent, n_latent)) * (-1e40))
        xs = (log_causal_posterior_all[:-1], log_causal_prior_all)
    else:
        do_concat = False
        xs = (log_causal_posterior_all, log_causal_prior_all)
    
    f = partial(smooth_one_step_latent, log_latent_transition_kernel=log_latent_transition_kernel)
    carry_final, log_acausal_posterior_all = scan(f, carry_init, xs=xs, reverse=True)
    
    # Extract accumulated joint from final carry
    _, log_accumulated_joint_final = carry_final
    
    if do_concat:
        log_acausal_posterior_all = jnp.concatenate([
            log_acausal_posterior_all, 
            log_causal_posterior_all[-1][None, ...]
        ], axis=0)
    
    return log_acausal_posterior_all, log_accumulated_joint_final

def smooth_all_step_combined_ma_chunk_latent(y, tuning, hyperparam, log_latent_transition_kernel,
                                           ma_neuron, ma_latent=None, likelihood_scale=1,
                                           n_time_per_chunk=10000, observation_model='poisson'):
    '''
    Chunked forward-backward algorithm for latent-only model
    '''
    n_time_tot = y.shape[0]
    n_chunks = int(jnp.ceil(n_time_tot / n_time_per_chunk))
    
    if ma_latent is None:
        ma_latent = jnp.ones(tuning.shape[0])
    
    # Forward pass (filter) - chunk by chunk
    filter_carry_init = None
    log_causal_posterior_all_allchunk = []
    log_causal_prior_all_allchunk = []
    log_one_step_predictive_marginals_allchunk = []
    slice_l = []
    
    for n in range(n_chunks):
        sl = slice(n * n_time_per_chunk, (n+1) * n_time_per_chunk)
        slice_l.append(sl)
        y_chunk = y[sl]
        ma_neuron_chunk = jnp.broadcast_to(ma_neuron, y_chunk.shape)
        
        log_causal_posterior_all, log_marginal_final, log_causal_prior_all, log_one_step_predictive_marginals = filter_all_step_combined_ma_latent(
            y_chunk, tuning, hyperparam, log_latent_transition_kernel,
            ma_neuron_chunk, ma_latent, carry_init=filter_carry_init,
            likelihood_scale=likelihood_scale, observation_model=observation_model
        )
        
        filter_carry_init = (log_causal_posterior_all[-1], log_marginal_final)
        log_causal_posterior_all_allchunk.append(log_causal_posterior_all)
        log_causal_prior_all_allchunk.append(log_causal_prior_all)
        log_one_step_predictive_marginals_allchunk.append(log_one_step_predictive_marginals)
    
    log_causal_prior_all_ = jnp.concatenate(log_causal_prior_all_allchunk, axis=0)
    log_one_step_predictive_marginals_allchunk = jnp.concatenate(log_one_step_predictive_marginals_allchunk, axis=0)
    
    # Backward pass (smoother) - reverse chunk order
    smooth_carry_init = None
    log_acausal_posterior_all_allchunk = []
    
    for n in range(n_chunks-1, -1, -1):
        sl = slice_l[n]
        log_causal_prior_all = log_causal_prior_all_[sl.start+1:sl.stop+1]
        log_causal_posterior_all = log_causal_posterior_all_allchunk[n]
        
        log_acausal_posterior_all, log_accumulated_joint_chunk = smooth_all_step_latent(
            log_causal_posterior_all, log_causal_prior_all, 
            log_latent_transition_kernel, carry_init=smooth_carry_init
        )
        smooth_carry_init = (log_acausal_posterior_all[0], log_accumulated_joint_chunk)
        log_acausal_posterior_all_allchunk.append(log_acausal_posterior_all)
    
    log_acausal_posterior_all_allchunk.reverse()
    log_acausal_posterior_all = jnp.concatenate(log_acausal_posterior_all_allchunk, axis=0)
    log_causal_posterior_all = jnp.concatenate(log_causal_posterior_all_allchunk, axis=0)
    
    return (log_acausal_posterior_all, log_marginal_final, log_causal_posterior_all, 
            log_one_step_predictive_marginals_allchunk, log_accumulated_joint_chunk)

@jit
def compute_transition_posterior_prob_latent(log_accumulated_joint_total):
    '''
    Compute transition probabilities for latent-only model
    log_accumulated_joint_total: n_latent x n_latent
    '''
    # Normalize joint distribution
    log_joint_latent = log_accumulated_joint_total - logsumexp(log_accumulated_joint_total)
    
    # Compute conditional transition probabilities
    log_transition_latent = log_joint_latent - logsumexp(log_joint_latent, axis=1, keepdims=True)
    
    # Convert to probabilities
    p_joint_latent = jnp.exp(log_joint_latent)
    p_transition_latent = jnp.exp(log_transition_latent)
    
    transition_posterior_prob_res = {
        'p_joint_latent': p_joint_latent,
        'p_transition_latent': p_transition_latent,
        'log_joint_latent': log_joint_latent,
        'log_transition_latent': log_transition_latent,
    }
    
    return transition_posterior_prob_res 