'''
decode the latent and dynamics
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

from poor_man_gplvm.decoder import filter_all_step
'''
_chunk: for dealing with longer data; chunk the data and only scan / vectorize within the chunk, for loop across chunks

y: observed data, n_neuron; y_l: n_time x n_neuron
tuning: n_latent x n_neuron
log_latent_transition_kernel_l: n_dynamics x n_latent x n_latent
log_dynamics_transition_kernel: n_dynamics x n_dynamics

ma_neuron: mask to make out neuron / (time); necessary for computing co-smoothing
ma_latent: mask out latent; necessary for downsampled_lml

'''


@jit
def smooth_one_step(carry,x,log_latent_transition_kernel_l,log_dynamics_transition_kernel):
    '''
    causal_prior here refers to the prior from filter, i.e. logp(x_k+1|o_1:k)
    '''
    log_acausal_posterior_next=carry # need "previous" smoother, i.e. next time step (we use next/prev to denote in time here, not in inference steps); 
    log_causal_posterior_curr,log_causal_prior_next=x
    # log_causal_prior_next = log_tuning_state_transition_kernel_l + log_non_tuning_transition_kernel + 

    # broadcast things into: (nontuning_curr, nontuning_next, tuning_curr, tuning_next)
    x_next_given_x_curr_I_next = log_latent_transition_kernel_l[None,:,:,:] # add the nontuning_curr dimension
    I_next_given_I_curr = log_dynamics_transition_kernel[:,:,None,None] # add the two tuning dimensions
    post_prior_diff = log_acausal_posterior_next - log_causal_prior_next
    post_prior_diff = post_prior_diff[None,:,None,:] # add the two curr dimensions
    
    inside_integral = x_next_given_x_curr_I_next + I_next_given_I_curr + post_prior_diff + log_causal_posterior_curr[:,None,:,None] # supply the two next dimensions in broadcast
    log_curr_next_joint = inside_integral # log p(x_k,x_k+1,I_k,I_k+1|O_1:k)
    log_acausal_posterior_curr = jscipy.special.logsumexp(inside_integral, axis = (1,3)) # logsumexp over the two "next" dimensions
    # to_return = (log_acausal_posterior_curr,log_curr_next_joint) # the joint is too large, come up with reduced version
    to_return = log_acausal_posterior_curr
    carry = log_acausal_posterior_curr

    return carry,to_return

    # # old way
    # inside_integral = x_next_given_x_curr_I_next + I_next_given_I_curr + post_prior_diff
    # inside_integral = jscipy.special.logsumexp(inside_integral, axis = (1,3)) # logsumexp over the two "next" dimensions

    # log_acausal_posterior_curr = log_causal_posterior_curr + inside_integral
    # return log_acausal_posterior_curr,log_acausal_posterior_curr

@jit
def smooth_all_step(log_causal_posterior_all, log_causal_prior_all,log_latent_transition_kernel_l,log_dynamics_transition_kernel,carry_init=None,):
    '''
    if carry_init is None: i.e. the last chunk, then the last of causal posterior is the first of acausal, scan the rest, and concatenate the two
    if carry_init is not None, then scan the whole causal, no need to concatenate
    '''
    if carry_init is None:
        do_concat=True
        carry_init = log_causal_posterior_all[-1]
        xs = (log_causal_posterior_all[:-1],log_causal_prior_all)  # causal prior and the acausal init have the same t+1 index, 1 more than the causal posterior; handled when fed in
    else:
        do_concat=False
        xs = (log_causal_posterior_all,log_causal_prior_all)

    
    f = partial(smooth_one_step,log_latent_transition_kernel_l=log_latent_transition_kernel_l,log_dynamics_transition_kernel=log_dynamics_transition_kernel)
    # carry_final, (log_acausal_posterior_all,log_acausal_curr_next_joint_all) = scan(f, carry_init, xs=xs,reverse=True)
    carry_final, log_acausal_posterior_all = scan(f, carry_init, xs=xs,reverse=True)
    if do_concat:
        log_acausal_posterior_all = jnp.concatenate([log_acausal_posterior_all,log_causal_posterior_all[-1][None,...]],axis=0)
    
    # return log_acausal_posterior_all,log_acausal_curr_next_joint_all
    return log_acausal_posterior_all

@jit
def get_loglikelihood_ma_poisson_gain(y, tuning, hyperparam, ma_neuron, ma_latent, dt=1., gain=1.):
    '''
    Modified Poisson likelihood with gain
    ma_neuron: n_neuron,
    ma_latent: n_latent,
    hyperparam: for consistency
    gain: scalar or broadcastable
    '''
    ll = jscipy.stats.poisson.logpmf(y, (tuning * gain * dt) + 1e-20) # n_pos x n_neuron
    
    ll_per_pos = (ll * ma_neuron[None,:]).sum(axis=1)  # n_latent
    ll_per_pos = jnp.where(ma_latent, ll_per_pos, -1e20)
    
    return ll_per_pos

def get_loglikelihood_ma_gaussian_gain(y, tuning, hyperparam, ma_neuron, ma_latent, dt=1., gain=1.):
    '''
    Modified Gaussian likelihood with gain
    '''
    noise_std = hyperparam['noise_std']
    ll = jscipy.stats.norm.logpdf(y, (tuning * gain * dt), noise_std)
    ll_per_pos = (ll * ma_neuron[None,:]).sum(axis=1)  # n_latent
    ll_per_pos = jnp.where(ma_latent, ll_per_pos, -1e20)
    return ll_per_pos

@partial(jit,static_argnames=['observation_model'])
def get_loglikelihood_ma_all_gain(y_l, tuning, hyperparam, ma_neuron, ma_latent, gain_l, observation_model='poisson'):
    '''
    Modified likelihood computation with gain
    gain_l: n_time, gain at each time point
    '''
    get_loglikelihood_ma = get_loglikelihood_ma_poisson_gain if observation_model == 'poisson' else get_loglikelihood_ma_gaussian_gain
    
    ma_neuron = jnp.broadcast_to(ma_neuron, y_l.shape)
    ll_per_pos_l = vmap(get_loglikelihood_ma, in_axes=(0, None, None, 0, None, None, 0))(
        y_l, tuning, hyperparam, ma_neuron, ma_latent, 1., gain_l)

    return ll_per_pos_l # n_time x n_pos

@partial(jit,static_argnames=['observation_model'])
def get_loglikelihood_ma_all_changing_dt_gain(y_l, tuning, hyperparam, ma_neuron, ma_latent, dt_l, gain_l, observation_model='poisson'):
    '''
    Modified likelihood with both changing dt and gain
    '''
    ma_neuron = jnp.broadcast_to(ma_neuron, y_l.shape)
    get_loglikelihood_ma = get_loglikelihood_ma_poisson_gain if observation_model == 'poisson' else get_loglikelihood_ma_gaussian_gain
    ll_per_pos_l = vmap(get_loglikelihood_ma, in_axes=(0, None, None, 0, None, 0, 0))(
        y_l, tuning, hyperparam, ma_neuron, ma_latent, dt_l, gain_l)

    return ll_per_pos_l # n_time x n_pos

@partial(jit,static_argnames=['observation_model'])
def get_naive_bayes_ma_gain(y_l, tuning, hyperparam, ma_neuron, ma_latent, dt_l=1, gain_l=1., observation_model='poisson'):
    '''
    Naive Bayes with gain
    '''
    dt_l = jnp.broadcast_to(dt_l, y_l.shape[0])
    gain_l = jnp.broadcast_to(gain_l, y_l.shape[0])
    ll_per_pos_l = get_loglikelihood_ma_all_changing_dt_gain(y_l, tuning, hyperparam, ma_neuron, ma_latent, dt_l, gain_l, observation_model=observation_model)
    log_marginal_l = jscipy.special.logsumexp(ll_per_pos_l, axis=-1, keepdims=True)
    log_post = ll_per_pos_l - log_marginal_l
    log_marginal = jnp.sum(log_marginal_l)
    return log_post, jnp.squeeze(log_marginal_l), log_marginal, ll_per_pos_l

def get_naive_bayes_ma_chunk_gain(y, tuning, hyperparam, ma_neuron, ma_latent, dt_l=1, n_time_per_chunk=10000, observation_model='poisson', gain_l=1.):
    '''
    Chunked naive Bayes with gain
    '''
    n_time_tot = y.shape[0]
    n_chunks = int(jnp.ceil(n_time_tot / n_time_per_chunk))
    dt_l = jnp.broadcast_to(dt_l, y.shape[0])
    gain_l = jnp.broadcast_to(gain_l, y.shape[0])

    ma_neuron = jnp.broadcast_to(ma_neuron, y.shape)
    
    log_post_l = []
    log_marginal_l_l = []
    log_marginal_l = []
    ll_per_pos_l_l = []
    for n in range(n_chunks):
        sl = slice(n * n_time_per_chunk, (n+1) * n_time_per_chunk)
        y_chunk = y[sl]
        ma_neuron_chunk = ma_neuron[sl]
        dt_l_chunk = dt_l[sl]
        gain_l_chunk = gain_l[sl]
        
        log_post, log_marginal_l_chunk, log_marginal, ll_per_pos_chunk = get_naive_bayes_ma_gain(
            y_chunk, tuning, hyperparam, ma_neuron_chunk, ma_latent, dt_l_chunk, gain_l_chunk, observation_model=observation_model)
        log_post_l.append(log_post)
        log_marginal_l_l.append(log_marginal_l_chunk)
        log_marginal_l.append(log_marginal)
        ll_per_pos_l_l.append(ll_per_pos_chunk)
    
    log_post_l = jnp.concatenate(log_post_l, axis=0)
    log_marginal_l = jnp.concatenate(log_marginal_l_l, axis=0)
    log_marginal_total = jnp.sum(jnp.array(log_marginal_l))
    ll_per_pos_l = jnp.concatenate(ll_per_pos_l_l, axis=0)
    
    return log_post_l, log_marginal_l, log_marginal_total, ll_per_pos_l

@partial(jit,static_argnames=['observation_model'])
def filter_all_step_combined_ma_gain(y, tuning, hyperparam, log_latent_transition_kernel_l, log_dynamics_transition_kernel, ma_neuron, ma_latent, carry_init=None, likelihood_scale=1, observation_model='poisson', gain_l=1.):
    '''
    Combined filter with gain
    '''
    gain_l = jnp.broadcast_to(gain_l, y.shape[0])
    log_likelihood_all = get_loglikelihood_ma_all_gain(y, tuning, hyperparam, ma_neuron, ma_latent, gain_l, observation_model=observation_model)
    log_posterior_all, log_marginal_final, log_prior_curr_all = filter_all_step(log_likelihood_all, log_latent_transition_kernel_l, log_dynamics_transition_kernel, carry_init=carry_init, likelihood_scale=likelihood_scale)
    return log_posterior_all, log_marginal_final, log_prior_curr_all, log_likelihood_all



def smooth_all_step_combined_ma_chunk_gain(y, tuning, hyperparam, log_latent_transition_kernel_l, log_dynamics_transition_kernel, ma_neuron, ma_latent=None, likelihood_scale=1,
                                    n_time_per_chunk=10000, observation_model='poisson', gain_l=1.
                                    ):
    '''
    Chunked forward-backward algorithm with gain
    '''
    n_time_tot = y.shape[0]
    n_chunks = int(jnp.ceil(n_time_tot / n_time_per_chunk))
    gain_l = jnp.broadcast_to(gain_l, y.shape[0])

    filter_carry_init = None
    log_causal_posterior_all_allchunk = []
    log_causal_prior_all_allchunk = []
    log_acausal_posterior_all_allchunk = []
    log_likelihood_all_allchunk = []

    if ma_latent is None:
        ma_latent = jnp.ones(tuning.shape[0])
    
    slice_l = []
    for n in range(n_chunks):
        sl = slice(n * n_time_per_chunk, (n+1) * n_time_per_chunk)
        slice_l.append(sl)
        y_chunk = y[sl]
        ma_neuron_chunk = jnp.broadcast_to(ma_neuron, y_chunk.shape)
        gain_l_chunk = gain_l[sl]
        
        log_causal_posterior_all, log_marginal_final, log_causal_prior_all, log_likelihood_all = filter_all_step_combined_ma_gain(
            y_chunk, tuning, hyperparam, log_latent_transition_kernel_l, log_dynamics_transition_kernel, 
            ma_neuron_chunk, ma_latent, carry_init=filter_carry_init, likelihood_scale=likelihood_scale, 
            observation_model=observation_model, gain_l=gain_l_chunk)
        
        filter_carry_init = (log_causal_posterior_all[-1], log_marginal_final)
        log_causal_posterior_all_allchunk.append(log_causal_posterior_all)
        log_causal_prior_all_allchunk.append(log_causal_prior_all)
        log_likelihood_all_allchunk.append(log_likelihood_all)
    
    log_causal_prior_all_ = jnp.concatenate(log_causal_prior_all_allchunk, axis=0)
    log_likelihood_all_concat = jnp.concatenate(log_likelihood_all_allchunk, axis=0)

    smooth_carry_init = None
    for n in range(n_chunks-1, -1, -1):
        sl = slice_l[n]
        log_causal_prior_all = log_causal_prior_all_[sl.start+1:sl.stop+1]
        log_causal_posterior_all = log_causal_posterior_all_allchunk[n]
        
        log_acausal_posterior_all, log_accumulated_joint_chunk = smooth_all_step(
            log_causal_posterior_all, log_causal_prior_all, log_latent_transition_kernel_l, 
            log_dynamics_transition_kernel, carry_init=smooth_carry_init)
        smooth_carry_init = (log_acausal_posterior_all[0], log_accumulated_joint_chunk)
        log_acausal_posterior_all_allchunk.append(log_acausal_posterior_all)
    
    log_acausal_posterior_all_allchunk.reverse()
    log_acausal_posterior_all = jnp.concatenate(log_acausal_posterior_all_allchunk, axis=0)
    log_causal_posterior_all = jnp.concatenate(log_causal_posterior_all_allchunk, axis=0)
    
    return log_acausal_posterior_all, log_marginal_final, log_causal_posterior_all, log_accumulated_joint_chunk, log_likelihood_all_concat