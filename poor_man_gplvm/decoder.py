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

'''
_chunk: for dealing with longer data; chunk the data and only scan / vectorize within the chunk, for loop across chunks
ma: mask to make out neuron / (time); necessary for computing co-smoothing
y: observed data, n_neuron; y_l: n_time x n_neuron
tuning: n_latent x n_neuron
log_latent_transition_kernel_l: n_dynamics x n_latent x n_latent
log_dynamics_transition_kernel: n_dynamics x n_dynamics

'''

@jit
def get_loglikelihood_ma(y,tuning,ma,dt=1.):
    '''
    ma: n_neuron,
    '''
    ll = jscipy.stats.poisson.logpmf(y,(tuning*dt)+1e-20) # n_pos x n_neuron
    
    ll_per_pos = (ll * ma[None,:]).sum(axis=1)
    return ll_per_pos
@jit
def get_loglikelihood_ma_all(y_l, tuning, ma):
    
    # ll_per_pos_l = vmap(get_loglikelihood_ma,in_axes=(0,None,None))(y_l,tuning,ma)
    
    # spatio-temporal mask
    ma = jnp.broadcast_to(ma,y_l.shape)
    ll_per_pos_l = vmap(get_loglikelihood_ma,in_axes=(0,None,0))(y_l,tuning,ma)

    return ll_per_pos_l # n_time x n_pos

@jit
def get_loglikelihood_ma_all_changing_dt(y_l, tuning, ma, dt_l):
    '''
    for each time, multiply tuning by dt in dt_l
    '''
    # ll_per_pos_l = vmap(get_loglikelihood_ma,in_axes=(0,None,None))(y_l,tuning,ma)
    
    # spatio-temporal mask
    ma = jnp.broadcast_to(ma,y_l.shape)
    ll_per_pos_l = vmap(get_loglikelihood_ma,in_axes=(0,None,0,0))(y_l,tuning,ma,dt_l)

    return ll_per_pos_l # n_time x n_pos


@jit
def get_naive_bayes_ma(y_l,tuning,ma,dt_l=1):
    dt_l = jnp.broadcast_to(dt_l,y_l.shape[0])
    ll_per_pos_l = get_loglikelihood_ma_all_changing_dt(y_l, tuning, ma,dt_l)
    log_marginal_l = jscipy.special.logsumexp(ll_per_pos_l,axis=-1,keepdims=True)
    log_post = ll_per_pos_l - log_marginal_l
    log_marginal = jnp.sum(log_marginal_l)
    return log_post, log_marginal


def get_naive_bayes_ma_chunk(y,tuning,ma,dt_l=1,n_time_per_chunk=10000):
    n_time_tot = y.shape[0]
    n_chunks = int( jnp.ceil(n_time_tot / n_time_per_chunk))
    dt_l= jnp.broadcast_to(dt_l,y.shape[0])

    # spatio-temporal mask
    ma = jnp.broadcast_to(ma,y.shape)
    slice_l = []
    log_post_l = []
    log_marginal_l = []
    for n in range(n_chunks):
        sl = slice((n) * n_time_per_chunk , (n+1) * n_time_per_chunk )
        y_chunk = y[sl]
        ma_chunk = ma[sl]
        dt_l_chunk = dt_l[sl]
        log_post,log_marginal = get_naive_bayes_ma(y_chunk,tuning,ma_chunk,dt_l_chunk)
        log_post_l.append(log_post)
        log_marginal_l.append(log_marginal)
    log_post_l = jnp.concatenate(log_post_l,axis=0)
    log_marginal_final = jnp.sum(jnp.array(log_marginal_l))
    return log_post_l, log_marginal_final

def filter_one_step(carry,ll_curr,log_latent_transition_kernel_l,log_dynamics_transition_kernel,likelihood_scale=1):
    '''
    log_posterior_prev: n_dynamics x n_latent,
    ll_curr: n_latent
    likelihood_scale: multiply the log likelihood, such that it plays a more important role
    
    '''
    log_posterior_prev, log_marginal_tillprev=carry
    log_p_tuning_prev_nontuning_curr_prior_ =log_posterior_prev[:,None,:] + log_latent_transition_kernel_l[:,:,None] # adding a target dimemsion, n_prev_nontuning x n_curr_nontuning x n_tuning # p(x_{t-1},I_k|O_{1:t-1}), by marginalizing over I_k; x-tuning; I-nontuning
    log_p_tuning_prev_nontuning_curr_prior = logsumexp(log_p_tuning_prev_nontuning_curr_prior_,axis=0) # sum over source for nontuning

    log_prior_curr_= log_p_tuning_prev_nontuning_curr_prior[:,:,None] + log_dynamics_transition_kernel # n_dynamics x n_latent x n_dynamics
    log_prior_curr = logsumexp(log_prior_curr_,axis=1) # sum over source for tuning

    log_post_curr_ = log_prior_curr  + likelihood_scale * ll_curr[None,:] # n_dynamics x n_latent
    log_marginal_ratio_curr = logsumexp(log_post_curr_) # important! the normalizing factor is not p(s_1:t), but p(s_1:t) / p(s_1:t-1)
    log_post_curr = log_post_curr_ - log_marginal_ratio_curr
    log_marginal_tillcurr =  log_marginal_tillprev + log_marginal_ratio_curr 
    
    carry_next = (log_post_curr,log_marginal_tillcurr)
    return carry_next,(log_post_curr,log_prior_curr) # return carry, y

def filter_all_step(ll_all,log_latent_transition_kernel_l,log_dynamics_transition_kernel,carry_init=None,likelihood_scale=1):
    '''
    run causal filter
    carry_init: posterior init, marginal ll till curr
    '''
    n_pos = log_latent_transition_kernel_l[0].shape[0]
    n_nontuning_state = log_dynamics_transition_kernel.shape[0]
    log_posterior_init = jnp.log(jnp.ones((n_nontuning_state,n_pos))/(n_nontuning_state*n_pos))
    if carry_init is None:
        carry_init = (log_posterior_init,jnp.array(0.))
    f = partial(filter_one_step,log_latent_transition_kernel_l=log_latent_transition_kernel_l,log_dynamics_transition_kernel=log_dynamics_transition_kernel,likelihood_scale=likelihood_scale)
    carry_final, (log_posterior_all,log_prior_curr_all) = scan(f,carry_init,xs=ll_all) 
    log_marginal_final = carry_final[1]
    return log_posterior_all, log_marginal_final, log_prior_curr_all


def filter_all_step_combined_ma(y, tuning, ma,log_latent_transition_kernel_l,log_dynamics_transition_kernel,carry_init=None,likelihood_scale=1):
    '''
    get ll and then filter
    
    '''
    
    ll_all=get_loglikelihood_ma_all(y,tuning,ma)
    log_posterior_all,log_marginal_final,log_prior_curr_all =filter_all_step(ll_all,log_latent_transition_kernel_l,log_dynamics_transition_kernel,carry_init=carry_init,likelihood_scale=likelihood_scale)
    return log_posterior_all,log_marginal_final,log_prior_curr_all