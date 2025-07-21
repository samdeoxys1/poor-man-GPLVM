'''
Fit the tuning to latent using basis functions (vectors).
'''

import jax
import jax.numpy as jnp
from jax import jit, vmap
import jaxopt
from functools import partial
import jax.scipy.special as jscipy

# old need to get rid; 
@jit
def glm_get_tuning(params,basis):
    '''
    params: n_feat (basis) x n_neuron
    basis: n_tuning_state x n_basis
    '''
    params_w,params_b = params
    tuning = jax.nn.softplus(basis.dot(params_w) + params_b)
    
    return tuning


@jit
def gaussian_logprior(params,var):
    return jnp.sum(-jnp.sum(params**2,axis=0) / (2*var) )

@jit
def get_log_prior_params(params_one,prior_hyper):
    var = prior_hyper
    params_w = params_one[0]
    return gaussian_logprior(params_w,var)

m_step_get_tuning_all_neuron_grouped_makefun = lambda maxiter,stepsize: jit(partial(m_step_get_tuning_all_neuron_grouped,maxiter=maxiter,stepsize=stepsize))

def m_step_get_tuning_all_neuron_grouped(params_init,spk,tuning_basis,posterior_marg,prior_hyper,
                                                    maxiter=500,stepsize=0.001,n_time_per_chunk=50000,n_neuron_per_chunk=100,dt=1):
    
    s_b,t_b = group_spk_occupancy_chunk_neuron(spk,posterior_marg,n_neuron_per_chunk=n_neuron_per_chunk,dt=dt)
    
    
    # opt=jaxopt.BFGS(get_neg_log_poisson_p_y_joint_params_oneneuron_grouped,stepsize=stepsize,maxiter=maxiter)
    opt=jaxopt.LBFGS(get_neg_log_poisson_p_y_joint_params_oneneuron_grouped,stepsize=stepsize,maxiter=maxiter)
    runner_vmap=vmap(opt.run,in_axes=(-1,-1,None,None,None),out_axes=-1)

    opt_res_vmap=runner_vmap(params_init,s_b,tuning_basis,t_b,prior_hyper)
    # pdb.set_trace()
    final_err = opt_res_vmap.state.value.sum()
    params_fit = opt_res_vmap.params
    tuning_fit=glm_get_tuning(params_fit,tuning_basis)
    return params_fit,tuning_fit,final_err

@jit
def get_s_b(spk_chunk,post_x_l):
    s_b = (spk_chunk[...,None] * post_x_l[:,None,:]).sum(axis=0).T #  n_pos x n_neuron; weighted spike per tuning state
    return s_b

def group_spk_occupancy_chunk_neuron(spk,post_x_l,n_neuron_per_chunk=2,dt=1.):
    n_time,n_neuron = spk.shape
    n_chunks = int(jnp.ceil(n_neuron / n_neuron_per_chunk))
    dt_l = jnp.broadcast_to(dt,(n_time,))

    # t_b = post_x_l.sum(axis=0) # n_pos
    
    t_b= (post_x_l * dt_l[:,None]).sum(axis=0) # if different chunks have different dt

    s_b_l = []
    for n in range(n_chunks):
        sl = slice((n) * n_neuron_per_chunk , (n+1) * n_neuron_per_chunk)
        spk_chunk = spk[:,sl]
        # dt_l_chunk = dt_l[:,sl]
        s_b=get_s_b(spk_chunk,post_x_l)
        s_b_l.append(s_b)
    s_b = jnp.concatenate(s_b_l,axis=1)
    return s_b,t_b


def get_log_poisson_p_y_given_params_oneneuron_grouped(params_one,s_b_one,basis,t_b):
    '''
    given the accumulated spike and occupancy, get log likelihood of y|params for one neuron
    grouped means grouped by the tuning state
    s_b_one: n_time
    '''
    
    pf_one = glm_get_tuning(params_one,basis) # n_pos
    l_p_y_given_f = jnp.sum(jscipy.special.xlogy(s_b_one,pf_one+1e-20) - pf_one * t_b) # crucial, this is different from poisson logpmf(s_b_one, pf_one*t_b)!!!!
    return l_p_y_given_f

def get_log_poisson_p_y_joint_params_oneneuron_grouped(params_one,s_b_one,basis,t_b,prior_hyper):    
    
    l_p_y_given_f = get_log_poisson_p_y_given_params_oneneuron_grouped(params_one,s_b_one,basis,t_b)
    l_p_params = get_log_prior_params(params_one,prior_hyper)
    l_p_joint = l_p_y_given_f + l_p_params
    l_p_joint = l_p_joint / s_b_one.shape[0] # normalize by time
    return l_p_joint

get_neg_log_poisson_p_y_joint_params_oneneuron_grouped = jit(lambda *args:-get_log_poisson_p_y_joint_params_oneneuron_grouped(*args))

def m_step_get_tuning_all_neuron_grouped(params_init,spk,tuning_basis,posterior_marg,prior_hyper,
                                                    maxiter=500,stepsize=0.001,n_time_per_chunk=50000,n_neuron_per_chunk=100,dt=1):
    
    s_b,t_b = group_spk_occupancy_chunk_neuron(spk,posterior_marg,n_neuron_per_chunk=n_neuron_per_chunk,dt=dt)
    
    

    opt=jaxopt.LBFGS(get_neg_log_poisson_p_y_joint_params_oneneuron_grouped,stepsize=stepsize,maxiter=maxiter)
    runner_vmap=vmap(opt.run,in_axes=(-1,-1,None,None,None),out_axes=-1)

    opt_res_vmap=runner_vmap(params_init,s_b,tuning_basis,t_b,prior_hyper)
    
    final_err = opt_res_vmap.state.value.sum()
    params_fit = opt_res_vmap.params
    tuning_fit=glm_get_tuning(params_fit,tuning_basis)
    return params_fit,tuning_fit,final_err