import jax
import jax.numpy as jnp
from jax import jit, vmap
import jaxopt
from functools import partial


@jit
def get_log(val):
    log_val = jnp.log(val)
    log_val=jax.lax.cond(log_val==jnp.inf,lambda x:-10000.,lambda x:x,log_val)
    return log_val

@jit
def rbf_kernel(x,y,ls,var):
    dist_sq = jnp.linalg.norm(x-y)**2
    val = jnp.exp(-dist_sq/ls**2) * var
    # log_val = get_log(val)
    log_val = -dist_sq/ls**2 + jnp.log(var)
    return val,log_val

def rbf_kernel_multi_d(x,y,ls,var):
    dist_sq_per_dim = (x-y)**2
    val = jnp.exp(-jnp.sum(dist_sq_per_dim/ls**2)) * var
    # log_val = get_log(val)
    log_val = -jnp.sum(dist_sq_per_dim/ls**2) +jnp.log(var)
    return val,log_val


@jit
def discrete_transition_kernel(x,y,trans_mat):
    val= trans_mat[x,y]
    log_val = get_log(val)
    return val,log_val

@jit
def uniform_kernel(x,y,n_tuning_state):
    val = 1/n_tuning_state
    log_val = get_log(val)
    return val,log_val

@jit
def create_transition_prob_1d(possible_latent_bin,possible_dynamics,movement_variance=1,p_move_to_jump=0.01,p_jump_to_move=0.01,custom_kernel=None):
    '''
    create the transition probability matrix for 1d latent and dynamics;
    this is done at the beginning of the fit; so the hyperparams can be selected easily

    log_latent_transition_kernel_l: n_dynamics x n_latent x n_latent
    log_dynamics_transition_kernel: n_dynamics x n_dynamics
    '''
    # multiple tuning state transition
    latent_transition_kernel_l = []
    log_latent_transition_kernel_l=[]
    n_latent_bin = len(possible_latent_bin)
    if custom_kernel is None:
        latent_transition_kernel_func_l = [rbf_kernel,uniform_kernel]
        latent_transition_kernel_args_l = [
            [movement_variance,1.],
            [n_latent_bin]
    ]
    else:
        latent_transition_kernel_func_l = [discrete_transition_kernel,uniform_kernel]
        latent_transition_kernel_args_l = [
            [custom_kernel],
            [n_latent_bin]
        ]
    
    
    
    dynamics_transition_kernel_func = discrete_transition_kernel
    
    for latent_transition_kernel_func,latent_transition_kernel_args in zip(latent_transition_kernel_func_l,
                                                                                        latent_transition_kernel_args_l
                                                                                        ):
        
        latent_transition_kernel,log_latent_transition_kernel = vmap(vmap(lambda x,y: latent_transition_kernel_func(x,y,*latent_transition_kernel_args),in_axes=(0,None),out_axes=0),out_axes=1,in_axes=(None,0))(possible_latent_bin,possible_latent_bin)
        normalizer = latent_transition_kernel.sum(axis=1,keepdims=True)
        latent_transition_kernel = latent_transition_kernel / normalizer # transition kernel need to be normalized
        log_latent_transition_kernel = log_latent_transition_kernel - jnp.log(normalizer)
        latent_transition_kernel_l.append(latent_transition_kernel)
        log_latent_transition_kernel_l.append(log_latent_transition_kernel)
    latent_transition_kernel_l = jnp.array(latent_transition_kernel_l) # n_non_tuning_state x n_tuning_state x n_tuning_state
    log_latent_transition_kernel_l = jnp.array(log_latent_transition_kernel_l)

    # classifier transition
    dynamics_transition_matrix = jnp.array([[1-p_move_to_jump,p_move_to_jump],[p_jump_to_move,1-p_jump_to_move]])
    dynamics_transition_kernel,log_dynamics_transition_kernel = vmap(vmap(lambda x,y:dynamics_transition_kernel_func(x,y,dynamics_transition_matrix),in_axes=(0,None),out_axes=0),in_axes=(None,0),out_axes=1)(possible_dynamics,possible_dynamics) 

    return latent_transition_kernel_l,log_latent_transition_kernel_l,dynamics_transition_kernel,log_dynamics_transition_kernel

@jit
def create_transition_prob_latent_1d(possible_latent_bin, movement_variance=1.,custom_kernel=None):
    '''
    create the transition probability matrix for 1d latent only (no dynamics);
    this is simplified version of create_transition_prob_1d for latent-only models
    
    Returns:
    latent_transition_kernel: n_latent x n_latent
    log_latent_transition_kernel: n_latent x n_latent  
    '''
    # Use RBF kernel for smooth transitions
    if custom_kernel is None:
        latent_transition_kernel, log_latent_transition_kernel = vmap(
        vmap(lambda x,y: rbf_kernel(x, y, movement_variance, 1.), 
             in_axes=(0,None), out_axes=0),
        out_axes=1, in_axes=(None,0)
        )(possible_latent_bin, possible_latent_bin)
    else:
        latent_transition_kernel, log_latent_transition_kernel = vmap(
        vmap(lambda x,y: discrete_transition_kernel(x,y,custom_kernel), 
             in_axes=(0,None), out_axes=0),
        out_axes=1, in_axes=(None,0)
        )(possible_latent_bin, possible_latent_bin)
    # Normalize to make it a proper transition matrix
    normalizer = latent_transition_kernel.sum(axis=1, keepdims=True)
    latent_transition_kernel = latent_transition_kernel / normalizer
    log_latent_transition_kernel = log_latent_transition_kernel - jnp.log(normalizer)
    
    return latent_transition_kernel, log_latent_transition_kernel


def get_custom_kernel_rbf_plus_isolated(possible_latent_bin,lengthscale,var=1):
    '''
    get custom kernel for tuning and transition:
        rbf kernel plus one isolated latent
    for tuning, the isolated latent has 
    for transition, the isolated latent has equal transition probability to all other latents
    '''
    n_latent_bin = len(possible_latent_bin)
    rbf_kernel, log_rbf_kernel = vmap(
        vmap(lambda x,y: rbf_kernel(x,y,lengthscale,var), 
             in_axes=(0,None), out_axes=0),
        out_axes=1, in_axes=(None,0)
        )(possible_latent_bin, possible_latent_bin)
    # for tuning, the isolated latent has no smoothness
    tuning_kernel = rbf_kernel.at[0].set(jnp.zeros(n_latent_bin))
    tuning_kernel=tuning_kernel.at[:,0].set(jnp.zeros(n_latent_bin))
    tuning_kernel = tuning_kernel.at[0,0].set(var)
    # for transition, the isolated latent has equal transition probability to all other latents
    transition_kernel = rbf_kernel.at[0].set(jnp.ones(n_latent_bin)) * (1/n_latent_bin)
    transition_kernel = transition_kernel.at[:,0].set(jnp.ones(n_latent_bin)) * (1/n_latent_bin)
    
    return tuning_kernel, transition_kernel
