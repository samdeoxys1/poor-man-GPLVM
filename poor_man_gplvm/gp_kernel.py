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


def create_transition_prob_from_transmat_list(possible_latent_bin, transmat_list, p_stay):
    '''
    Build transition kernels from a list of latent transition matrices + uniform fragmented.

    Difference from create_transition_prob_1d:
    - create_transition_prob_1d: uses p_move_to_jump / p_jump_to_move for 2-dynamics (continuous + jump).
    - This function: uses single p_stay; equal probability to all other dynamics. Use for multi-dynamics
      (several latent transition matrices + one uniform "fragmented" dynamics).

    Parameters
    ----------
    possible_latent_bin : array
        Same as in create_transition_prob_1d.
    transmat_list : list of (L,L) arrays or (K,L,L) array
        Latent transition matrix per "behavior" dynamics. Rows normalized inside.
    p_stay : float in (0,1)
        Probability to stay in current dynamics; (1-p_stay) split equally to other dynamics.

    Returns
    -------
    Same 4-tuple as create_transition_prob_1d: latent_transition_kernel_l, log_latent_transition_kernel_l,
    dynamics_transition_kernel, log_dynamics_transition_kernel. Shapes: (K+1,L,L), (K+1,L,L), (K+1,K+1), (K+1,K+1)
    with K = len(transmat_list), L = n_latent. Last dynamics index = fragmented (uniform).

    When using multi-dynamics (custom_kernel as list), do not: rely on index 0 = continuous or -1 = fragmented;
    use clean_jump.compute_clean_transition_and_decode; use select_inverse_temperature_match_step_continuous
    without redefining which index is continuous; or use helpers that assume posterior_dynamics_marg has
    exactly 2 columns or column 0 = P(continuous).
    '''
    n_latent = len(possible_latent_bin)
    if isinstance(transmat_list, (list, tuple)):
        transmat_stack = jnp.stack([jnp.asarray(t) for t in transmat_list])
    else:
        transmat_stack = jnp.asarray(transmat_list)
    K = transmat_stack.shape[0]
    latent_kernels = []
    for i in range(K):
        m = transmat_stack[i]
        normalizer = m.sum(axis=1, keepdims=True) + 1e-30
        m_norm = m / normalizer
        log_m = jnp.log(m_norm + 1e-30)
        latent_kernels.append((m_norm, log_m))
    uniform_k = jnp.ones((n_latent, n_latent)) / n_latent
    log_uniform = jnp.log(uniform_k)
    latent_transition_kernel_l = jnp.stack([lk[0] for lk in latent_kernels] + [uniform_k])
    log_latent_transition_kernel_l = jnp.stack([lk[1] for lk in latent_kernels] + [log_uniform])
    n_dyn = K + 1
    off_diag = (1.0 - p_stay) / K
    dynamics_transition_matrix = jnp.eye(n_dyn) * p_stay + (1.0 - jnp.eye(n_dyn)) * off_diag
    possible_dynamics = jnp.arange(n_dyn)
    dynamics_transition_kernel, log_dynamics_transition_kernel = vmap(
        vmap(lambda x, y: discrete_transition_kernel(x, y, dynamics_transition_matrix),
             in_axes=(0, None), out_axes=0),
        in_axes=(None, 0), out_axes=1
    )(possible_dynamics, possible_dynamics)
    return latent_transition_kernel_l, log_latent_transition_kernel_l, dynamics_transition_kernel, log_dynamics_transition_kernel


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


def get_custom_kernel_rbf_plus_isolated(possible_latent_bin,tuning_lengthscale,transition_lengthscale,var=1,p_to_isolated=0.001):
    '''
    get custom kernel for tuning and transition:
        rbf kernel plus one isolated latent
    for tuning, the isolated latent has no smoothness with other latents; magnitude set to var
    for transition, the isolated latent has equal transition probability to all other latents; all others have a set probability to the isolated
    '''
    n_latent_bin = len(possible_latent_bin)
    kernel_mat, log_kernel_mat = vmap(
        vmap(lambda x,y: rbf_kernel(x,y,tuning_lengthscale,var), 
             in_axes=(0,None), out_axes=0),
        out_axes=1, in_axes=(None,0)
        )(possible_latent_bin, possible_latent_bin)
    # for tuning, the isolated latent has no smoothness
    tuning_kernel = kernel_mat.at[0].set(jnp.zeros(n_latent_bin))
    tuning_kernel=tuning_kernel.at[:,0].set(jnp.zeros(n_latent_bin))
    tuning_kernel = tuning_kernel.at[0,0].set(var)
    # for transition, the isolated latent has equal transition probability to all other latents
    transition_kernel, log_transition_kernel = vmap(
        vmap(lambda x,y: rbf_kernel(x,y,transition_lengthscale,var), 
             in_axes=(0,None), out_axes=0),
        out_axes=1, in_axes=(None,0)
        )(possible_latent_bin, possible_latent_bin)
    transition_kernel = transition_kernel.at[0].set(jnp.ones(n_latent_bin)) * (1/n_latent_bin) # p from isolated, uniform
    transition_kernel = transition_kernel.at[1:,0].set(jnp.ones(n_latent_bin-1) * p_to_isolated) # p to isolated, set
    the_rest_normalized = (transition_kernel[1:,1:] / transition_kernel[1:,1:].sum(axis=1,keepdims=True)) * (1-p_to_isolated)
    transition_kernel = transition_kernel.at[1:,1:].set(the_rest_normalized)
    return tuning_kernel, transition_kernel
