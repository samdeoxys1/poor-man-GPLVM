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