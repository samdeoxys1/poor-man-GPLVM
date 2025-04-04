"""Core implementation of the Poor Man's GPLVM."""

import numpy as np
from jax import jit, vmap

from .gp_kernel import rbf_kernel,uniform_kernel
import jax.numpy as jnp
import jax
import jax.random as jr
from jax.scipy.special import logsumexp



'''
hyperparams = {'tuning_lengthscale':,'movement_variance':,'prior_variance':}
'''

def generate_basis(lengthscale,n_latent_bin,explained_variance_threshold_basis = 0.999 ):
    possible_latent_bin = jnp.linspace(0,1,n_latent_bin)
    tuning_kernel,log_tuning_kernel = vmap(vmap(lambda x,y: rbf_kernel(x,y,lengthscale,1.),in_axes=(0,None),out_axes=0),out_axes=1,in_axes=(None,0))(possible_latent_bin,possible_latent_bin)

    tuning_basis,sing_val,_ = jnp.linalg.svd(tuning_kernel)
    # filter out basis for numerical instability
    n_basis = (jnp.cumsum(sing_val / sing_val.sum()) < explained_variance_threshold_basis).sum() + 1 # first dimension that cross the thresh, n below + 1
    sqrt_eigval=jnp.sqrt(jnp.sqrt(sing_val))
    tuning_basis = tuning_basis[:,:n_basis] * sqrt_eigval[:n_basis][None,:] 
    return tuning_basis

class PoissonGPLVMJump1D:
    """Poisson GPLVM with jumps.
    The latent governs firing rate; the dynamics governs the transition probabilities between the latent states;
    """
    
    def __init__(self, n_latent_bin = 100, tuning_lengthscale=1.,
                 movement_variance=1.,
                 explained_variance_threshold_basis=0.999,
                 rng_init_int = 123,
                 w_init_variance=1.,
                 w_init_mean=0.,
                 b_init_variance=1.,
                 b_init_mean=0.,
                 ):
        self.n_latent_bin = n_latent_bin
        self.tuning_lengthscale = tuning_lengthscale
        self.movement_variance = movement_variance
        self.explained_variance_threshold_basis = explained_variance_threshold_basis
        self.rng_init_int = rng_init_int
        self.rng_init = jr.PRNGKey(rng_init_int)

        self.possible_latent_bin = jnp.arange(self.n_latent_bin)
        self.possible_dynamics = jnp.arange(2)
        
        # generate the basis
        self.tuning_basis = generate_basis(self.tuning_lengthscale,self.n_latent_bin,self.explained_variance_threshold_basis)
        self.n_basis = self.tuning_basis.shape[1]

    def create_transition_prob(self,movement_variance=1,p_move_to_jump=0.01,p_jump_to_move=0.01):
        '''
        create the transition probability matrix for the latent and dynamics;
        this is done at the beginning of the fit; so the hyperparams can be selected easily
        '''
        # multiple tuning state transition
        latent_transition_kernel_l = []
        log_latent_transition_kernel_l=[]
        latent_transition_kernel_func_l = [rbf_kernel,uniform_kernel]
        latent_transition_kernel_args_l = [
            [movement_variance,1.],
            [self.n_latent_bin]
        ]
        dynamics_transition_kernel_func = gpk.
        
        for latent_transition_kernel_func,latent_transition_kernel_args in zip(latent_transition_kernel_func_l,
                                                                                           latent_transition_kernel_args_l
                                                                                          ):
            
            latent_transition_kernel,log_latent_transition_kernel = vmap(vmap(lambda x,y: latent_transition_kernel_func(x,y,*latent_transition_kernel_args),in_axes=(0,None),out_axes=0),out_axes=1,in_axes=(None,0))(self.possible_latent_bin,self.possible_latent_bin)
            normalizer = latent_transition_kernel.sum(axis=1,keepdims=True)
            latent_transition_kernel = latent_transition_kernel / normalizer # transition kernel need to be normalized
            log_latent_transition_kernel = log_latent_transition_kernel - jnp.log(normalizer)
            latent_transition_kernel_l.append(latent_transition_kernel)
            log_latent_transition_kernel_l.append(log_latent_transition_kernel)
        latent_transition_kernel_l = jnp.array(latent_transition_kernel_l) # n_non_tuning_state x n_tuning_state x n_tuning_state
        log_latent_transition_kernel_l = jnp.array(log_latent_transition_kernel_l)

        # classifier transition
        dynamics_transition_matrix = jnp.array([[1-p_move_to_jump,p_move_to_jump],[p_jump_to_move,1-p_jump_to_move]])
        dynamics_transition_kernel,log_dynamics_transition_kernel = vmap(vmap(lambda x,y:self.dynamics_transition_kernel_func(x,y,dynamics_transition_matrix),in_axes=(0,None),out_axes=0),in_axes=(None,0),out_axes=1)(self.possible_dynamics,self.possible_dynamics) 

        return latent_transition_kernel_l,log_latent_transition_kernel_l,dynamics_transition_kernel,log_dynamics_transition_kernel

    



    