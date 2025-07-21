'''
Helper functions for the M-step in the EM algorithm
'''

import jax
import jax.numpy as jnp
from jax import jit, vmap


# new==
@jit
def get_tuning_linear(params,basis):
    '''
    params: n_feat (basis) x n_neuron
    basis: n_tuning_state x n_basis
    '''
    return basis.dot(params)

@jit
def get_tuning_softplus(params,basis):
    '''
    params: n_feat (basis) x n_neuron
    basis: n_tuning_state x n_basis
    '''
    return jax.nn.softplus(get_tuning_linear(params,basis))
#==


@jit
def gaussian_m_step_analytic(basis_mat,y_weighted,t_weighted,noise_stddev):
    '''
    basis_mat: n_latent x n_basis
    y_weighted: n_latent x n_neuron
    t_weighted: n_latent
    
    '''
    n_latent,n_basis = basis_mat.shape
    n_neuron = y_weighted.shape[1]
    noise_var = noise_stddev**2

    G = jnp.einsum('qd,q,qb->db',basis_mat,t_weighted,basis_mat)
    H = G / noise_var + jnp.eye(n_basis)    # compute the covariance matrix
    RHS = basis_mat.T @ y_weighted / noise_var
    w = jnp.linalg.solve(H,RHS)
    return w