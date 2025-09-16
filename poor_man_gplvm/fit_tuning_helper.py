'''
Helper functions for the M-step in the EM algorithm
'''

import jax
import jax.numpy as jnp
from jax import jit, vmap
import jax.scipy as jscipy

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

def get_statistics(log_posterior_probs,y,):
    '''
    get posterior weighted observation, and posterior weighted time, for each latent bin
    posterior_probs: n_time x n_latent 
    y: n_time x n_neuron
    return:
    y_weighted: n_latent x n_neuron (A matrix)
    t_weighted: n_latent  (B vector)
    '''
    
    posterior_probs = jnp.exp(log_posterior_probs)
    # y_weighted = jnp.einsum('tl,tn->ln',posterior_probs,y)
    y_weighted=posterior_probs.T @ y # n_latent x n_neuron; see if edge case is fixed
    t_weighted = posterior_probs.sum(axis=0) # n_latent,
    return y_weighted, t_weighted

@jit
def gaussian_m_step_analytic(hyperparam,basis_mat,y_weighted,t_weighted):
    '''
    basis_mat: n_latent x n_basis
    y_weighted: n_latent x n_neuron
    t_weighted: n_latent
    
    '''
    n_latent,n_basis = basis_mat.shape
    n_neuron = y_weighted.shape[1]
    noise_var = hyperparam['noise_std']**2
    param_prior_std = hyperparam['param_prior_std']

    G = jnp.einsum('qd,q,qb->db',basis_mat,t_weighted,basis_mat)
    H = G / noise_var + jnp.eye(n_basis) / (param_prior_std**2)    # compute the covariance matrix
    RHS = basis_mat.T @ y_weighted / noise_var
    w = jnp.linalg.solve(H,RHS)
    return w

def poisson_m_step_objective(param,hyperparam,basis_mat,y_weighted,t_weighted):
    '''
    param: n_basis x n_neuron
    basis_mat: n_latent x n_basis
    y_weighted: n_latent x n_neuron
    t_weighted: n_latent
    
    return:
    negative log joint
    '''
    param_prior_std = hyperparam['param_prior_std']
    pf_hat = get_tuning_softplus(param,basis_mat) # n_latent x n_neuron
    # log_likelihood = jax.scipy.stats.poisson.logpmf(y_weighted,yhat * t_weighted[:,None]).sum()

    norm_term = pf_hat * t_weighted[:,None] # n_latent x n_neuron
    fit_term = vmap(jscipy.special.xlogy,in_axes=(1,1),out_axes=1)(y_weighted,pf_hat+1e-20) # n_latent x n_neuron
    log_likelihood = jnp.sum(fit_term - norm_term) # crucial, this is different from poisson logpmf(s_b_one, pf_one*t_b)!!!!
    log_prior = jax.scipy.stats.norm.logpdf(param,0,param_prior_std).sum()
    return -log_likelihood - log_prior

def poisson_m_step_objective_smoothness(param,hyperparam,basis_mat,y_weighted,t_weighted):
    '''
    used for bspline basis smoothness penalty; penalize the squared second derivative of the tuning curve;
    param: n_basis x n_neuron
    basis_mat: n_latent x n_basis
    y_weighted: n_latent x n_neuron
    t_weighted: n_latent
    
    return:
    negative log joint
    '''
    param_prior_std = hyperparam['param_prior_std']
    smoothness_penalty = hyperparam['smoothness_penalty'] # if using bspline basis
    tuning_curves = get_tuning_softplus(param,basis_mat) # n_latent x n_neuron
    
    # Calculate second order finite differences for each neuron's tuning curve
    # Exclude edges since we're not using periodic boundary conditions
    center = tuning_curves[1:-1]  # All but first and last points
    forward1 = tuning_curves[2:]  # From second point to end
    backward1 = tuning_curves[:-2]  # From start to second-to-last point
    
    # Second derivative approximation
    second_diff = forward1 - 2*center + backward1
    
    # Square the differences and sum for each neuron
    roughness = jnp.sum(second_diff**2, axis=0)  # Sum over latent positions for each neuron
    total_roughness = jnp.sum(roughness)  # Sum over neurons
    
    # Add weighted roughness penalty to objective
    roughness_term = smoothness_penalty * total_roughness
    # log_likelihood = jax.scipy.stats.poisson.logpmf(y_weighted,yhat * t_weighted[:,None]).sum()

    norm_term = tuning_curves * t_weighted[:,None] # n_latent x n_neuron
    fit_term = vmap(jscipy.special.xlogy,in_axes=(1,1),out_axes=1)(y_weighted,tuning_curves+1e-20) # n_latent x n_neuron
    log_likelihood = jnp.sum(fit_term - norm_term) # crucial, this is different from poisson logpmf(s_b_one, pf_one*t_b)!!!!
    log_prior = jax.scipy.stats.norm.logpdf(param,0,param_prior_std).sum()
    return -log_likelihood - log_prior + roughness_term

import optax
from jax import tree_util

def make_adam_runner(fun, step_size, maxiter=1000, tol=1e-6):
    '''
    make a function that run adam optimizer with a given objective function
    
    '''
    
    # fun(params, *args) -> loss scalar
    opt = optax.adam(step_size)
    init_fun = opt.init
    @jax.jit
    def run(init_params, opt_state, *args):
        # Always receives a valid optimizer state (initialization handled outside)
    
        params = init_params
        
        # compute initial error (e.g. gradient norm)
        loss, grads = jax.value_and_grad(fun)(params, *args)
        error = tree_l2_norm(grads)  # Keep for monitoring

        # Pre-allocate history arrays (JIT-compatible)
        loss_history = jnp.zeros(maxiter)
        error_history = jnp.zeros(maxiter)
        
        # Set initial values
        loss_history = loss_history.at[0].set(loss)
        error_history = error_history.at[0].set(error)

        # carry: (iter, params, opt_state, error, loss, loss_prev, loss_history, error_history)
        carry = (0, params, opt_state, error, loss, loss, loss_history, error_history)  # loss_prev = loss initially

        def cond_fun(carry):
            i, params, opt_state, error, loss, loss_prev, loss_history, error_history = carry
            # Continue if: haven't hit maxiter AND (still in warmup OR loss is still changing significantly)
            min_iters = 5  # Run at least 5 iterations before checking convergence
            relative_loss_change = jnp.abs(loss - loss_prev) / jnp.maximum(jnp.abs(loss), 1e-8)
            
            in_warmup = i < min_iters
            not_converged = relative_loss_change > tol
            not_maxed_out = i < (maxiter - 1)
            
            return not_maxed_out & (in_warmup | not_converged)

        def body_fun(carry):
            i, params, opt_state, error, loss, loss_prev, loss_history, error_history = carry
            new_loss, grads = jax.value_and_grad(fun)(params, *args)
            updates, new_opt_state = opt.update(grads, opt_state, params)
            new_params = optax.apply_updates(params, updates)
            new_error = tree_l2_norm(grads)
            
            # Update histories
            new_i = i + 1
            new_loss_history = loss_history.at[new_i].set(new_loss)
            new_error_history = error_history.at[new_i].set(new_error)
            
            # Pass current loss as previous loss for next iteration
            return (new_i, new_params, new_opt_state, new_error, new_loss, loss, new_loss_history, new_error_history)

        # run the loop
        i, params, opt_state, error, loss, loss_prev, loss_history, error_history = jax.lax.while_loop(cond_fun, body_fun, carry)
        
        # Return full arrays with actual length - trimming handled outside JIT for shape stability
        n_actual_iter = i + 1
        
        adam_res = {'params': params, 
                   'opt_state': opt_state,         # Return updated optimizer state
                   'n_iter': n_actual_iter, 
                   'final_loss': loss, 
                   'final_error': error,
                   'loss_history': loss_history,      # Full array (maxiter length)
                   'error_history': error_history}    # Full array (maxiter length)
        return adam_res

    return run, init_fun


def tree_l2_norm(tree_x, squared=False):
    # Square each leaf 
    squared_tree = tree_util.tree_map(lambda leaf: jnp.sum(jnp.square(leaf)), tree_x)
    # Sum up all squares across the pytree
    sqnorm = tree_util.tree_reduce(jnp.add, squared_tree)
    # Return either the squared norm or its square root
    return sqnorm if squared else jnp.sqrt(sqnorm)
