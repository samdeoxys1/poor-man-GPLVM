"""Core implementation of the Poor Man's GPLVM."""

import numpy as np
from jax import jit, vmap

import poor_man_gplvm.gp_kernel as gpk
# from .gp_kernel import rbf_kernel,uniform_kernel
import jax.numpy as jnp
import jax
import jax.random as jr
from jax.scipy.special import logsumexp
from poor_man_gplvm import fit_tuning_with_basis as ftwb
from abc import ABC, abstractmethod


'''
hyperparams = {'tuning_lengthscale':,'movement_variance':,'prior_variance':}
to model non jump can use the transition matrix
at each EM iteration, create the transition matrix based on the hyperparams;
fix lenscale, so eigenvalue and eigenvectors are fixed; but allow a latent mask in decoder such that i can do downsampled test lml for model selection;
'''

def generate_basis(lengthscale,n_latent_bin,explained_variance_threshold_basis = 0.999,include_bias=True ):
    possible_latent_bin = jnp.linspace(0,1,n_latent_bin)
    tuning_kernel,log_tuning_kernel = vmap(vmap(lambda x,y: gpk.rbf_kernel(x,y,lengthscale,1.),in_axes=(0,None),out_axes=0),out_axes=1,in_axes=(None,0))(possible_latent_bin,possible_latent_bin)

    tuning_basis,sing_val,_ = jnp.linalg.svd(tuning_kernel)
    # filter out basis for numerical instability
    n_basis = (jnp.cumsum(sing_val / sing_val.sum()) < explained_variance_threshold_basis).sum() + 1 # first dimension that cross the thresh, n below + 1
    sqrt_eigval=jnp.sqrt(jnp.sqrt(sing_val))
    tuning_basis = tuning_basis[:,:n_basis] * sqrt_eigval[:n_basis][None,:] 
    if include_bias:
        tuning_basis = jnp.concatenate([jnp.ones((n_latent_bin,1)),tuning_basis],axis=1)
    return tuning_basis


class AbstractGPLVMJump1D(ABC):
    """GPLVM with smooth 1d latent + jumps.
    The latent governs firing rate; the dynamics governs the transition probabilities between the latent states;
    """
    
    def __init__(self,n_neuron, n_latent_bin = 100, tuning_lengthscale=1.,
                 movement_variance=1.,
                 explained_variance_threshold_basis=0.999,
                 rng_init_int = 123,
                 w_init_variance=1.,
                 w_init_mean=0.,
                 p_move_to_jump=0.01,
                 p_jump_to_move=0.01,
                 ):
        self.n_latent_bin = n_latent_bin
        self.tuning_lengthscale = tuning_lengthscale
        self.movement_variance = movement_variance
        self.p_move_to_jump = p_move_to_jump
        self.p_jump_to_move = p_jump_to_move
        self.explained_variance_threshold_basis = explained_variance_threshold_basis
        self.rng_init_int = rng_init_int
        self.rng_init = jr.PRNGKey(rng_init_int)
        self.n_neuron = n_neuron
        self.possible_latent_bin = jnp.arange(self.n_latent_bin)
        self.possible_dynamics = jnp.arange(2)
        self.w_init_variance = w_init_variance
        self.w_init_mean = w_init_mean
        # self.b_init_variance = b_init_variance
        # self.b_init_mean = b_init_mean

        # generate the basis
        self.tuning_basis = generate_basis(self.tuning_lengthscale,self.n_latent_bin,self.explained_variance_threshold_basis,include_bias=True)
        self.n_basis = self.tuning_basis.shape[1]
       
        # default masks
        self.ma_neuron_default = jnp.ones(self.n_neuron)
        self.ma_latent_default = jnp.ones(self.n_latent_bin)

        # initialize the params and tuning
        self.initialize_params(self.rng_init)
    
    @abstractmethod
    def get_tuning(self,params,hyperparam):
        '''
        hyperparam currently not used; for potential alternative parameterizations
        '''
        pass
    
    def initialize_params(self,key):
        params_init_w = jax.random.normal(key,(self.n_basis,self.n_neuron)) * jnp.sqrt(self.w_init_variance) # prior_hyper here is variance
        # params_init_b = jax.random.normal(key,(self.n_neuron,)) * jnp.sqrt(self.b_init_variance) + self.b_init_mean
        # params_init = (params_init_w,params_init_b)
        params_init = params_init_w
        tuning_init = self.get_tuning(params_init,hyperparam={})
        self.params = params_init
        self.tuning = tuning_init
        return params_init,tuning_init

    def _decode_latent(self,y,log_latent_transition_kernel_l,log_dynamics_transition_kernel,likelihood_scale=1.):
        '''
        decode the latent and dynamics
        y: observed data, spike counts here; n_time x n_neuron

        log_latent_transition_kernel_l: n_dynamics x n_latent x n_latent
        log_dynamics_transition_kernel: n_dynamics x n_dynamics
        '''
        pass


    def sample_latent(self,T,key=jax.random.PRNGKey(0),movement_variance=1,p_move_to_jump=0.01,p_jump_to_move=0.01,
                      init_dynamics=None,init_latent=None):
        
        latent_transition_kernel_l,log_latent_transition_kernel_l,dynamics_transition_kernel,log_dynamics_transition_kernel = gpk.create_transition_prob_1d(self.possible_latent_bin,self.possible_dynamics,movement_variance,p_move_to_jump,p_jump_to_move)

        if init_dynamics is None:
            init_dynamics = jax.random.choice(key,self.possible_dynamics)
        if init_latent is None:
            init_latent = jax.random.choice(key,self.possible_latent_bin)
        key_l = jax.random.split(key,T)
        dynamics_prev = init_dynamics
        latent_prev = init_latent

        # nontuning_state_l = [nontuning_state_prev]
        # tuning_state_l = [tuning_state_prev]
        carry_init = (dynamics_prev, latent_prev)

        @jit
        def step(carry, key):
            k1,k2=jax.random.split(key,2)
            dynamics_prev,latent_prev = carry 
            dynamics_curr = jax.random.choice(k1,self.possible_dynamics, p=dynamics_transition_kernel[dynamics_prev])
            latent_curr = jax.random.choice(k2, self.possible_latent_bin,p=latent_transition_kernel_l[dynamics_curr][latent_prev])
            carry = dynamics_curr,latent_curr
            state = jnp.array([dynamics_curr,latent_curr])
            return carry, state

        _,latent_l=jax.lax.scan(step,carry_init,xs=key_l)

        return latent_l

    
    def sample(self,T,hyperparam={},key=jax.random.PRNGKey(0),
                      init_dynamics=None,init_latent=None,dt=1.,tuning=None):
        '''
        sample both latent and y
        '''
        key_l = jax.random.split(key,T)
        movement_variance = hyperparam.get('movement_variance',self.movement_variance)
        p_move_to_jump = hyperparam.get('p_move_to_jump',self.p_move_to_jump)
        p_jump_to_move = hyperparam.get('p_jump_to_move',self.p_jump_to_move)
        latent_l = self.sample_latent(T,key_l[0],movement_variance,p_move_to_jump,p_jump_to_move,init_dynamics,init_latent)
        y_l = self.sample_y(latent_l[:,1],hyperparam,tuning,dt,key_l[1]) # only using the latent and not the dynamics
        return latent_l,y_l

    


class PoissonGPLVMJump1D(AbstractGPLVMJump1D):
    """Poisson GPLVM with jumps.
    The latent governs firing rate; the dynamics governs the transition probabilities between the latent states;
    """
    
    def loglikelihood(self,y,ypred,hyperparam):
        return jax.scipy.stats.poisson.logpmf(y,ypred+1e-40)

    def get_tuning(self,params,hyperparam):
        tuning = ftwb.get_tuning_softplus(params,self.tuning_basis)
        return tuning
        

    def _decode_latent(self,y,log_latent_transition_kernel_l,log_dynamics_transition_kernel,likelihood_scale=1.):
        '''
        decode the latent and dynamics
        y: observed data, spike counts here; n_time x n_neuron

        log_latent_transition_kernel_l: n_dynamics x n_latent x n_latent
        log_dynamics_transition_kernel: n_dynamics x n_dynamics
        '''
        pass


    def sample_y(self,latent_l,hyperparam={},tuning=None,dt=1.,key=jax.random.PRNGKey(10)):
        if tuning is None:
            tuning = self.tuning
        rate = tuning[latent_l,:]
        
        spk_sim=jax.random.poisson(key,rate * dt)
        return spk_sim
    

class GaussianGPLVMJump1D(AbstractGPLVMJump1D):
    """Gaussian GPLVM with jumps.
    The latent governs firing rate; the dynamics governs the transition probabilities between the latent states;
    """
    def __init__(self,n_neuron,noise_std=0.5,**kwargs):
        super(GaussianGPLVMJump1D,self).__init__(n_neuron,**kwargs)
        self.noise_std = noise_std
        
    def loglikelihood(self,y,ypred,hyperparam):
        return jax.scipy.stats.norm.logpdf(y,ypred,hyperparam['noise_std'])
    
    def get_tuning(self,params,hyperparam):
        tuning = ftwb.get_tuning_linear(params,self.tuning_basis)
        return tuning
        

    def _decode_latent(self,y,log_latent_transition_kernel_l,log_dynamics_transition_kernel,likelihood_scale=1.):
        '''
        decode the latent and dynamics
        y: observed data, spike counts here; n_time x n_neuron

        log_latent_transition_kernel_l: n_dynamics x n_latent x n_latent
        log_dynamics_transition_kernel: n_dynamics x n_dynamics
        '''
        pass


    def sample_y(self,latent_l,hyperparam={},tuning=None,dt=1.,key=jax.random.PRNGKey(10)):
        if tuning is None:
            tuning = self.tuning
        noise_std = hyperparam.get('noise_std',self.noise_std)
        rate = tuning[latent_l,:] * dt
        noise_std = noise_std * jnp.sqrt(dt)
        spk_sim=jax.random.normal(key,shape=rate.shape) * noise_std + rate
        return spk_sim
    