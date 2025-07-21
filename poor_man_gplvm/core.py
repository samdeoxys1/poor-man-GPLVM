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
from poor_man_gplvm import fit_tuning_helper as fth
from poor_man_gplvm import decoder
import tqdm

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
    def get_tuning(self,params,hyperparam,tuning_basis):
        '''
        hyperparam currently not used; for potential alternative parameterizations
        '''
        pass
    
    def initialize_params(self,key):
        params_init_w = jax.random.normal(key,(self.n_basis,self.n_neuron)) * jnp.sqrt(self.w_init_variance) # prior_hyper here is variance
        # params_init_b = jax.random.normal(key,(self.n_neuron,)) * jnp.sqrt(self.b_init_variance) + self.b_init_mean
        # params_init = (params_init_w,params_init_b)
        params_init = params_init_w
        tuning_init = self.get_tuning(params_init,hyperparam={},tuning_basis=self.tuning_basis)
        self.params = params_init
        self.tuning = tuning_init
        return params_init,tuning_init

    @abstractmethod
    def decode_latent(self,y,tuning,hyperparam,log_latent_transition_kernel_l,log_dynamics_transition_kernel,ma_neuron,ma_latent=None,likelihood_scale=1.,n_time_per_chunk=10000):
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
    
    def init_latent_posterior(self,T,key,random_scale=0.1):
        '''
        initialize the posterior of the latent
        start with equal posterior but add some noise then renormalize
        '''
        posterior = jnp.ones((T,self.n_latent_bin)) / self.n_latent_bin
        posterior = posterior + jax.random.normal(key,shape=posterior.shape) * random_scale
        posterior = posterior / posterior.sum(axis=1,keepdims=True)
        log_posterior = jnp.log(posterior)
        log_posterior = jnp.where(log_posterior ==-jnp.inf,-1e40,log_posterior)
        return log_posterior,posterior
    
    @abstractmethod
    def m_step(self,y,log_posterior_curr,tuning_basis,hyperparam):
        '''
        m-step
        '''
        pass
    
    def fit_em(self,y,hyperparam={},key=jax.random.PRNGKey(0),
                    n_iter=20,
                      posterior_init=None,ma_neuron=None,ma_latent=None,n_time_per_chunk=10000,dt=1.,likelihood_scale=1.,
                      save_every=None,
                      **kwargs):
        '''
        fit the model using EM
        '''

        # use existing or update ingredients for fitting
        tuning_lengthscale = hyperparam.get('tuning_lengthscale',self.tuning_lengthscale)
        movement_variance = hyperparam.get('movement_variance',self.movement_variance)
        p_move_to_jump = hyperparam.get('p_move_to_jump',self.p_move_to_jump)
        p_jump_to_move = hyperparam.get('p_jump_to_move',self.p_jump_to_move)

        if save_every is None:
            save_every = n_iter

        log_posterior_all_saved = []
        params_saved = []
        tuning_saved = []
        iter_saved = []
        log_marginal_saved = []

        log_latent_transition_kernel_l,log_dynamics_transition_kernel,_,_ = gpk.create_transition_prob_1d(self.possible_latent_bin,self.possible_dynamics,movement_variance,p_move_to_jump,p_jump_to_move)
        if ma_neuron is None:
            ma_neuron = self.ma_neuron_default
        if ma_latent is None:
            ma_latent = self.ma_latent_default

        # generate the basis if new tuning_lengthscale is provided
        if 'tuning_lengthscale' in hyperparam:
            tuning_basis = generate_basis(tuning_lengthscale,self.n_latent_bin,self.explained_variance_threshold_basis,include_bias=True)
        else:
            tuning_basis = self.tuning_basis
        
        if posterior_init is None:
            log_posterior_init,posterior_init = self.init_latent_posterior(y.shape[0],key)
            key,_=jax.random.split(key,2)
        
        log_posterior_curr = log_posterior_init
        log_marginal_l = []
        
        for i in tqdm.trange(n_iter):
            # M-step
            m_res = self.m_step(y,log_posterior_curr,tuning_basis,hyperparam) # figure out what i need [[]] return tuning since that's what matters
            params = m_res['params']
            tuning = self.get_tuning(params,hyperparam,tuning_basis)
            # E-step
            log_posterior_all,log_marginal_final,log_prior_curr_all = self.decode_latent(y,tuning,hyperparam,log_latent_transition_kernel_l,log_dynamics_transition_kernel,ma_neuron,ma_latent,likelihood_scale=likelihood_scale,n_time_per_chunk=n_time_per_chunk)
            log_posterior_curr = logsumexp(log_posterior_all,axis=1) # sum over the dynamics dimension; get log posterior over latent
            log_marginal_l.append(log_marginal_final)

            if i % save_every == 0:
                log_posterior_all_saved.append(log_posterior_all)
                params_saved.append(params)
                tuning_saved.append(tuning)
                log_marginal_saved.append(log_marginal_final)
                iter_saved.append(i)
        
        # update attributes
        self.params = params
        self.tuning = tuning
        self.log_posterior_curr = log_posterior_curr
        self.log_marginal_final = log_marginal_final
        self.log_prior_curr_all = log_prior_curr_all
        self.log_latent_transition_kernel_l = log_latent_transition_kernel_l
        self.log_dynamics_transition_kernel = log_dynamics_transition_kernel
        self.tuning_basis = tuning_basis

        em_res = {'log_posterior_all_saved':log_posterior_all_saved,
                  'log_posterior_init':log_posterior_init,
                  'params_saved':params_saved,
                  'tuning_saved':tuning_saved,
                  'iter_saved':iter_saved,
                  'params':params,
                  'tuning':tuning,
                  'log_posterior_final':log_posterior_all,
                  'log_marginal':log_marginal_final,
                  }
        return em_res


        

            






        


class PoissonGPLVMJump1D(AbstractGPLVMJump1D):
    """Poisson GPLVM with jumps.
    The latent governs firing rate; the dynamics governs the transition probabilities between the latent states;
    """
    
    def loglikelihood(self,y,ypred,hyperparam):
        return jax.scipy.stats.poisson.logpmf(y,ypred+1e-40)

    def get_tuning(self,params,hyperparam,tuning_basis):
        tuning = fth.get_tuning_softplus(params,tuning_basis)
        return tuning
        

    def decode_latent(self,y,tuning,hyperparam,log_latent_transition_kernel_l,log_dynamics_transition_kernel,ma_neuron,ma_latent=None,likelihood_scale=1.,n_time_per_chunk=10000):
        '''
        decode the latent and dynamics
        y: observed data, spike counts here; n_time x n_neuron

        log_latent_transition_kernel_l: n_dynamics x n_latent x n_latent
        log_dynamics_transition_kernel: n_dynamics x n_dynamics
        '''
        log_posterior_all,log_marginal_final,log_prior_curr_all = decoder.smooth_all_step_combined_ma_chunk(y,tuning,hyperparam,log_latent_transition_kernel_l,log_dynamics_transition_kernel,ma_neuron,ma_latent,likelihood_scale=likelihood_scale,n_time_per_chunk=n_time_per_chunk,observation_model='poisson')
        return log_posterior_all,log_marginal_final,log_prior_curr_all


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
    
    def get_tuning(self,params,hyperparam,tuning_basis):
        tuning = fth.get_tuning_linear(params,tuning_basis)
        return tuning
        

    def decode_latent(self,y,tuning,hyperparam,log_latent_transition_kernel_l,log_dynamics_transition_kernel,ma_neuron,ma_latent=None,likelihood_scale=1.,n_time_per_chunk=10000):
        '''
        decode the latent and dynamics
        y: observed data, spike counts here; n_time x n_neuron

        log_latent_transition_kernel_l: n_dynamics x n_latent x n_latent
        log_dynamics_transition_kernel: n_dynamics x n_dynamics
        '''
        log_posterior_all,log_marginal_final,log_prior_curr_all = decoder.smooth_all_step_combined_ma_chunk(y,tuning,hyperparam,log_latent_transition_kernel_l,log_dynamics_transition_kernel,ma_neuron,ma_latent,likelihood_scale=likelihood_scale,n_time_per_chunk=n_time_per_chunk,observation_model='gaussian')
        return log_posterior_all,log_marginal_final,log_prior_curr_all


    def sample_y(self,latent_l,hyperparam={},tuning=None,dt=1.,key=jax.random.PRNGKey(10)):
        if tuning is None:
            tuning = self.tuning
        noise_std = hyperparam.get('noise_std',self.noise_std)
        rate = tuning[latent_l,:] * dt
        noise_std = noise_std * jnp.sqrt(dt)
        spk_sim=jax.random.normal(key,shape=rate.shape) * noise_std + rate
        return spk_sim
    
    def m_step(self,y,log_posterior_curr,tuning_basis,hyperparam):
        '''
        m-step
        '''
        y_weighted,t_weighted = fth.get_statistics(log_posterior_curr,y)
        params_new = fth.gaussian_m_step_analytic(tuning_basis,y_weighted,t_weighted,hyperparam['noise_std'])
        m_step_res = {'params':params_new}
        return m_step_res