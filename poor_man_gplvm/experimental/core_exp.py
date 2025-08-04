'''
Models that are not yet ready for production
'''

import jax
import jax.numpy as jnp
import jax.random as jr
import jax.scipy as jsp
import jax.scipy.stats as jsps
import jax.numpy as jnp

from poor_man_gplvm import PoissonGPLVMJump1D
from poor_man_gplvm import fit_tuning_helper as fth
from poor_man_gplvm import gp_kernel as gpk
from poor_man_gplvm import experimental.fit_tuning_helper_exp as fth_exp
from poor_man_gplvm import experimental.decoder_exp as dec_exp

# gain model
# at each time learn a gain variable that controls the population
# the firing rate is now given by g_t lambda_tn
# the MLE estimator of g is given by s / lambda
# params now given by {'weight':,'gain':}


class PoissonGPLVMGain1D_gain(PoissonGPLVMJump1D):
    
    def __init__(self, n_neuron, gain_init_mean=1., gain_init_variance=0.1, **kwargs):
        super().__init__(n_neuron, **kwargs)
        self.gain_init_mean = gain_init_mean
        self.gain_init_variance = gain_init_variance
        
    def initialize_params(self, key):
        super().initialize_params(key)
        self.gain_init = 1.
        # Initialize gain as class attribute (will be set properly during fit_em)
        self.gain = None
        
    def get_gain(self, y, log_posterior_curr):
        '''
        M-step for gain: total spike / total predicted rate for each time
        '''
        return fth_exp.get_gain_mstep(y, log_posterior_curr, self.tuning)
        
    def get_gain_chunk(self, y, log_posterior_curr, n_time_per_chunk=10000):
        '''
        Chunked version of get_gain for large arrays
        '''
        return fth_exp.get_gain_mstep_chunk(y, log_posterior_curr, self.tuning, n_time_per_chunk)
        
    def sample_y(self, latent_l, hyperparam={}, tuning=None, dt=1., gain=None, key=jax.random.PRNGKey(10)):
        if tuning is None:
            tuning = self.tuning
        if gain is None:
            if self.gain is not None and len(self.gain) == len(latent_l):
                gain = self.gain
            else:
                gain = jnp.ones(len(latent_l))
        rate = tuning[latent_l, :] * gain[:, None]
        spk_sim = jax.random.poisson(key, rate * dt)
        return spk_sim
        
    def sample(self, T, hyperparam={}, key=jax.random.PRNGKey(0),
               init_dynamics=None, init_latent=None, dt=1., tuning=None, gain=None):
        '''
        sample both latent and y with gain
        '''
        key_l = jax.random.split(key, T+1)
        movement_variance = hyperparam.get('movement_variance', self.movement_variance)
        p_move_to_jump = hyperparam.get('p_move_to_jump', self.p_move_to_jump)
        p_jump_to_move = hyperparam.get('p_jump_to_move', self.p_jump_to_move)
        latent_l = self.sample_latent(T, key_l[0], movement_variance, p_move_to_jump, p_jump_to_move, init_dynamics, init_latent)
        if gain is None:
            if self.gain is not None and len(self.gain) == T:
                gain = self.gain
            else:
                gain = jnp.ones(T)
        y_l = self.sample_y(latent_l[:, 1], hyperparam, tuning, dt, gain, key_l[1])
        return latent_l, y_l
        
    def _decode_latent(self, y, tuning, hyperparam, log_latent_transition_kernel_l, log_dynamics_transition_kernel, ma_neuron, ma_latent=None, likelihood_scale=1., n_time_per_chunk=10000, gain=None):
        '''
        Decode with gain parameter
        '''
        if gain is None:
            if self.gain is not None and len(self.gain) == len(y):
                gain = self.gain
            else:
                gain = jnp.ones(len(y))
        import experimental.decoder_exp as dec_exp
        return dec_exp.smooth_all_step_combined_ma_chunk_gain(
            y, tuning, hyperparam, log_latent_transition_kernel_l, log_dynamics_transition_kernel,
            ma_neuron, ma_latent, likelihood_scale, n_time_per_chunk, 'poisson', gain)
            
    def decode_latent_naive_bayes(self, y, tuning=None, hyperparam={}, ma_neuron=None, ma_latent=None, likelihood_scale=1., n_time_per_chunk=10000, dt_l=1., gain=None):
        '''
        Naive Bayes decoding with gain
        '''
        if tuning is None:
            tuning = self.tuning
        if ma_neuron is None:
            ma_neuron = self.ma_neuron_default
        if ma_latent is None:
            ma_latent = self.ma_latent_default
        if gain is None:
            if self.gain is not None and len(self.gain) == len(y):
                gain = self.gain
            else:
                gain = jnp.ones(len(y))
            
        
        log_post_l, log_marginal_l, log_marginal_total = dec_exp.get_naive_bayes_ma_chunk_gain(
            y, tuning, hyperparam, ma_neuron, ma_latent, dt_l, n_time_per_chunk, 'poisson', gain)
        
        decoding_res = {
            'log_posterior': log_post_l,
            'log_marginal_l': log_marginal_l, 
            'log_marginal': log_marginal_total
        }
        return decoding_res
        
    def m_step(self, param_curr, y, log_posterior_curr, tuning_basis, hyperparam, opt_state_curr=None, gain_curr=None):
        '''
        M-step with gain parameter
        '''
        if gain_curr is None:
            if self.gain is not None and len(self.gain) == len(y):
                gain_curr = self.gain
            else:
                gain_curr = jnp.ones(len(y))
            
        
        # Get gain-weighted statistics
        y_weighted, t_weighted, gain_weighted = fth_exp.get_statistics_gain(log_posterior_curr, y, gain_curr)
        
        # Update tuning parameters
        adam_res = self.adam_runner(
            param_curr, opt_state_curr, hyperparam, tuning_basis, y_weighted, t_weighted, gain_weighted
        )
        
        # Update gain
        self.tuning = self.get_tuning(adam_res['params'], hyperparam, tuning_basis)
        if len(y) > 50000:  # Use chunked version for large data
            gain_new = self.get_gain_chunk(y, log_posterior_curr)
        else:
            gain_new = self.get_gain(y, log_posterior_curr)
        self.gain = gain_new
        
        n_iter = adam_res['n_iter']
        loss_history_trimmed = adam_res['loss_history'][:n_iter]
        error_history_trimmed = adam_res['error_history'][:n_iter]
        
        m_step_res = {
            'params': adam_res['params'],
            'tuning':self.tuning,
            'gain': gain_new,
            'opt_state': adam_res['opt_state'],
            'n_iter': adam_res['n_iter'],
            'final_loss': adam_res['final_loss'],
            'final_error': adam_res['final_error'],
            'loss_history': loss_history_trimmed,
            'error_history': error_history_trimmed
        }
        return m_step_res
        
    def fit_em(self, y, hyperparam={}, key=jax.random.PRNGKey(0),
               n_iter=20, log_posterior_init=None, ma_neuron=None, ma_latent=None,
               n_time_per_chunk=10000, dt=1., likelihood_scale=1.,
               save_every=None, gain_init=None,
               m_step_step_size=0.01, m_step_maxiter=1000, m_step_tol=1e-6,
               **kwargs):
               
        hyperparam['param_prior_std'] = hyperparam.get('param_prior_std', self.param_prior_std)
        
        # Initialize gain if not provided
        if gain_init is None:
            gain_init = jnp.ones(len(y))
            
        # Initialize gain as class attribute
        self.gain = gain_init
            
        # Create the adam runner for tuning parameters
        import experimental.fit_tuning_helper_exp as fth_exp
        self.adam_runner, opt_state_init_fun = fth_exp.make_adam_runner(
            fth_exp.poisson_m_step_objective_gain,
            step_size=m_step_step_size,
            maxiter=m_step_maxiter,
            tol=m_step_tol
        )
        opt_state_curr = opt_state_init_fun(self.params)
        
        # Initialize transition kernels
        movement_variance = hyperparam.get('movement_variance', self.movement_variance)
        p_move_to_jump = hyperparam.get('p_move_to_jump', self.p_move_to_jump)
        p_jump_to_move = hyperparam.get('p_jump_to_move', self.p_jump_to_move)
        
        _, self.log_latent_transition_kernel_l, _, self.log_dynamics_transition_kernel = gpk.create_transition_prob_1d(
            self.possible_latent_bin, self.possible_dynamics, movement_variance, p_move_to_jump, p_jump_to_move)
        
        # Custom EM loop to handle gain properly
        if ma_neuron is None:
            ma_neuron = self.ma_neuron_default
        if ma_latent is None:
            ma_latent = self.ma_latent_default
            
        if log_posterior_init is None:
            log_posterior_init, _ = self.init_latent_posterior(len(y), key)
            
        log_posterior_curr = log_posterior_init
        param_curr = self.params
        gain_curr = self.gain
        
        em_history = {
            'log_likelihood': [],
            'params': [],
            'gain': [],
            'log_posterior': []
        }
        
        for i in range(n_iter):
            
            # M-step: update both tuning and gain
            m_step_res = self.m_step(param_curr, y, log_posterior_curr, self.tuning_basis, 
                                   hyperparam, opt_state_curr, gain_curr)
            param_curr = m_step_res['params']
            gain_curr = m_step_res['gain']
            opt_state_curr = m_step_res['opt_state']
            tuning = m_step_res['tuning']
            
            # Use gain-aware decoder
            decode_res = self._decode_latent(
                y, tuning, hyperparam, 
                self.log_latent_transition_kernel_l, self.log_dynamics_transition_kernel,
                ma_neuron, ma_latent, likelihood_scale, n_time_per_chunk, gain_curr)
            log_posterior_curr, log_marginal_final, _ = decode_res
            
            
            
            # Store history
            em_history['log_likelihood'].append(log_marginal_final)
            if save_every is not None and i % save_every == 0:
                em_history['params'].append(param_curr)
                em_history['tuning'].append(tuning)
                em_history['gain'].append(gain_curr)
                em_history['log_posterior'].append(log_posterior_curr)
                
        # Final update
        self.params = param_curr
        self.tuning = tuning
        self.gain = gain_curr
        
        em_res = {
            'params': param_curr,
            'gain': gain_curr,
            'tuning': tuning,
            'log_posterior': log_posterior_curr,
            'log_likelihood': jnp.array(em_history['log_likelihood']),
            'history': em_history
        }
        
        return em_res
        

