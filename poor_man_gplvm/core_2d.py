"""2D (ND) GPLVM models — jump variants with Kronecker-product kernels."""

import numpy as np
import jax
import jax.numpy as jnp
import jax.random as jr
from jax import jit
from jax.scipy.special import logsumexp

import pynapple as nap
import tqdm

import poor_man_gplvm.gp_kernel as gpk
import poor_man_gplvm.fit_tuning_helper as fth
import poor_man_gplvm.decoder as decoder
import poor_man_gplvm.core as core


def generate_basis_nd(lengthscale, grid_shape, explained_variance_threshold_basis=0.999,
                      include_bias=True, basis_type='rbf', custom_kernel=None):
    '''
    Kronecker-product ND tuning basis from per-dim 1D bases.

    lengthscale: scalar (same for all dims) or array-like (per dim)
    grid_shape: tuple, e.g. (n_x, n_y)
    custom_kernel: if provided, (n_flat, n_flat) matrix; SVD directly, skip Kronecker
    '''
    grid_shape = tuple(grid_shape)
    n_dim = len(grid_shape)
    n_flat = int(np.prod(grid_shape))

    if custom_kernel is not None:
        tuning_basis = core.generate_basis(
            None, n_flat,
            explained_variance_threshold_basis=explained_variance_threshold_basis,
            include_bias=include_bias,
            basis_type='custom_kernel',
            custom_kernel=custom_kernel,
        )
        return tuning_basis

    ls = jnp.broadcast_to(jnp.atleast_1d(jnp.asarray(lengthscale, dtype=float)), (n_dim,))

    basis_per_dim = []
    for d in range(n_dim):
        b_d = core.generate_basis(
            float(ls[d]), grid_shape[d],
            explained_variance_threshold_basis=explained_variance_threshold_basis,
            include_bias=False,
            basis_type=basis_type,
        )
        basis_per_dim.append(b_d)

    tuning_basis = basis_per_dim[0]
    for b_d in basis_per_dim[1:]:
        tuning_basis = jnp.kron(tuning_basis, b_d)

    if include_bias:
        tuning_basis = jnp.concatenate([jnp.ones((n_flat, 1)), tuning_basis], axis=1)

    return tuning_basis


class AbstractGPLVMJump2D(core.AbstractGPLVMJump1D):
    """GPLVM with smooth ND latent + jumps.

    Latent lives on a grid of shape ``grid_shape``; tuning basis and transition
    kernels are built as Kronecker products of per-dimension 1D kernels.
    The flat index ``n_latent_bin = prod(grid_shape)`` is used everywhere the
    parent expects a 1D latent count, so decoder / M-step code is unchanged.
    """

    def __init__(self, n_neuron, n_latent_bin_per_dim=(50, 50),
                 tuning_lengthscale=5., param_prior_std=1.,
                 movement_variance=1.,
                 explained_variance_threshold_basis=0.999,
                 rng_init_int=123,
                 w_init_variance=1., w_init_mean=0.,
                 p_move_to_jump=0.01, p_jump_to_move=0.01,
                 basis_type='rbf',
                 custom_tuning_kernel=None,
                 custom_transition_kernel=None,
                 smoothness_penalty=0.):

        n_latent_bin_per_dim = tuple(n_latent_bin_per_dim)
        n_dim = len(n_latent_bin_per_dim)

        # ---- store 2D-specific attributes ----
        self.n_latent_bin_per_dim = n_latent_bin_per_dim
        self.grid_shape = n_latent_bin_per_dim
        self.n_latent_bin = int(np.prod(n_latent_bin_per_dim))

        # broadcast scalar -> per-dim arrays
        self.tuning_lengthscale_per_dim = np.broadcast_to(
            np.atleast_1d(np.asarray(tuning_lengthscale, dtype=float)), (n_dim,)).copy()
        self.movement_variance_per_dim = np.broadcast_to(
            np.atleast_1d(np.asarray(movement_variance, dtype=float)), (n_dim,)).copy()

        # keep scalar-ish attributes for parent code that reads self.tuning_lengthscale / .movement_variance
        self.tuning_lengthscale = self.tuning_lengthscale_per_dim
        self.movement_variance = self.movement_variance_per_dim

        self.param_prior_std = param_prior_std
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
        self.smoothness_penalty = smoothness_penalty

        # tuning basis — Kronecker of per-dim 1D bases
        self.basis_type = basis_type
        self.custom_tuning_kernel = custom_tuning_kernel
        self.tuning_basis = generate_basis_nd(
            self.tuning_lengthscale_per_dim, self.grid_shape,
            self.explained_variance_threshold_basis,
            include_bias=True, basis_type=basis_type,
            custom_kernel=self.custom_tuning_kernel,
        )
        self.n_basis = self.tuning_basis.shape[1]

        # transition kernel — triggers _refresh_transition_cache via property setter
        self.custom_transition_kernel = custom_transition_kernel

        # default masks
        self.ma_neuron_default = jnp.ones(self.n_neuron)
        self.ma_latent_default = jnp.ones(self.n_latent_bin)

        # initialize params and tuning
        self.initialize_params(self.rng_init)

    # ------------------------------------------------------------------
    # Transition cache (override 1D versions to use ND Kronecker kernels)
    # ------------------------------------------------------------------

    def _refresh_transition_cache(self):
        (
            latent_transition_kernel_l,
            log_latent_transition_kernel_l,
            dynamics_transition_kernel,
            log_dynamics_transition_kernel,
        ) = gpk.create_transition_prob_nd(
            self.grid_shape,
            self.possible_dynamics,
            self.movement_variance,
            self.p_move_to_jump,
            self.p_jump_to_move,
            custom_kernel=self.custom_transition_kernel,
        )
        self.latent_transition_kernel_l = latent_transition_kernel_l
        self.log_latent_transition_kernel_l = log_latent_transition_kernel_l
        self.dynamics_transition_kernel = dynamics_transition_kernel
        self.log_dynamics_transition_kernel = log_dynamics_transition_kernel
        return self

    def _post_unpickle_fixup_transition_cache(self):
        d = object.__getattribute__(self, "__dict__")
        if "_custom_transition_kernel" not in d:
            if "custom_transition_kernel" in d:
                d["_custom_transition_kernel"] = d.pop("custom_transition_kernel")
            else:
                d["_custom_transition_kernel"] = None
        if "possible_latent_bin" not in d and "n_latent_bin" in d:
            self.possible_latent_bin = jnp.arange(self.n_latent_bin)
        if "possible_dynamics" not in d:
            self.possible_dynamics = jnp.arange(2)
        if "grid_shape" not in d and "n_latent_bin_per_dim" in d:
            d["grid_shape"] = d["n_latent_bin_per_dim"]
        if ("log_latent_transition_kernel_l" not in d) or ("log_dynamics_transition_kernel" not in d):
            self._refresh_transition_cache()
        return self

    # ------------------------------------------------------------------
    # decode_latent — override to use ND kernel creation
    # ------------------------------------------------------------------

    def decode_latent(self, y, tuning=None, hyperparam={}, ma_neuron=None, ma_latent=None,
                      likelihood_scale=1., n_time_per_chunk=10000, t_l=None,
                      custom_transition_kernel=None):
        if isinstance(y, nap.TsdFrame):
            t_l = y.t
            y = y.d
        if tuning is None:
            tuning = self.tuning
        if ma_neuron is None:
            ma_neuron = self.ma_neuron_default
        if ma_latent is None:
            ma_latent = self.ma_latent_default

        movement_variance = hyperparam.get('movement_variance', self.movement_variance)
        p_move_to_jump = hyperparam.get('p_move_to_jump', self.p_move_to_jump)
        p_jump_to_move = hyperparam.get('p_jump_to_move', self.p_jump_to_move)

        if hasattr(self, 'custom_transition_kernel') and custom_transition_kernel is None:
            custom_transition_kernel_ = self.custom_transition_kernel
        else:
            custom_transition_kernel_ = custom_transition_kernel

        if isinstance(custom_transition_kernel_, (list, tuple)):
            p_dynamics_transmat = hyperparam.get('p_dynamics_transmat', None)
            if p_dynamics_transmat is not None:
                res_tup = gpk.create_transition_prob_from_transmat_list(
                    self.possible_latent_bin, custom_transition_kernel_,
                    p_dynamics_transmat=p_dynamics_transmat)
            else:
                p_stay = hyperparam.get('p_stay', None)
                if p_stay is None:
                    p_stay = max(1.0 - self.p_jump_to_move, 1.0 - self.p_move_to_jump)
                res_tup = gpk.create_transition_prob_from_transmat_list(
                    self.possible_latent_bin, custom_transition_kernel_, p_stay=p_stay)
            latent_transition_kernel_l, log_latent_transition_kernel_l, dynamics_transition_kernel, log_dynamics_transition_kernel = res_tup
        else:
            # ---- ND kernel creation (the key difference from 1D) ----
            latent_transition_kernel_l, log_latent_transition_kernel_l, dynamics_transition_kernel, log_dynamics_transition_kernel = gpk.create_transition_prob_nd(
                self.grid_shape, self.possible_dynamics, movement_variance,
                p_move_to_jump, p_jump_to_move, custom_kernel=custom_transition_kernel_)

        log_posterior_all, log_marginal_final, log_causal_posterior_all, log_one_step_predictive_marginals_all, log_accumulated_joint_total, log_likelihood_all = self._decode_latent(
            y, tuning, hyperparam, log_latent_transition_kernel_l, log_dynamics_transition_kernel,
            ma_neuron, ma_latent=ma_latent, likelihood_scale=likelihood_scale,
            n_time_per_chunk=n_time_per_chunk)

        posterior_all = np.exp(log_posterior_all)
        posterior_latent_marg = posterior_all.sum(axis=1)
        posterior_dynamics_marg = posterior_all.sum(axis=2)
        if t_l is not None:
            posterior_latent_marg = nap.TsdFrame(d=posterior_latent_marg, t=t_l)
            posterior_dynamics_marg = nap.TsdFrame(d=posterior_dynamics_marg, t=t_l)

        decoding_res = {
            'log_posterior_all': np.array(log_posterior_all),
            'log_marginal_final': log_marginal_final.item(),
            'posterior_all': posterior_all,
            'posterior_latent_marg': posterior_latent_marg,
            'posterior_dynamics_marg': posterior_dynamics_marg,
            'log_one_step_predictive_marginals_all': log_one_step_predictive_marginals_all,
            'log_likelihood_all': np.array(log_likelihood_all),
        }
        if log_accumulated_joint_total is not None:
            transition_posterior_prob_res = decoder.compute_transition_posterior_prob(log_accumulated_joint_total)
            decoding_res.update(transition_posterior_prob_res)
        return decoding_res

    # ------------------------------------------------------------------
    # fit_em — override to use ND kernel / basis creation
    # ------------------------------------------------------------------

    def fit_em(self, y, hyperparam={}, key=jax.random.PRNGKey(0),
               n_iter=20,
               log_posterior_init=None, opt_state_curr=None,
               ma_neuron=None, ma_latent=None,
               n_time_per_chunk=10000, dt=1., likelihood_scale=1.,
               save_every=None,
               posterior_init_kwargs={'random_scale': 0.1},
               verboase=True,
               custom_transition_kernel=None,
               **kwargs):

        if isinstance(y, nap.TsdFrame):
            y_ = jnp.array(y.d)
        else:
            y_ = jnp.array(y)

        tuning_lengthscale = hyperparam.get('tuning_lengthscale', self.tuning_lengthscale)
        movement_variance = hyperparam.get('movement_variance', self.movement_variance)
        p_move_to_jump = hyperparam.get('p_move_to_jump', self.p_move_to_jump)
        p_jump_to_move = hyperparam.get('p_jump_to_move', self.p_jump_to_move)

        self.tuning_lengthscale = tuning_lengthscale
        self.movement_variance = movement_variance
        self.p_move_to_jump = p_move_to_jump
        self.p_jump_to_move = p_jump_to_move

        if save_every is None:
            save_every = n_iter

        log_posterior_all_saved = []
        params_saved = []
        tuning_saved = []
        iter_saved = []
        log_marginal_saved = []

        custom_transition_kernel_attr = self.custom_transition_kernel if hasattr(self, 'custom_transition_kernel') else None
        custom_transition_kernel_ = custom_transition_kernel_attr if custom_transition_kernel is None else custom_transition_kernel
        if custom_transition_kernel is not None:
            self.custom_transition_kernel = custom_transition_kernel

        # ---- ND kernel creation ----
        _, log_latent_transition_kernel_l, _, log_dynamics_transition_kernel = gpk.create_transition_prob_nd(
            self.grid_shape, self.possible_dynamics, movement_variance,
            p_move_to_jump, p_jump_to_move, custom_kernel=custom_transition_kernel_)

        if ma_neuron is None:
            ma_neuron = self.ma_neuron_default
        if ma_latent is None:
            ma_latent = self.ma_latent_default

        # ---- ND basis (re)generation ----
        if 'tuning_lengthscale' in hyperparam:
            custom_tuning_kernel_ = self.custom_tuning_kernel if hasattr(self, 'custom_tuning_kernel') else None
            tuning_basis = generate_basis_nd(
                tuning_lengthscale, self.grid_shape,
                self.explained_variance_threshold_basis,
                include_bias=True, basis_type=self.basis_type,
                custom_kernel=custom_tuning_kernel_)
        else:
            tuning_basis = self.tuning_basis

        if log_posterior_init is None:
            log_posterior_init, posterior_init = self.init_latent_posterior(y_.shape[0], key, **posterior_init_kwargs)
            key, _ = jax.random.split(key, 2)

        log_posterior_curr = log_posterior_init
        log_marginal_l = []
        m_step_res_l = {}
        params = self.params

        for i in tqdm.trange(n_iter, desc='EM', disable=not verboase):
            m_res = self.m_step(params, y_, log_posterior_curr, tuning_basis, hyperparam, opt_state_curr=opt_state_curr)
            if i == 0:
                m_step_res_l = {k: [] for k in m_res.keys()}
            for k in m_res.keys():
                if k not in ['params', 'opt_state']:
                    m_step_res_l[k].append(m_res[k])

            params = m_res['params']
            opt_state_curr = m_res.get('opt_state', None)
            tuning = self.get_tuning(params, hyperparam, tuning_basis)

            # E-step
            log_posterior_all, log_marginal_final, log_causal_posterior_all, log_one_step_predictive_marginals_allchunk, log_accumulated_joint_total, log_likelihood_all = self._decode_latent(
                y_, tuning, hyperparam, log_latent_transition_kernel_l, log_dynamics_transition_kernel,
                ma_neuron, ma_latent, likelihood_scale=likelihood_scale, n_time_per_chunk=n_time_per_chunk)

            log_posterior_curr = logsumexp(log_posterior_all, axis=1)
            log_marginal_l.append(log_marginal_final)

            if i % save_every == 0:
                log_posterior_all_saved.append(log_posterior_all)
                params_saved.append(params)
                tuning_saved.append(tuning)
                log_marginal_saved.append(log_marginal_final)
                iter_saved.append(i)

        self.params = params
        self.tuning = tuning
        self.log_marginal_final = log_marginal_final
        self.log_latent_transition_kernel_l = log_latent_transition_kernel_l
        self.log_dynamics_transition_kernel = log_dynamics_transition_kernel
        self.tuning_basis = tuning_basis

        posterior = np.exp(log_posterior_all)
        posterior_latent_marg = posterior.sum(axis=1)
        posterior_dynamics_marg = posterior.sum(axis=2)
        if isinstance(y, nap.TsdFrame):
            posterior_latent_marg = nap.TsdFrame(d=posterior_latent_marg, t=y.t)
            posterior_dynamics_marg = nap.TsdFrame(d=posterior_dynamics_marg, t=y.t)

        em_res = {
            'log_posterior_all_saved': log_posterior_all_saved,
            'log_posterior_init': log_posterior_init,
            'params_saved': params_saved,
            'tuning_saved': tuning_saved,
            'iter_saved': iter_saved,
            'params': params,
            'tuning': tuning,
            'log_posterior_final': log_posterior_all,
            'log_marginal': log_marginal_final,
            'log_marginal_l': log_marginal_l,
            'log_marginal_saved': log_marginal_saved,
            'posterior': posterior,
            'posterior_latent_marg': posterior_latent_marg,
            'posterior_dynamics_marg': posterior_dynamics_marg,
            'm_step_res_l': m_step_res_l,
        }
        return em_res

    # ------------------------------------------------------------------
    # sample_latent — override to use ND kernel
    # ------------------------------------------------------------------

    def sample_latent(self, T, key=jax.random.PRNGKey(0), movement_variance=None,
                      p_move_to_jump=0.01, p_jump_to_move=0.01,
                      init_dynamics=None, init_latent=None):
        if movement_variance is None:
            movement_variance = self.movement_variance

        latent_transition_kernel_l, _, dynamics_transition_kernel, _ = gpk.create_transition_prob_nd(
            self.grid_shape, self.possible_dynamics, movement_variance, p_move_to_jump, p_jump_to_move)

        if init_dynamics is None:
            init_dynamics = jax.random.choice(key, self.possible_dynamics)
        if init_latent is None:
            init_latent = jax.random.choice(key, self.possible_latent_bin)
        key_l = jax.random.split(key, T)
        carry_init = (init_dynamics, init_latent)

        @jit
        def step(carry, key):
            k1, k2 = jax.random.split(key, 2)
            dynamics_prev, latent_prev = carry
            dynamics_curr = jax.random.choice(k1, self.possible_dynamics,
                                              p=dynamics_transition_kernel[dynamics_prev])
            latent_curr = jax.random.choice(k2, self.possible_latent_bin,
                                            p=latent_transition_kernel_l[dynamics_curr][latent_prev])
            carry = dynamics_curr, latent_curr
            state = jnp.array([dynamics_curr, latent_curr])
            return carry, state

        _, latent_l = jax.lax.scan(step, carry_init, xs=key_l)
        return latent_l

    # ------------------------------------------------------------------
    # Convenience: flat <-> grid reshaping
    # ------------------------------------------------------------------

    def posterior_to_grid(self, posterior_flat):
        """Reshape flat posterior to grid.

        posterior_flat: (n_time, n_flat) or (n_time, n_dynamics, n_flat)
        Returns: (n_time, *grid_shape) or (n_time, n_dynamics, *grid_shape)
        """
        if isinstance(posterior_flat, nap.TsdFrame):
            posterior_flat = posterior_flat.d
        arr = np.asarray(posterior_flat)
        if arr.ndim == 2:
            return arr.reshape(arr.shape[0], *self.grid_shape)
        elif arr.ndim == 3:
            return arr.reshape(arr.shape[0], arr.shape[1], *self.grid_shape)
        return arr.reshape(*arr.shape[:-1], *self.grid_shape)

    def tuning_to_grid(self, tuning_flat=None):
        """Reshape tuning (n_flat, n_neuron) -> (*grid_shape, n_neuron)."""
        if tuning_flat is None:
            tuning_flat = self.tuning
        return np.asarray(tuning_flat).reshape(*self.grid_shape, -1)

    def flat_to_grid_idx(self, flat_idx):
        """flat_idx (int or array) -> tuple of per-dim index arrays."""
        return np.unravel_index(flat_idx, self.grid_shape)

    def grid_to_flat_idx(self, grid_idx):
        """grid_idx: tuple of per-dim indices -> flat index array."""
        return np.ravel_multi_index(grid_idx, self.grid_shape)


# ======================================================================
# Concrete classes
# ======================================================================

class PoissonGPLVMJump2D(AbstractGPLVMJump2D):
    """Poisson GPLVM with jumps, ND latent (Kronecker kernels)."""

    def __getstate__(self):
        state = self.__dict__.copy()
        state['adam_runner'] = None
        state['opt_state_init_fun'] = None
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        d = object.__getattribute__(self, "__dict__")
        if "custom_tuning_kernel" not in d:
            d["custom_tuning_kernel"] = None
        if "basis_type" not in d:
            d["basis_type"] = "rbf"
        if ("possible_latent_bin" not in d) and ("n_latent_bin" in d):
            d["possible_latent_bin"] = jnp.arange(self.n_latent_bin)
        if "possible_dynamics" not in d:
            d["possible_dynamics"] = jnp.arange(2)
        if ("ma_neuron_default" not in d) and ("n_neuron" in d):
            d["ma_neuron_default"] = jnp.ones(self.n_neuron)
        if ("ma_latent_default" not in d) and ("n_latent_bin" in d):
            d["ma_latent_default"] = jnp.ones(self.n_latent_bin)
        if "grid_shape" not in d and "n_latent_bin_per_dim" in d:
            d["grid_shape"] = d["n_latent_bin_per_dim"]
        if ("tuning_basis" not in d) and ("n_latent_bin_per_dim" in d):
            d["tuning_basis"] = generate_basis_nd(
                getattr(self, "tuning_lengthscale_per_dim",
                        getattr(self, "tuning_lengthscale", 5.)),
                self.grid_shape,
                getattr(self, "explained_variance_threshold_basis", 0.999),
                include_bias=True,
                basis_type=getattr(self, "basis_type", "rbf"),
                custom_kernel=getattr(self, "custom_tuning_kernel", None),
            )
            d["n_basis"] = d["tuning_basis"].shape[1]
        self._post_unpickle_fixup_transition_cache()

    def loglikelihood(self, y, ypred, hyperparam):
        return jax.scipy.stats.poisson.logpmf(y, ypred + 1e-40)

    def get_tuning(self, params, hyperparam, tuning_basis):
        return fth.get_tuning_softplus(params, tuning_basis)

    def _decode_latent(self, y, tuning, hyperparam,
                       log_latent_transition_kernel_l, log_dynamics_transition_kernel,
                       ma_neuron, ma_latent=None, likelihood_scale=1., n_time_per_chunk=10000):
        return decoder.smooth_all_step_combined_ma_chunk(
            y, tuning, hyperparam,
            log_latent_transition_kernel_l, log_dynamics_transition_kernel,
            ma_neuron, ma_latent, likelihood_scale=likelihood_scale,
            n_time_per_chunk=n_time_per_chunk, observation_model='poisson')

    def decode_latent_naive_bayes(self, y, tuning=None, hyperparam={},
                                 ma_neuron=None, ma_latent=None,
                                 likelihood_scale=1., n_time_per_chunk=10000,
                                 dt_l=1., t_l=None):
        return super().decode_latent_naive_bayes(
            y, tuning=tuning, hyperparam=hyperparam, ma_neuron=ma_neuron,
            ma_latent=ma_latent, likelihood_scale=likelihood_scale,
            n_time_per_chunk=n_time_per_chunk, dt_l=dt_l,
            observation_model='poisson', t_l=t_l)

    def sample_y(self, latent_l, hyperparam={}, tuning=None, dt=1., key=jax.random.PRNGKey(10)):
        if tuning is None:
            tuning = self.tuning
        rate = tuning[latent_l, :]
        return jax.random.poisson(key, rate * dt)

    def m_step(self, param_curr, y, log_posterior_curr, tuning_basis, hyperparam, opt_state_curr=None):
        y_weighted, t_weighted = fth.get_statistics(log_posterior_curr, y)
        adam_res = self.adam_runner(
            param_curr, opt_state_curr, hyperparam, tuning_basis, y_weighted, t_weighted)
        n_iter = adam_res['n_iter']
        return {
            'params': adam_res['params'],
            'opt_state': adam_res['opt_state'],
            'n_iter': n_iter,
            'final_loss': adam_res['final_loss'],
            'final_error': adam_res['final_error'],
            'loss_history': adam_res['loss_history'][:n_iter],
            'error_history': adam_res['error_history'][:n_iter],
        }

    def fit_em(self, y, hyperparam={}, key=jax.random.PRNGKey(0),
               n_iter=20, log_posterior_init=None, ma_neuron=None, ma_latent=None,
               n_time_per_chunk=10000, dt=1., likelihood_scale=1.,
               save_every=None,
               m_step_step_size=0.01, m_step_maxiter=1000, m_step_tol=1e-6,
               **kwargs):
        hyperparam_ = hyperparam.copy()
        hyperparam_['param_prior_std'] = hyperparam_.get('param_prior_std', self.param_prior_std)
        hyperparam_['smoothness_penalty'] = hyperparam_.get('smoothness_penalty', self.smoothness_penalty)
        self.adam_runner, self.opt_state_init_fun = fth.make_adam_runner(
            fth.poisson_m_step_objective_smoothness if self.basis_type == 'bspline' else fth.poisson_m_step_objective,
            step_size=m_step_step_size, maxiter=m_step_maxiter, tol=m_step_tol)
        opt_state_curr = self.opt_state_init_fun(self.params)
        return super().fit_em(
            y, hyperparam=hyperparam_, key=key, n_iter=n_iter,
            log_posterior_init=log_posterior_init, opt_state_curr=opt_state_curr,
            ma_neuron=ma_neuron, ma_latent=ma_latent,
            n_time_per_chunk=n_time_per_chunk, dt=dt,
            likelihood_scale=likelihood_scale, save_every=save_every, **kwargs)


class GaussianGPLVMJump2D(AbstractGPLVMJump2D):
    """Gaussian GPLVM with jumps, ND latent (Kronecker kernels)."""

    def __init__(self, n_neuron, noise_std=0.5, **kwargs):
        super().__init__(n_neuron, **kwargs)
        self.noise_std = noise_std

    def loglikelihood(self, y, ypred, hyperparam):
        return jax.scipy.stats.norm.logpdf(y, ypred, hyperparam['noise_std'])

    def get_tuning(self, params, hyperparam, tuning_basis):
        return fth.get_tuning_linear(params, tuning_basis)

    def _decode_latent(self, y, tuning, hyperparam,
                       log_latent_transition_kernel_l, log_dynamics_transition_kernel,
                       ma_neuron, ma_latent=None, likelihood_scale=1., n_time_per_chunk=10000):
        return decoder.smooth_all_step_combined_ma_chunk(
            y, tuning, hyperparam,
            log_latent_transition_kernel_l, log_dynamics_transition_kernel,
            ma_neuron, ma_latent, likelihood_scale=likelihood_scale,
            n_time_per_chunk=n_time_per_chunk, observation_model='gaussian')

    def decode_latent(self, y, tuning=None, hyperparam={}, ma_neuron=None, ma_latent=None,
                      likelihood_scale=1., n_time_per_chunk=10000,
                      custom_transition_kernel=None):
        hyperparam_ = hyperparam.copy()
        hyperparam_['noise_std'] = hyperparam_.get('noise_std', self.noise_std)
        return super().decode_latent(
            y, tuning=tuning, hyperparam=hyperparam_, ma_neuron=ma_neuron,
            ma_latent=ma_latent, likelihood_scale=likelihood_scale,
            n_time_per_chunk=n_time_per_chunk,
            custom_transition_kernel=custom_transition_kernel)

    def decode_latent_naive_bayes(self, y, tuning=None, hyperparam={},
                                 ma_neuron=None, ma_latent=None,
                                 likelihood_scale=1., n_time_per_chunk=10000,
                                 dt_l=1., t_l=None):
        hyperparam_ = hyperparam.copy()
        hyperparam_['noise_std'] = hyperparam_.get('noise_std', self.noise_std)
        return super().decode_latent_naive_bayes(
            y, tuning=tuning, hyperparam=hyperparam_, ma_neuron=ma_neuron,
            ma_latent=ma_latent, likelihood_scale=likelihood_scale,
            n_time_per_chunk=n_time_per_chunk, dt_l=dt_l,
            observation_model='gaussian', t_l=t_l)

    def sample_y(self, latent_l, hyperparam={}, tuning=None, dt=1., key=jax.random.PRNGKey(10)):
        if tuning is None:
            tuning = self.tuning
        noise_std = hyperparam.get('noise_std', self.noise_std)
        rate = tuning[latent_l, :] * dt
        noise_std = noise_std * jnp.sqrt(dt)
        return jax.random.normal(key, shape=rate.shape) * noise_std + rate

    def m_step(self, param_curr, y, log_posterior_curr, tuning_basis, hyperparam, opt_state_curr=None):
        y_weighted, t_weighted = fth.get_statistics(log_posterior_curr, y)
        params_new = fth.gaussian_m_step_analytic(hyperparam, tuning_basis, y_weighted, t_weighted)
        return {'params': params_new, 'opt_state': None}

    def fit_em(self, y, hyperparam={}, key=jax.random.PRNGKey(0),
               n_iter=20, log_posterior_init=None, ma_neuron=None, ma_latent=None,
               n_time_per_chunk=10000, dt=1., likelihood_scale=1.,
               save_every=None, **kwargs):
        hyperparam_ = hyperparam.copy()
        hyperparam_['noise_std'] = hyperparam_.get('noise_std', self.noise_std)
        hyperparam_['param_prior_std'] = hyperparam_.get('param_prior_std', self.param_prior_std)
        return super().fit_em(
            y, hyperparam=hyperparam_, key=key, n_iter=n_iter,
            log_posterior_init=log_posterior_init, ma_neuron=ma_neuron,
            ma_latent=ma_latent, n_time_per_chunk=n_time_per_chunk, dt=dt,
            likelihood_scale=likelihood_scale, save_every=save_every, **kwargs)
