"""
Minimal KDE vs GPLVM-style M-step (2D basis) debugging helpers.

Use this when supervised KDE tuning + 1 EM step disagrees strongly with KDE on the maze.

Key alignment issues (often cause *large* disagreements):
1) **Latent shape**: `get_tuning_supervised` decoding uses *valid bins only*
   (`tuning_flat`, `coord_to_flat_idx`) while `core_2d.generate_basis_nd` lives on the
   *full* raster `prod(grid_shape)`. GPLVM EM must use the same latent support and masks.
2) **Movement variance units**: `decoder_supervised.get_latent_transition_kernel_multi_maze`
   uses variance in **label units** (meters). `gp_kernel.create_transition_prob_latent_nd`
   uses variance in **grid index** units (bins). Map with ~ var_index = var_physical / bin_size^2
   per dimension (see `movement_variance_index_from_physical`).
3) **Decoder emission convention** (`decoder_supervised.decode_with_dynamics`):
   pass *expected counts per bin* as `tuning` (Hz * dt * gain), because `decoder.py` uses
   `lam = tuning * dt` with internal `dt=1` in the likelihood helper.
4) **M-step vs dt**: `fit_tuning_helper.get_statistics` sums `t_weighted = sum_t p(z|t)` with no
   `dt`. If firing is in Hz, exposure for bin z should be ~ `sum_t p(z|t) * dt`. Core `fit_em`
   does not multiply by `dt` inside `m_step`; try `scale_t_weighted_by_dt=True` below if your
   spike counts are per bin of width `dt` and tuning is Hz.

This module uses the same softplus tuning as `PoissonGPLVMJump2D` (`fit_tuning_helper.get_tuning_softplus`),
not `fit_tuning_with_basis.glm_get_tuning` (that one has an extra per-neuron bias term).

Cluster Jupyter example (paste after you have `tuning_res`, `spk_tsdf`, `label_l`, `dt`):

```python
import jax.numpy as jnp
import poor_man_gplvm.experimental.debug_kde_vs_mstep_2d as dbg

out = dbg.run_kde_vs_mstep_debug(
    tuning_res,
    jnp.asarray(spk_tsdf.d),
    jnp.asarray(label_l.d),
    dt=0.02,
    tuning_lengthscale_bins=5.0,
    continuous_transition_movement_variance_physical=0.1,
    label_bin_size=(0.05, 0.05),
    param_prior_std=1.0,
    m_step_step_size=0.02,
    m_step_maxiter=800,
    scale_t_weighted_by_dt=True,
)
print(out["summary"])
```
"""

import numpy as np
import jax
import jax.numpy as jnp
import jax.scipy as jscipy

import poor_man_gplvm.core_2d as core_2d
import poor_man_gplvm.decoder as decoder
import poor_man_gplvm.fit_tuning_helper as fth
import poor_man_gplvm.gp_kernel as gpk


def movement_variance_index_from_physical(var_physical, label_bin_size_per_dim):
    """
    Map Gaussian movement variance in label space to per-dimension *index* variance
    for `gp_kernel.create_transition_prob_latent_nd` (grid indices 0..n-1).

    If the ND kernel treats dims independently, use var_index_d = var_physical / bin_size_d**2
    (same scaling as a discrete Gaussian on a regular grid with spacing bin_size).
    """
    lbs = np.broadcast_to(np.atleast_1d(label_bin_size_per_dim), (2,))
    v = np.atleast_1d(var_physical)
    if v.size == 1:
        v = np.full(2, float(v[0]))
    return tuple(float(v[d] / (lbs[d] ** 2)) for d in range(2))


def rate_hz_full_from_tuning_xarray(tuning_res):
    """
    Full raster tuning (n_flat, n_neuron) in Hz with NaN on invalid / unoccupied bins.
    """
    da = tuning_res["tuning"]
    v = np.asarray(da.values)
    grid_shape = tuple(int(x) for x in v.shape[:2])
    n_neuron = v.shape[2]
    rate = v.reshape(-1, n_neuron)
    return rate, grid_shape


def ma_latent_full(tuning_res):
    """Boolean (n_flat,) — decode / M-step mask for bins we allow."""
    occ = np.asarray(tuning_res["occupied_mask"]).reshape(-1)
    return occ


def inv_softplus_np(y):
    y = np.maximum(np.asarray(y, dtype=float), 1e-8)
    return np.log(np.expm1(y))


def params_init_lstsq_from_rates(tuning_basis, rate_hz, ridge=1e-6):
    """
    Least-squares init for `fth.get_tuning_softplus`: columns of W minimize ||B w - inv_softplus(rate)||.

    tuning_basis: (n_latent, n_basis)
    rate_hz: (n_latent, n_neuron) — NaNs replaced by 0 before inversion
    """
    B = np.asarray(tuning_basis)
    R = np.nan_to_num(np.asarray(rate_hz), nan=0.0)
    tgt = inv_softplus_np(R)
    n_basis = B.shape[1]
    reg = ridge * np.eye(n_basis)
    W = np.zeros((n_basis, R.shape[1]))
    for n in range(R.shape[1]):
        W[:, n] = np.linalg.lstsq(B.T @ B + reg, B.T @ tgt[:, n], rcond=None)[0]
    return jnp.asarray(W, dtype=jnp.float32)


def build_transition_single_dynamics_log(grid_shape, movement_variance_per_dim):
    """
    Match the common notebook pattern: one dynamics row, stay-put log kernel = 0.

    Returns (log_latent_transition_kernel_l, log_dynamics_transition_kernel) with shapes
    compatible with `decoder.smooth_all_step_combined_ma_chunk` for Poisson.
    """
    K_move, log_K_move = gpk.create_transition_prob_latent_nd(
        tuple(grid_shape), movement_variance_per_dim, custom_kernel=None
    )
    log_K = jnp.log(jnp.maximum(jnp.asarray(log_K_move, dtype=jnp.float32), 1e-30))
    log_latent_transition_kernel_l = log_K[None, :, :]
    log_dynamics_transition_kernel = jnp.zeros((1, 1), dtype=jnp.float32)
    return log_latent_transition_kernel_l, log_dynamics_transition_kernel


def oracle_log_posterior_from_labels(label_xy, grid_shape, bin_centers_per_dim=None):
    """
    Sharp posterior: each time bin assigned to nearest grid index (debug upper bound).

    label_xy: (n_time, 2) physical coords
    If bin_centers_per_dim is None, use np.arange(nx), np.arange(ny) like default GPLVM bins.
    """
    if bin_centers_per_dim is None:
        bx = np.arange(grid_shape[0], dtype=float)
        by = np.arange(grid_shape[1], dtype=float)
    else:
        bx, by = bin_centers_per_dim
    n_flat = int(np.prod(grid_shape))
    xy = np.asarray(label_xy, dtype=float)
    ix = np.argmin(np.abs(xy[:, 0, None] - bx[None, :]), axis=1)
    iy = np.argmin(np.abs(xy[:, 1, None] - by[None, :]), axis=1)
    flat = np.ravel_multi_index((ix, iy), grid_shape)
    one_hot = np.zeros((xy.shape[0], n_flat), dtype=np.float32)
    one_hot[np.arange(xy.shape[0]), flat] = 1.0
    logp = np.log(np.maximum(one_hot, 1e-40))
    return jnp.asarray(logp)


def _m_step_adam_once(
    params_init,
    y,
    log_post_latent,
    tuning_basis,
    param_prior_std,
    opt_state,
    adam_runner,
    scale_t_weighted_by_dt,
    dt,
):
    y_w, t_w = fth.get_statistics(log_post_latent, y)
    if scale_t_weighted_by_dt:
        t_w = t_w * dt
    hyperparam = {"param_prior_std": param_prior_std}
    return adam_runner(
        params_init,
        opt_state,
        hyperparam,
        tuning_basis,
        y_w,
        t_w,
    )


def run_kde_vs_mstep_debug(
    tuning_res,
    spk,
    label_xy,
    dt,
    tuning_lengthscale_bins,
    continuous_transition_movement_variance_physical,
    label_bin_size,
    explained_variance_threshold_basis=0.999,
    basis_type="rbf",
    param_prior_std=1.0,
    m_step_step_size=0.02,
    m_step_maxiter=800,
    m_step_tol=1e-6,
    scale_t_weighted_by_dt=True,
    n_time_per_chunk=20000,
):
    """
    1) Full-grid KDE rate from `tuning_res['tuning']` xr
    2) Same 2D basis as GPLVM (`generate_basis_nd`)
    3) Decode with KDE (expected counts per bin) + transition kernel in *index units*
    4) One GPLVM-style M-step (Adam on `fth.poisson_m_step_objective`)
    5) Oracle M-step with sharp labels (optional comparison)

    Returns dict with kde / glm rates, summaries, and raw arrays.
    """
    spk = jnp.asarray(spk, dtype=jnp.float32)
    rate_hz_full, grid_shape = rate_hz_full_from_tuning_xarray(tuning_res)
    n_flat, n_neuron = rate_hz_full.shape
    ma_lat = jnp.asarray(ma_latent_full(tuning_res), dtype=bool)
    ma_neuron = jnp.ones((n_neuron,), dtype=jnp.float32)

    mv_idx = movement_variance_index_from_physical(
        continuous_transition_movement_variance_physical, label_bin_size
    )
    log_latent_transition_kernel_l, log_dynamics_transition_kernel = build_transition_single_dynamics_log(
        grid_shape, mv_idx
    )

    tuning_basis = core_2d.generate_basis_nd(
        tuning_lengthscale_bins,
        grid_shape,
        explained_variance_threshold_basis=explained_variance_threshold_basis,
        include_bias=True,
        basis_type=basis_type,
        custom_kernel=None,
    )

    rate_hz_j = jnp.asarray(np.nan_to_num(rate_hz_full, nan=0.0), dtype=jnp.float32)
    tuning_eff = rate_hz_j * jnp.float32(dt)

    log_post_all, _, _, _, _, _ = decoder.smooth_all_step_combined_ma_chunk(
        spk,
        tuning_eff,
        {},
        log_latent_transition_kernel_l,
        log_dynamics_transition_kernel,
        ma_neuron,
        ma_latent=ma_lat,
        likelihood_scale=1.0,
        n_time_per_chunk=n_time_per_chunk,
        observation_model="poisson",
    )
    log_post_latent = jscipy.special.logsumexp(log_post_all, axis=1)

    params_init = params_init_lstsq_from_rates(tuning_basis, rate_hz_full)
    adam_runner, opt_init = fth.make_adam_runner(
        fth.poisson_m_step_objective,
        step_size=m_step_step_size,
        maxiter=m_step_maxiter,
        tol=m_step_tol,
    )
    opt_state = opt_init(params_init)
    adam_res = _m_step_adam_once(
        params_init,
        spk,
        log_post_latent,
        tuning_basis,
        param_prior_std,
        opt_state,
        adam_runner,
        scale_t_weighted_by_dt,
        dt,
    )
    params_fit = adam_res["params"]
    rate_glm = np.asarray(fth.get_tuning_softplus(params_fit, tuning_basis))

    bin_centers = tuning_res["bin_centers"]
    log_post_oracle = oracle_log_posterior_from_labels(label_xy, grid_shape, bin_centers_per_dim=bin_centers)
    opt_state2 = opt_init(params_init)
    adam_res_o = _m_step_adam_once(
        params_init,
        spk,
        log_post_oracle,
        tuning_basis,
        param_prior_std,
        opt_state2,
        adam_runner,
        scale_t_weighted_by_dt,
        dt,
    )
    rate_oracle = np.asarray(fth.get_tuning_softplus(adam_res_o["params"], tuning_basis))

    mask = np.asarray(ma_lat) & np.isfinite(rate_hz_full[:, 0])
    kde_m = rate_hz_full[mask]
    glm_m = rate_glm[mask]
    ora_m = rate_oracle[mask]

    def _rmse(a, b):
        return float(np.sqrt(np.nanmean((a - b) ** 2)))

    def _corr(a, b):
        return float(np.corrcoef(a.ravel(), b.ravel())[0, 1])

    summary = {
        "rmse_glm_vs_kde": _rmse(glm_m, kde_m),
        "corr_glm_vs_kde": _corr(glm_m, kde_m),
        "rmse_oracle_vs_kde": _rmse(ora_m, kde_m),
        "corr_oracle_vs_kde": _corr(ora_m, kde_m),
        "n_basis": int(tuning_basis.shape[1]),
        "n_latent_full": int(n_flat),
        "n_occupied_mask": int(mask.sum()),
        "movement_variance_index_per_dim": mv_idx,
    }

    return {
        "summary": summary,
        "rate_hz_kde_full": rate_hz_full,
        "rate_hz_glm_full": rate_glm,
        "rate_hz_oracle_mstep_full": rate_oracle,
        "tuning_basis": tuning_basis,
        "params_init": params_init,
        "params_fit_decode": params_fit,
        "params_fit_oracle": adam_res_o["params"],
        "log_post_latent": np.asarray(log_post_latent),
        "adam_res_decode": adam_res,
        "adam_res_oracle": adam_res_o,
        "ma_latent": np.asarray(ma_lat),
    }


def synthetic_demo(rng_seed=0, grid_shape=(24, 24), n_time=20000, dt=0.02):
    """
    Synthetic maze-like path + Gaussian bump tuning; runs `get_tuning` then pipeline.

    Returns (tuning_res, spk_tsdf, label_l, dt) suitable for `run_kde_vs_mstep_debug`.
    """
    import pynapple as nap

    rng = np.random.default_rng(rng_seed)
    nx, ny = grid_shape
    # random walk in [0, nx-1] x [0, ny-1] with bin size 1 in "label" units
    xy = np.zeros((n_time, 2), dtype=float)
    xy[0] = [nx / 2, ny / 2]
    for t in range(1, n_time):
        step = rng.integers(-1, 2, size=2)
        xy[t] = np.clip(xy[t - 1] + step, [0, 0], [nx - 1, ny - 1])
    # spikes: higher rate near center
    cx, cy = nx / 2, ny / 2
    dist = np.sqrt((xy[:, 0] - cx) ** 2 + (xy[:, 1] - cy) ** 2)
    rate_t = 2.0 + 15.0 * np.exp(-(dist ** 2) / (2 * 4.0 ** 2))
    spk = rng.poisson(rate_t * dt)[:, None].astype(np.float32)

    spk_tsdf = nap.TsdFrame(t=np.arange(n_time) * dt, d=spk)
    label_l = nap.TsdFrame(t=spk_tsdf.t, d=xy, columns=["x", "y"])

    import poor_man_gplvm.supervised_analysis.get_tuning_supervised as gts

    tuning_res = gts.get_tuning(
        spk_tsdf,
        label_l,
        label_bin_size=(1.0, 1.0),
        smooth_std=(2.0, 2.0),
        occupancy_threshold=0.05,
        categorical_labels=[],
    )
    return tuning_res, spk_tsdf, label_l, dt


if __name__ == "__main__":
    tr, spk, lab, dt = synthetic_demo()
    out = run_kde_vs_mstep_debug(
        tr,
        jnp.asarray(spk.d),
        jnp.asarray(lab.d),
        dt=dt,
        tuning_lengthscale_bins=4.0,
        continuous_transition_movement_variance_physical=0.5,
        label_bin_size=(1.0, 1.0),
        param_prior_std=1.0,
        scale_t_weighted_by_dt=True,
    )
    print(out["summary"])
