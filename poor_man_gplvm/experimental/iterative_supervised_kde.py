"""
Iterative supervised KDE tuning + no-jump dynamics decoding.

Pipeline:
1) Fit supervised tuning from labels (`get_tuning_supervised.get_tuning`)
2) Decode with dynamics (`decoder_supervised.decode_with_dynamics`, no jump)
3) Refit tuning using posterior-weighted occupancy/spike counts per latent bin
4) Repeat decode/refit for `n_iter`

The posterior-weighted KDE update reuses the same smoothing matrix structure as
the supervised initialization, but projected to valid bins used in decoding.

Cluster Jupyter example:

```python
import poor_man_gplvm.experimental.iterative_supervised_kde as iskde

res = iskde.fit_decode_iterative_supervised_kde(
    spk_mat,
    label,
    n_iter=4,
    label_bin_size=(0.05, 0.05),
    smooth_std=(0.08, 0.08),
    occupancy_threshold=0.02,
    dt=0.02,
    continuous_transition_movement_variance=(0.01, 0.01),
    n_time_per_chunk=20000,
    stats_chunk_size=50000,
    verbose=True,
)

# xarray outputs
res["initial"]["tuning_flat_xr"]
res["initial"]["decode"]["posterior_latent_marg_xr"]
res["final"]["tuning_flat_xr"]
res["final"]["decode"]["posterior_latent_marg_xr"]
```
"""

import numpy as np
import pandas as pd
import xarray as xr
import jax
import jax.numpy as jnp
import pynapple as nap

import poor_man_gplvm.fit_tuning_helper as fit_tuning_helper
import poor_man_gplvm.supervised_analysis.decoder_supervised as decoder_supervised
import poor_man_gplvm.supervised_analysis.get_tuning_supervised as get_tuning_supervised
import poor_man_gplvm.supervised_analysis.xarray_wrappers as xarray_wrappers


@jax.jit
def _posterior_stats_jit(log_posterior_latent, spk_chunk):
    y_weighted, t_weighted = fit_tuning_helper.get_statistics(log_posterior_latent, spk_chunk)
    return y_weighted, t_weighted


@jax.jit
def _smooth_ratio_update_jit(spk_count_weighted, occupancy_weighted, smoothing_mat_valid, eps):
    spk_count_smth = smoothing_mat_valid @ spk_count_weighted
    occupancy_smth = smoothing_mat_valid @ occupancy_weighted
    tuning_flat = spk_count_smth / (occupancy_smth[:, None] + eps)
    return tuning_flat, spk_count_smth, occupancy_smth


def _row_normalize(mat, eps=1e-12):
    row_sum = mat.sum(axis=1, keepdims=True)
    row_sum = np.where(row_sum > eps, row_sum, 1.0)
    return mat / row_sum


def _as_nap_tsdframe(arr, dt, columns):
    t = np.arange(arr.shape[0], dtype=float) * float(dt)
    return nap.TsdFrame(t=t, d=arr, columns=columns)


def _ensure_supervised_inputs(spk_mat, label, dt):
    if hasattr(spk_mat, "d"):
        spk_tsdf = spk_mat
    else:
        spk_arr = np.asarray(spk_mat)
        neuron_cols = np.arange(spk_arr.shape[1])
        spk_tsdf = _as_nap_tsdframe(spk_arr, dt=dt, columns=neuron_cols)

    if isinstance(label, dict):
        label_out = {}
        for maze_key, label_one in label.items():
            if hasattr(label_one, "d"):
                label_out[maze_key] = label_one
            else:
                label_arr = np.asarray(label_one)
                label_cols = [f"label_{i}" for i in range(label_arr.shape[1])]
                label_out[maze_key] = _as_nap_tsdframe(label_arr, dt=dt, columns=label_cols)
        return spk_tsdf, label_out

    if hasattr(label, "d"):
        return spk_tsdf, label

    label_arr = np.asarray(label)
    label_cols = [f"label_{i}" for i in range(label_arr.shape[1])]
    label_tsdf = _as_nap_tsdframe(label_arr, dt=dt, columns=label_cols)
    return spk_tsdf, label_tsdf


def _prepare_full_grid_iteration_ops(tuning_res):
    if isinstance(tuning_res["smoothing_matrix"], dict):
        op_l = []
        for maze_key, smoothing_full in tuning_res["smoothing_matrix"].items():
            valid_mask_local = np.asarray(tuning_res["valid_flat_mask"][maze_key], dtype=bool)
            global_flat_idx = np.asarray(tuning_res["coord_to_flat_idx"][maze_key].values, dtype=int)
            op_l.append(
                {
                    "maze_key": maze_key,
                    "smoothing_full_j": jnp.asarray(np.asarray(smoothing_full), dtype=jnp.float32),
                    "valid_mask_local": valid_mask_local,
                    "global_flat_idx": global_flat_idx,
                    "n_full": int(valid_mask_local.size),
                }
            )
        return {"is_multi": True, "ops": op_l}

    valid_mask_local = np.asarray(tuning_res["valid_flat_mask"], dtype=bool)
    return {
        "is_multi": False,
        "ops": [
            {
                "maze_key": "single",
                "smoothing_full_j": jnp.asarray(np.asarray(tuning_res["smoothing_matrix"]), dtype=jnp.float32),
                "valid_mask_local": valid_mask_local,
                "global_flat_idx": np.arange(int(np.asarray(tuning_res["tuning_flat"]).shape[0]), dtype=int),
                "n_full": int(valid_mask_local.size),
            }
        ],
    }


def _kde_update_on_full_grid(y_weighted_valid, t_weighted_valid, full_grid_ops, dt, eps):
    n_latent_valid = int(y_weighted_valid.shape[0])
    n_neuron = int(y_weighted_valid.shape[1])
    tuning_valid_next = np.zeros((n_latent_valid, n_neuron), dtype=float)
    occ_smth_valid_next = np.zeros((n_latent_valid,), dtype=float)
    spk_smth_valid_next = np.zeros((n_latent_valid, n_neuron), dtype=float)
    tuning_full_next = {}

    for one in full_grid_ops["ops"]:
        global_flat_idx = one["global_flat_idx"]
        valid_mask_local = one["valid_mask_local"]
        smoothing_full_j = one["smoothing_full_j"]
        n_full = int(one["n_full"])
        maze_key = one["maze_key"]

        spk_full_local = np.zeros((n_full, n_neuron), dtype=float)
        occ_full_local = np.zeros((n_full,), dtype=float)
        spk_full_local[valid_mask_local] = np.asarray(y_weighted_valid)[global_flat_idx]
        occ_full_local[valid_mask_local] = np.asarray(t_weighted_valid)[global_flat_idx] * float(dt)

        tuning_full_local, spk_smth_full_local, occ_smth_full_local = _smooth_ratio_update_jit(
            jnp.asarray(spk_full_local, dtype=jnp.float32),
            jnp.asarray(occ_full_local, dtype=jnp.float32),
            smoothing_full_j,
            jnp.asarray(eps, dtype=jnp.float32),
        )
        tuning_full_next[maze_key] = np.asarray(jax.device_get(tuning_full_local))

        tuning_valid_next[global_flat_idx] = np.asarray(jax.device_get(tuning_full_local))[valid_mask_local]
        occ_smth_valid_next[global_flat_idx] = np.asarray(jax.device_get(occ_smth_full_local))[valid_mask_local]
        spk_smth_valid_next[global_flat_idx] = np.asarray(jax.device_get(spk_smth_full_local))[valid_mask_local]

    return tuning_valid_next, spk_smth_valid_next, occ_smth_valid_next, tuning_full_next


def _posterior_weighted_stats_chunked(log_posterior_latent, spk_arr, stats_chunk_size):
    log_posterior_latent = jnp.asarray(log_posterior_latent)
    spk_arr = jnp.asarray(spk_arr)
    n_time = int(log_posterior_latent.shape[0])
    if stats_chunk_size is None or stats_chunk_size <= 0 or stats_chunk_size >= n_time:
        return _posterior_stats_jit(log_posterior_latent, spk_arr)

    n_latent = int(log_posterior_latent.shape[1])
    n_neuron = int(spk_arr.shape[1])
    y_weighted = jnp.zeros((n_latent, n_neuron), dtype=spk_arr.dtype)
    t_weighted = jnp.zeros((n_latent,), dtype=spk_arr.dtype)

    start = 0
    while start < n_time:
        end = min(start + int(stats_chunk_size), n_time)
        y_chunk, t_chunk = _posterior_stats_jit(
            log_posterior_latent[start:end],
            spk_arr[start:end],
        )
        y_weighted = y_weighted + y_chunk
        t_weighted = t_weighted + t_chunk
        start = end
    return y_weighted, t_weighted


def _make_decode_xr(decode_res, flat_idx_to_coord, time_coord):
    posterior_latent_marg_xr = xarray_wrappers.latent_time_series_to_xr(
        decode_res["posterior_latent_marg"],
        flat_idx_to_coord,
        time_coord=time_coord,
        dim_name="posterior_latent_marg",
        split_maze=False,
    )
    log_posterior_latent_marg_xr = xarray_wrappers.latent_time_series_to_xr(
        decode_res["log_posterior_latent_marg"],
        flat_idx_to_coord,
        time_coord=time_coord,
        dim_name="log_posterior_latent_marg",
        split_maze=False,
    )
    log_likelihood_xr = xarray_wrappers.latent_time_series_to_xr(
        decode_res["log_likelihood"],
        flat_idx_to_coord,
        time_coord=time_coord,
        dim_name="log_likelihood",
        split_maze=False,
    )

    label_coord = posterior_latent_marg_xr.coords["label_bin"]
    posterior_all = np.asarray(decode_res["posterior_all"])
    log_posterior_all = np.asarray(decode_res["log_posterior_all"])
    dyn_coord = np.arange(posterior_all.shape[1])

    posterior_all_xr = xr.DataArray(
        posterior_all,
        dims=("time", "dyn", "label_bin"),
        coords={"time": np.asarray(time_coord), "dyn": dyn_coord, "label_bin": label_coord},
        name="posterior_all",
    )
    log_posterior_all_xr = xr.DataArray(
        log_posterior_all,
        dims=("time", "dyn", "label_bin"),
        coords={"time": np.asarray(time_coord), "dyn": dyn_coord, "label_bin": label_coord},
        name="log_posterior_all",
    )
    posterior_dynamics_marg_xr = xr.DataArray(
        np.asarray(decode_res["posterior_dynamics_marg"]),
        dims=("time", "dyn"),
        coords={"time": np.asarray(time_coord), "dyn": dyn_coord},
        name="posterior_dynamics_marg",
    )
    log_marginal_l_xr = xr.DataArray(
        np.asarray(decode_res["log_marginal_l"]),
        dims=("time",),
        coords={"time": np.asarray(time_coord)},
        name="log_marginal_l",
    )

    return {
        "posterior_latent_marg_xr": posterior_latent_marg_xr,
        "log_posterior_latent_marg_xr": log_posterior_latent_marg_xr,
        "log_likelihood_xr": log_likelihood_xr,
        "posterior_all_xr": posterior_all_xr,
        "log_posterior_all_xr": log_posterior_all_xr,
        "posterior_dynamics_marg_xr": posterior_dynamics_marg_xr,
        "log_marginal_l_xr": log_marginal_l_xr,
    }


def _make_tuning_flat_xr(tuning_flat, tuning_res):
    return xarray_wrappers.tuning_flat_to_xr(
        tuning_flat,
        tuning_res["flat_idx_to_coord"],
        neuron_names=tuning_res.get("neuron_names", None),
        split_maze=False,
    )


def _single_maze_tuning_grid_from_full(tuning_full_single, tuning_res):
    grid_shape = tuple(int(x) for x in tuning_res["grid_shape"])
    grid_arr = np.asarray(tuning_full_single).reshape((*grid_shape, int(np.asarray(tuning_full_single).shape[1])))
    template = tuning_res["tuning"]
    return xr.DataArray(
        grid_arr,
        dims=template.dims,
        coords=template.coords,
        name="tuning",
    )


def _multi_maze_tuning_grid_from_full(tuning_full_d, tuning_res):
    out = {}
    for maze_key, template in tuning_res["tuning"].items():
        grid_shape = tuple(int(x) for x in tuning_res["grid_shape"][maze_key])
        grid_arr = np.asarray(tuning_full_d[maze_key]).reshape((*grid_shape, int(np.asarray(tuning_full_d[maze_key]).shape[1])))
        out[maze_key] = xr.DataArray(
            grid_arr,
            dims=template.dims,
            coords=template.coords,
            name="tuning",
        )
    return out


def _initial_tuning_full_from_supervised(tuning_res):
    if isinstance(tuning_res["tuning"], dict):
        out = {}
        for maze_key, da in tuning_res["tuning"].items():
            arr = np.asarray(da.values, dtype=float).reshape((-1, int(np.asarray(tuning_res["tuning_flat"]).shape[1])))
            arr = np.nan_to_num(arr, nan=0.0)
            out[maze_key] = arr
        return out
    arr = np.asarray(tuning_res["tuning"].values, dtype=float).reshape((-1, int(np.asarray(tuning_res["tuning_flat"]).shape[1])))
    arr = np.nan_to_num(arr, nan=0.0)
    return {"single": arr}


def _movement_variance_to_smooth_std_single(continuous_transition_movement_variance, n_dim):
    var = continuous_transition_movement_variance
    if isinstance(var, dict):
        if "single" in var:
            var = var["single"]
        else:
            var = list(var.values())[0]
    var_arr = np.asarray(np.atleast_1d(var), dtype=float)
    if var_arr.size == 1:
        return np.repeat(np.sqrt(var_arr[0]), n_dim)
    if var_arr.size != n_dim:
        raise ValueError(f'continuous_transition_movement_variance size={var_arr.size} != n_dim={n_dim}')
    return np.sqrt(var_arr)


def _build_full_transition_kernel_single(tuning_res, continuous_transition_movement_variance):
    bin_centers_l = tuning_res["bin_centers"]
    grid_shape = tuning_res["grid_shape"]
    n_dim = len(bin_centers_l)
    smooth_std = _movement_variance_to_smooth_std_single(continuous_transition_movement_variance, n_dim=n_dim)
    kernel = get_tuning_supervised.get_smoothing_matrix(
        bin_centers_l=bin_centers_l,
        grid_shape=grid_shape,
        smooth_std=smooth_std,
    )
    return _row_normalize(np.asarray(kernel, dtype=float))


def _build_full_flat_idx_to_coord_single(tuning_res):
    bin_centers_l = tuning_res["bin_centers"]
    label_dim_names = tuning_res["label_dim_names"]
    mesh = np.meshgrid(*bin_centers_l, indexing="ij")
    coord_cols = [m.reshape(-1) for m in mesh]
    df = pd.DataFrame({k: v for k, v in zip(label_dim_names, coord_cols)})
    df.insert(0, "maze", "single")
    df.index = np.arange(df.shape[0], dtype=int)
    df.index.name = "flat_idx"
    return df


def fit_decode_iterative_supervised_kde(
    spk_mat,
    label,
    n_iter=3,
    label_bin_size=1.0,
    smooth_std=None,
    occupancy_threshold=None,
    label_min=None,
    label_max=None,
    categorical_labels=None,
    dt=None,
    gain=1.0,
    continuous_transition_movement_variance=1.0,
    n_time_per_chunk=10000,
    stats_chunk_size=50000,
    eps=1e-12,
    verbose=True,
):
    """
    Iterative supervised KDE fit/decode loop with posterior-weighted updates.

    Inputs
    - spk_mat: (T, N) array or `nap.TsdFrame`
    - label: (T, D) array or `nap.TsdFrame` (or dict for multi-maze)
    - n_iter: number of posterior-weighted KDE refit iterations
    - label_bin_size, smooth_std, occupancy_threshold, label_min, label_max:
      forwarded to supervised tuning initialization
    - dt: bin size in seconds (used if array inputs without timestamps)
    - continuous_transition_movement_variance: forwarded to dynamics decoder
    - n_time_per_chunk: decode chunk size
    - stats_chunk_size: chunk size for posterior stat accumulation

    Returns dict (xarray-forward):
    - initial: initial tuning + initial decode (xarray)
    - final: final tuning + final decode (xarray)
    - diagnostics: iteration-level traces and weighted occupancy/spike stats
    """
    if dt is None:
        if hasattr(spk_mat, "t"):
            dt = float(np.median(np.diff(np.asarray(spk_mat.t))))
        else:
            dt = 1.0

    spk_tsdf, label_in = _ensure_supervised_inputs(spk_mat, label, dt=dt)
    spk_arr = np.asarray(spk_tsdf.d)
    time_coord = np.asarray(spk_tsdf.t)

    tuning_res = get_tuning_supervised.get_tuning(
        spk_tsdf,
        label_in,
        label_bin_size=label_bin_size,
        smooth_std=smooth_std,
        occupancy_threshold=occupancy_threshold,
        label_min=label_min,
        label_max=label_max,
        categorical_labels=categorical_labels,
        verbose=verbose,
    )

    full_grid_ops = _prepare_full_grid_iteration_ops(tuning_res)
    spk_j = jnp.asarray(spk_arr, dtype=jnp.float32)
    is_multi = isinstance(tuning_res["tuning"], dict)
    current_tuning_full = _initial_tuning_full_from_supervised(tuning_res)

    if verbose:
        if is_multi:
            n_latent_msg = int(np.asarray(tuning_res["tuning_flat"]).shape[0])
            print(f"[iterative_supervised_kde] init: n_time={spk_arr.shape[0]}, n_neuron={spk_arr.shape[1]}, n_latent_valid={n_latent_msg} (multi-maze valid-bin decode)")
        else:
            n_latent_msg = int(current_tuning_full["single"].shape[0])
            print(f"[iterative_supervised_kde] init: n_time={spk_arr.shape[0]}, n_neuron={spk_arr.shape[1]}, n_latent_full={n_latent_msg} (full-grid decode)")

    if is_multi:
        current_tuning_valid = jnp.asarray(tuning_res["tuning_flat"], dtype=jnp.float32)

        def _decode_once_valid(tuning_flat):
            return decoder_supervised.decode_with_dynamics(
                spk_arr,
                np.asarray(tuning_flat),
                coord_to_flat_idx=tuning_res["coord_to_flat_idx"],
                flat_idx_to_coord=None,
                dt=float(dt),
                gain=float(gain),
                continuous_transition_movement_variance=continuous_transition_movement_variance,
                p_move_to_jump=0.0,
                p_jump_to_move=0.0,
                n_time_per_chunk=int(n_time_per_chunk),
                observation_model="poisson",
            )

        decode_init = _decode_once_valid(current_tuning_valid)
        decode_curr = decode_init
    else:
        transition_full_kernel = _build_full_transition_kernel_single(
            tuning_res,
            continuous_transition_movement_variance=continuous_transition_movement_variance,
        )

        def _decode_once_full(tuning_full):
            return decoder_supervised.decode_with_dynamics(
                spk_arr,
                np.asarray(tuning_full),
                coord_to_flat_idx=None,
                flat_idx_to_coord=None,
                dt=float(dt),
                gain=float(gain),
                custom_continuous_transition_kernel=transition_full_kernel,
                p_move_to_jump=0.0,
                p_jump_to_move=0.0,
                n_time_per_chunk=int(n_time_per_chunk),
                observation_model="poisson",
            )

        decode_init = _decode_once_full(current_tuning_full["single"])
        decode_curr = decode_init

    log_marginal_history = [float(np.asarray(decode_init["log_marginal"]))]
    occupancy_history = []
    spk_count_history = []

    for iter_i in range(int(n_iter)):
        if is_multi:
            y_weighted, t_weighted = _posterior_weighted_stats_chunked(
                decode_curr["log_posterior_latent_marg"],
                spk_j,
                stats_chunk_size=stats_chunk_size,
            )
            updated_tuning, spk_count_smth, occupancy_smth, updated_tuning_full = _kde_update_on_full_grid(
                y_weighted,
                t_weighted,
                full_grid_ops=full_grid_ops,
                dt=dt,
                eps=eps,
            )
            current_tuning_valid = jnp.asarray(updated_tuning, dtype=jnp.float32)
            current_tuning_full = updated_tuning_full
            decode_curr = _decode_once_valid(current_tuning_valid)
        else:
            y_weighted, t_weighted = _posterior_weighted_stats_chunked(
                decode_curr["log_posterior_latent_marg"],
                spk_j,
                stats_chunk_size=stats_chunk_size,
            )
            occupancy_weighted = t_weighted * jnp.asarray(dt, dtype=jnp.float32)
            tuning_full_next, spk_count_smth, occupancy_smth = _smooth_ratio_update_jit(
                y_weighted,
                occupancy_weighted,
                jnp.asarray(transition_full_kernel, dtype=jnp.float32),
                jnp.asarray(eps, dtype=jnp.float32),
            )
            current_tuning_full["single"] = np.asarray(jax.device_get(tuning_full_next))
            decode_curr = _decode_once_full(current_tuning_full["single"])

        occupancy_history.append(np.asarray(occupancy_smth))
        spk_count_history.append(np.asarray(spk_count_smth))
        log_marginal_history.append(float(np.asarray(decode_curr["log_marginal"])))

        if verbose:
            print(f"[iterative_supervised_kde] iter={iter_i+1}/{n_iter}, log_marginal={log_marginal_history[-1]:.6f}")

    if is_multi:
        initial_tuning_flat = np.asarray(tuning_res["tuning_flat"])
        final_tuning_flat = np.asarray(jax.device_get(current_tuning_valid))
        flat_idx_to_coord_decode = tuning_res["flat_idx_to_coord"]
    else:
        valid_mask = np.asarray(tuning_res["valid_flat_mask"], dtype=bool)
        initial_tuning_flat = np.asarray(tuning_res["tuning_flat"])
        final_tuning_flat = np.asarray(current_tuning_full["single"])[valid_mask]
        flat_idx_to_coord_decode = _build_full_flat_idx_to_coord_single(tuning_res)

    initial_tuning_flat_xr = _make_tuning_flat_xr(initial_tuning_flat, tuning_res)
    final_tuning_flat_xr = _make_tuning_flat_xr(final_tuning_flat, tuning_res)

    if is_multi:
        final_tuning_grid_xr = _multi_maze_tuning_grid_from_full(current_tuning_full, tuning_res)
    else:
        final_tuning_grid_xr = _single_maze_tuning_grid_from_full(current_tuning_full["single"], tuning_res)

    decode_init_xr = _make_decode_xr(decode_init, flat_idx_to_coord_decode, time_coord=time_coord)
    decode_final_xr = _make_decode_xr(decode_curr, flat_idx_to_coord_decode, time_coord=time_coord)
    diag_label_coord = decode_final_xr["posterior_latent_marg_xr"].coords["label_bin"]
    n_diag_label = int(diag_label_coord.size)

    diagnostics = {
        "log_marginal_history_xr": xr.DataArray(
            np.asarray(log_marginal_history, dtype=float),
            dims=("iteration",),
            coords={"iteration": np.arange(len(log_marginal_history))},
            name="log_marginal",
        ),
        "occupancy_smth_history_xr": xr.DataArray(
            np.asarray(occupancy_history, dtype=float) if len(occupancy_history) else np.zeros((0, n_diag_label)),
            dims=("iteration", "label_bin"),
            coords={"iteration": np.arange(len(occupancy_history)), "label_bin": diag_label_coord},
            name="occupancy_smth",
        ),
        "spk_count_smth_history_xr": xr.DataArray(
            np.asarray(spk_count_history, dtype=float) if len(spk_count_history) else np.zeros((0, n_diag_label, final_tuning_flat.shape[1])),
            dims=("iteration", "label_bin", "neuron"),
            coords={
                "iteration": np.arange(len(spk_count_history)),
                "label_bin": diag_label_coord,
                "neuron": final_tuning_flat_xr.coords["neuron"],
            },
            name="spk_count_smth",
        ),
    }

    return {
        "initial": {
            "tuning_grid_xr": tuning_res["tuning"],
            "tuning_flat_xr": initial_tuning_flat_xr,
            "decode": decode_init_xr,
        },
        "final": {
            "tuning_grid_xr": final_tuning_grid_xr,
            "tuning_flat_xr": final_tuning_flat_xr,
            "decode": decode_final_xr,
        },
        "diagnostics": diagnostics,
        "config": {
            "n_iter": int(n_iter),
            "dt": float(dt),
            "gain": float(gain),
            "label_bin_size": label_bin_size,
            "smooth_std": smooth_std,
            "occupancy_threshold": occupancy_threshold,
            "continuous_transition_movement_variance": continuous_transition_movement_variance,
            "n_time_per_chunk": int(n_time_per_chunk),
            "stats_chunk_size": None if stats_chunk_size is None else int(stats_chunk_size),
            "eps": float(eps),
            "jit_enabled": True,
        },
    }

