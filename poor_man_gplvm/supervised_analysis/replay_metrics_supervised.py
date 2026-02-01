"""
Replay metrics for supervised replay analysis (ported from latent_reactivation HMM replay metrics).

This module computes LR-style replay metrics from:
- `label_posterior_marginal`: posterior over position/label bins, either as:
  - xr.DataArray with dims ('time','label_bin'), or
  - dict[str->xr.DataArray] keyed by maze name, each ('time','label_bin') (maze-split outputs), or
  - numpy array (n_time, n_label_bin) (single-maze / unlabeled).
- `dynamics_posterior_marginal`: posterior over dynamics states, as:
  - xr.DataArray with dims ('time','dyn') or ('time','state'), or
  - numpy array (n_time, n_dyn).

### Continuous segment definition (this is the key behavioral spec)
We define "continuous replay segments" (a.k.a. "move state segments") as:

1) **Probabilistic state threshold**:
   Mark time bins where:
     P(dynamics == `continuous_state_idx`) > `continuous_prob_thresh`
   Then find contiguous runs in this boolean mask; each run is a candidate segment.

2) **Minimum duration filter**:
   Discard segments shorter than `min_segment_duration` seconds.
   Uses binsize = `binsize` if provided, otherwise inferred from the `time` coord if available.

3) **Hard discontinuities (maze change)**:
   If `label_posterior_marginal` is maze-split (dict of mazes), we define the winning maze
   per time bin by **global MAP**:
     (maze*, label_bin*) = argmax over all mazes and bins of posterior(t, maze, bin)
   Any time `maze*` changes between adjacent time bins is treated as a "jump", and we split
   segments at those boundaries, even if dynamics suggests continuous.

4) **Hard discontinuities (large MAP steps)**:
   If you provide `position_key` so we can map bins to 1D or 2D coordinates, we compute a
   decoded position trajectory (MAP by default; set `use_posterior_weighted=True` to use E[x]).
   Any time the step size exceeds `stepsize_split_thresh` (in position units) we split the
   segments at that boundary. Default `stepsize_split_thresh=None` (do not split on steps).

After splitting, we re-apply the minimum duration filter.

### Metrics returned (LR-style keys; list-valued + scalar)
State-level:
- n_bins, binsize, duration_sec
- state_{i}_fraction: mean posterior of dynamics state i across time
- total_continuous_bins, frac_continuous

Segment-level (after all filtering/splitting):
- n_continuous_segments
- segment_durations (list[int], in bins)
- median_segment_duration (float, bins), max_segment_duration (int, bins)

Spatial trajectory metrics (only if `position_key` successfully maps label bins to coordinates):
- segment_path_lengths (list[float]) : sum of step distances in segment after discarding steps > thresh
- segment_max_displacements (list[float]) : max dist from start among "valid" positions
- segment_max_spans (list[float]) : 1D range or 2D max pairwise dist among "valid" positions
- segment_avg_speeds (list[float]) : segment_path_length / segment_duration_sec
- total_path_length, max_displacement, max_span
- median_segment_path_length, median_segment_max_span, median_segment_speed

Notes:
- "valid positions" follow LR semantics: position i is valid iff the step TO i was <= `stepsize_discard_thresh`.
  If `stepsize_discard_thresh=None` (default), all finite steps are treated as valid (no discarding).
- If `position_key` is provided but label-bin coordinate parsing fails, we print a warning by default and
  spatial metrics are returned as zeros/empty lists. Set `warn_on_position_key_fail=False` to silence,
  or `raise_on_position_key_fail=True` to error.
- If `position_key` is None or cannot be resolved from the label coordinates, spatial metrics are returned
  as empty lists and scalar spatial aggregates are 0.0.

Example:
```python
import poor_man_gplvm.supervised_analysis.replay_metrics_supervised as rms

# single-maze xarray
m = rms.compute_replay_metrics(label_post, dyn_post, position_key='lin', binsize=0.02)

# maze-split dict
m = rms.compute_replay_metrics(label_post_by_maze, dyn_post, position_key=('x','y'))

# per-event metrics from time-concat decode (e.g. tensor mode in decoder_supervised.decode_with_dynamics)
out = rms.compute_replay_metrics(
    res['posterior_latent_marg'],
    res['posterior_dynamics_marg'],
    starts=res['starts'],
    ends=res['ends'],
    binsize=0.02,
    position_key={'familiar': ('x','y'), 'novel': 'lin'},
)
metrics_df = out['metrics_df']
```
"""

import math
import numpy as np


def _as_numpy(a):
    if isinstance(a, np.ndarray):
        return a
    # xarray / pandas / jax arrays
    try:
        return np.asarray(a)
    except Exception:
        return np.array(a)


def _infer_binsize(time_vec, binsize):
    if binsize is not None:
        return float(binsize)
    if time_vec is None:
        return 1.0
    t = _as_numpy(time_vec).astype(float)
    if t.size <= 1:
        return 1.0
    dt = np.diff(t)
    dt = dt[np.isfinite(dt)]
    return float(np.median(dt)) if dt.size else 1.0


def _get_time_coord_from_xr(x):
    # minimal duck-typing to avoid hard dependency in callers
    if hasattr(x, "coords") and ("time" in getattr(x, "coords", {})):
        try:
            return _as_numpy(x.coords["time"].values)
        except Exception:
            return _as_numpy(x.coords["time"])
    if hasattr(x, "dims") and ("time" in getattr(x, "dims", ())):
        # fallback: try to index coord by dim name
        try:
            return _as_numpy(x["time"].values)
        except Exception:
            return None
    return None


def _get_dyn_array(dynamics_posterior_marginal):
    dyn = dynamics_posterior_marginal
    time_vec = None
    if hasattr(dyn, "dims"):
        # xarray DataArray
        time_vec = _get_time_coord_from_xr(dyn)
        dyn = _as_numpy(dyn)
    else:
        dyn = _as_numpy(dyn)
    if dyn.ndim != 2:
        raise ValueError(f"dynamics_posterior_marginal must be 2D (time, n_dyn). Got shape {dyn.shape}")
    return dyn, time_vec


def _find_runs_above_threshold(p, thresh):
    p = _as_numpy(p)
    above = p > float(thresh)
    diff = np.diff(np.concatenate([[0], above.astype(int), [0]]))
    starts = np.where(diff == 1)[0].astype(int)
    ends = (np.where(diff == -1)[0] - 1).astype(int)  # inclusive
    return [(int(s), int(e)) for s, e in zip(starts, ends)]


def _split_segments_on_breakpoints(segments, break_between_mask):
    """
    break_between_mask: bool array of shape (n_time-1,). True means break between t and t+1.
    """
    if segments is None:
        return []
    if break_between_mask is None:
        return list(segments)
    b = _as_numpy(break_between_mask).astype(bool)
    if b.size == 0:
        return list(segments)
    out = []
    for s, e in list(segments):
        s = int(s)
        e = int(e)
        if e <= s:
            out.append((s, e))
            continue
        # break indices inside [s, e-1] (because break_between is between i and i+1)
        bi = np.where(b[s:e])[0]  # indices relative to s
        cur_s = s
        for rel in bi:
            cut_end = s + int(rel)  # last index before break (inclusive)
            out.append((cur_s, cut_end))
            cur_s = cut_end + 1
        out.append((cur_s, e))
    return [(s, e) for s, e in out if s <= e]


def _get_label_map_trajectory_single(label_post):
    """
    label_post: (n_time, n_label_bin) array
    Returns:
      map_label_idx: (n_time,) int
      max_prob: (n_time,) float (posterior at MAP bin)
    """
    post = _as_numpy(label_post)
    if post.ndim != 2:
        raise ValueError(f"label_posterior_marginal must be 2D (time,label_bin). Got shape {post.shape}")
    map_idx = np.argmax(post, axis=1).astype(int)
    max_p = post[np.arange(post.shape[0]), map_idx]
    return map_idx, max_p


def _get_global_map_trajectory(label_post_by_maze):
    """
    label_post_by_maze: dict[maze]->(n_time,n_label_bin_maze)
    Returns:
      winning_maze: (n_time,) object array of maze keys
      map_label_idx_by_maze: dict[maze]->(n_time,) int (MAP idx within that maze)
      map_label_idx: (n_time,) int (MAP idx within winning maze)
    """
    maze_l = list(label_post_by_maze.keys())
    if len(maze_l) == 0:
        raise ValueError("label_posterior_marginal dict is empty.")

    # per maze MAP + value
    map_idx_by_maze = {}
    max_p_by_maze = {}
    n_time = None
    for maze in maze_l:
        arr = label_post_by_maze[maze]
        arr_np = _as_numpy(arr)
        if arr_np.ndim != 2:
            raise ValueError(f"label_posterior_marginal[{maze}] must be 2D. Got shape {arr_np.shape}")
        if n_time is None:
            n_time = int(arr_np.shape[0])
        if int(arr_np.shape[0]) != int(n_time):
            raise ValueError("All mazes must share the same time axis length.")
        mi, mp = _get_label_map_trajectory_single(arr_np)
        map_idx_by_maze[maze] = mi
        max_p_by_maze[maze] = mp

    # winner per time based on max prob at each maze's MAP
    max_p_stack = np.stack([max_p_by_maze[m] for m in maze_l], axis=1)  # (time, n_maze)
    win_maze_idx = np.argmax(max_p_stack, axis=1).astype(int)  # (time,)
    winning_maze = np.asarray([maze_l[i] for i in win_maze_idx], dtype=object)

    map_idx = np.zeros((int(n_time),), dtype=int)
    for mi, maze in enumerate(maze_l):
        mask = win_maze_idx == mi
        if np.any(mask):
            map_idx[mask] = map_idx_by_maze[maze][mask]
    return winning_maze, map_idx_by_maze, map_idx


def _extract_positions_from_label_coord(label_coord, position_key):
    """
    label_coord: expected to be a pandas.MultiIndex (from decoder_supervised wrappers)
    position_key: str (1D) or tuple/list of 2 str (2D)
    Returns:
      is_2d (bool), pos_per_bin (np.ndarray (n_bin,) or (n_bin,2))
    """
    if label_coord is None or position_key is None:
        return False, None

    # duck-type pandas.MultiIndex to avoid a hard pandas import here
    if not (hasattr(label_coord, "get_level_values") and hasattr(label_coord, "names")):
        return False, None

    if isinstance(position_key, (list, tuple)) and len(position_key) == 2:
        kx, ky = position_key[0], position_key[1]
        if (kx not in label_coord.names) or (ky not in label_coord.names):
            return True, None
        x = _as_numpy(label_coord.get_level_values(kx)).astype(float)
        y = _as_numpy(label_coord.get_level_values(ky)).astype(float)
        pos = np.stack([x, y], axis=1)
        return True, pos

    # 1D
    k = position_key
    if k not in label_coord.names:
        return False, None
    x = _as_numpy(label_coord.get_level_values(k)).astype(float)
    return False, x


def _extract_positions_from_post(post, position_key):
    """
    Prefer reading coordinate arrays directly from xarray coords (e.g. coords['x'], coords['y'] on label_bin),
    and fall back to MultiIndex level extraction if needed.

    Returns:
      is_2d (bool), pos_per_bin (np.ndarray) or None
    """
    if post is None or position_key is None:
        return False, None

    # 1) xarray-style: coords live on label_bin (your screenshot: coords include x,y as (label_bin,))
    if hasattr(post, "coords"):
        coords = post.coords
        if isinstance(position_key, (list, tuple)) and len(position_key) == 2:
            kx, ky = position_key[0], position_key[1]
            if (kx in coords) and (ky in coords):
                try:
                    x = _as_numpy(coords[kx].values).astype(float)
                except Exception:
                    x = _as_numpy(coords[kx]).astype(float)
                try:
                    y = _as_numpy(coords[ky].values).astype(float)
                except Exception:
                    y = _as_numpy(coords[ky]).astype(float)
                if x.ndim == 1 and y.ndim == 1 and x.shape[0] == y.shape[0]:
                    return True, np.stack([x, y], axis=1)
        else:
            k = position_key
            if k in coords:
                try:
                    x = _as_numpy(coords[k].values).astype(float)
                except Exception:
                    x = _as_numpy(coords[k]).astype(float)
                if x.ndim == 1:
                    return False, x

    # 2) fallback: MultiIndex levels on label_bin coord
    label_coord = _get_label_coord_from_post(post)
    return _extract_positions_from_label_coord(label_coord, position_key)


def _xr_map_positions_from_post(post, position_key):
    """
    Xarray-native MAP coordinate extraction:

        post.idxmax(dim='label_bin')[coord_name]

    This matches the user's suggested workflow and supports the common case where `x`/`y` are
    separate coords defined on the `label_bin` dimension (not necessarily MultiIndex levels).

    Returns:
      is_2d (bool), pos_traj (np.ndarray (n_time,) or tuple(x,y)) or (False,None) on failure.
    """
    if post is None or position_key is None:
        return False, None
    if not (hasattr(post, "dims") and hasattr(post, "idxmax")):
        return False, None
    if "label_bin" not in getattr(post, "dims", ()):
        return False, None
    try:
        idx = post.idxmax(dim="label_bin")
    except Exception:
        return False, None

    if isinstance(position_key, (list, tuple)) and len(position_key) == 2:
        kx, ky = position_key[0], position_key[1]
        try:
            x = _as_numpy(idx[kx].values).astype(float)
            y = _as_numpy(idx[ky].values).astype(float)
            return True, (x, y)
        except Exception:
            return True, None

    k = position_key
    try:
        x = _as_numpy(idx[k].values).astype(float)
        return False, x
    except Exception:
        return False, None


def _xr_expected_positions_from_post(post, position_key):
    """
    Xarray-native posterior-weighted position:
      E[x] = sum_{label_bin} post * coord(label_bin)
    Returns (is_2d, pos_traj) or (False,None) on failure.
    """
    if post is None or position_key is None:
        return False, None
    if not (hasattr(post, "dims") and hasattr(post, "coords") and ("label_bin" in getattr(post, "dims", ()))):
        return False, None
    try:
        if isinstance(position_key, (list, tuple)) and len(position_key) == 2:
            kx, ky = position_key[0], position_key[1]
            if (kx not in post.coords) or (ky not in post.coords):
                return True, None
            ex = (post * post.coords[kx]).sum(dim="label_bin")
            ey = (post * post.coords[ky]).sum(dim="label_bin")
            return True, (_as_numpy(ex.values).astype(float), _as_numpy(ey.values).astype(float))
        k = position_key
        if k not in post.coords:
            return False, None
        ex = (post * post.coords[k]).sum(dim="label_bin")
        return False, _as_numpy(ex.values).astype(float)
    except Exception:
        if isinstance(position_key, (list, tuple)) and len(position_key) == 2:
            return True, None
        return False, None

def _position_key_fail_reason(label_coord, position_key):
    if label_coord is None:
        return "label_bin coord missing"
    if not (hasattr(label_coord, "get_level_values") and hasattr(label_coord, "names")):
        return f"label_bin coord not MultiIndex-like (type={type(label_coord)})"

    names = list(label_coord.names) if label_coord.names is not None else []
    if isinstance(position_key, (list, tuple)) and len(position_key) == 2:
        missing = [k for k in position_key if k not in names]
        if missing:
            return f"position_key={position_key} missing levels={missing} (available={names})"
        return None

    if position_key not in names:
        return f"position_key={position_key} missing level (available={names})"
    return None


def _position_key_fail_reason_from_post(post, position_key):
    if post is None:
        return "post is None"
    if position_key is None:
        return "position_key is None"

    if hasattr(post, "coords"):
        coords = post.coords
        if isinstance(position_key, (list, tuple)) and len(position_key) == 2:
            kx, ky = position_key[0], position_key[1]
            missing = [k for k in (kx, ky) if k not in coords]
            if not missing:
                return None
            # don't spam huge coords repr
            avail = list(coords.keys())
            return f"missing coords {missing} (available coord keys include {avail[:10]})"
        else:
            if position_key in coords:
                return None
            avail = list(coords.keys())
            return f"missing coord '{position_key}' (available coord keys include {avail[:10]})"

    label_coord = _get_label_coord_from_post(post)
    return _position_key_fail_reason(label_coord, position_key)


def _compute_segment_spatial_metrics(pos_traj, segments, *, binsize, stepsize_discard_thresh, is_2d):
    if pos_traj is None:
        return {
            "segment_path_lengths": [],
            "segment_max_displacements": [],
            "segment_max_spans": [],
            "segment_avg_speeds": [],
            "total_path_length": 0.0,
            "max_displacement": 0.0,
            "max_span": 0.0,
            "median_segment_path_length": 0.0,
            "median_segment_max_span": 0.0,
            "median_segment_speed": 0.0,
        }

    stepsize_discard_thresh = float(stepsize_discard_thresh)

    seg_path = []
    seg_max_disp = []
    seg_max_span = []
    seg_speed = []

    for s, e in segments:
        s = int(s)
        e = int(e)
        n = e - s + 1
        if n < 2:
            seg_path.append(0.0)
            seg_max_disp.append(0.0)
            seg_max_span.append(0.0)
            seg_speed.append(0.0)
            continue

        if is_2d:
            x_seg = _as_numpy(pos_traj[0][s : e + 1]).astype(float)
            y_seg = _as_numpy(pos_traj[1][s : e + 1]).astype(float)
            dx = np.diff(x_seg)
            dy = np.diff(y_seg)
            step_d = np.sqrt(dx ** 2 + dy ** 2)
        else:
            x_seg = _as_numpy(pos_traj[s : e + 1]).astype(float)
            step_d = np.abs(np.diff(x_seg))

        valid_step = (step_d <= stepsize_discard_thresh) & np.isfinite(step_d)
        valid_steps = step_d[valid_step]
        path_len = float(np.sum(valid_steps)) if valid_steps.size else 0.0

        valid_pos = np.ones((n,), dtype=bool)
        valid_pos[1:] = valid_step

        if is_2d:
            x_valid = x_seg[valid_pos]
            y_valid = y_seg[valid_pos]
            if x_valid.size:
                dist_from_start = np.sqrt((x_valid - x_seg[0]) ** 2 + (y_valid - y_seg[0]) ** 2)
                max_disp = float(np.max(dist_from_start))
            else:
                max_disp = 0.0

            if x_valid.size >= 2:
                dxp = x_valid[:, None] - x_valid[None, :]
                dyp = y_valid[:, None] - y_valid[None, :]
                pair = np.sqrt(dxp ** 2 + dyp ** 2)
                max_span = float(np.max(pair))
            else:
                max_span = 0.0
        else:
            x_valid = x_seg[valid_pos]
            if x_valid.size:
                max_disp = float(np.max(np.abs(x_valid - x_seg[0])))
            else:
                max_disp = 0.0

            if x_valid.size >= 2:
                max_span = float(np.max(x_valid) - np.min(x_valid))
            else:
                max_span = 0.0

        dur_sec = float(n) * float(binsize)
        avg_speed = float(path_len / dur_sec) if dur_sec > 0 else 0.0

        seg_path.append(path_len)
        seg_max_disp.append(max_disp)
        seg_max_span.append(max_span)
        seg_speed.append(avg_speed)

    total_path = float(np.sum(seg_path)) if len(seg_path) else 0.0
    max_disp_all = float(np.max(seg_max_disp)) if len(seg_max_disp) else 0.0
    max_span_all = float(np.max(seg_max_span)) if len(seg_max_span) else 0.0

    return {
        "segment_path_lengths": seg_path,
        "segment_max_displacements": seg_max_disp,
        "segment_max_spans": seg_max_span,
        "segment_avg_speeds": seg_speed,
        "total_path_length": total_path,
        "max_displacement": max_disp_all,
        "max_span": max_span_all,
        "median_segment_path_length": float(np.median(seg_path)) if len(seg_path) else 0.0,
        "median_segment_max_span": float(np.median(seg_max_span)) if len(seg_max_span) else 0.0,
        "median_segment_speed": float(np.median(seg_speed)) if len(seg_speed) else 0.0,
    }


def _slice_time(obj, s, e):
    """
    Slice time dimension for numpy arrays or xarray DataArray-like objects.
    Uses half-open [s:e).
    """
    if obj is None:
        return None
    if hasattr(obj, "isel"):
        try:
            return obj.isel(time=slice(int(s), int(e)))
        except Exception:
            pass
    arr = _as_numpy(obj)
    return arr[int(s) : int(e)]


def _slice_label(label_posterior_marginal, s, e):
    if isinstance(label_posterior_marginal, dict):
        return {k: _slice_time(v, s, e) for k, v in label_posterior_marginal.items()}
    return _slice_time(label_posterior_marginal, s, e)


def _position_key_for_maze(position_key, maze):
    if isinstance(position_key, dict):
        pk = position_key.get(maze, None)
        if pk is not None:
            return pk
        if maze is not None:
            return position_key.get(str(maze), None)
        return None
    return position_key


def _get_label_coord_from_post(post):
    if hasattr(post, "coords") and ("label_bin" in post.coords):
        coord = post.coords["label_bin"]
        # xarray stores pandas.MultiIndex via a special coordinate; `.values` may drop to ndarray of tuples.
        # Prefer `.to_index()` when available to preserve MultiIndex and support get_level_values().
        try:
            if hasattr(coord, "to_index"):
                return coord.to_index()
        except Exception:
            pass
        try:
            return coord.values
        except Exception:
            return coord
    return None


def _segment_break_between_from_pos(pos_seg, thr, is_2d):
    """
    pos_seg: 1D array (n,) or tuple (x_seg,y_seg), each (n,)
    Returns break_between_mask: (n-1,) bool; True means break between i and i+1.
    """
    if thr is None:
        return None
    thr = float(thr)
    if not np.isfinite(thr):
        return None
    if is_2d:
        x = _as_numpy(pos_seg[0]).astype(float)
        y = _as_numpy(pos_seg[1]).astype(float)
        dx = np.diff(x)
        dy = np.diff(y)
        step_d = np.sqrt(dx**2 + dy**2)
    else:
        x = _as_numpy(pos_seg).astype(float)
        step_d = np.abs(np.diff(x))
    # LR semantics: break on any step > thr and also on NaNs (comparison False => break True after negation)
    return ~(step_d <= thr)


def _segment_spatial_metrics_from_pos(pos_seg, *, binsize, stepsize_discard_thresh, is_2d):
    """
    Compute LR-style spatial metrics for one segment given decoded positions (already segment-sliced).
    """
    n = int(_as_numpy(pos_seg[0]).shape[0]) if is_2d else int(_as_numpy(pos_seg).shape[0])
    if n < 2:
        return 0.0, 0.0, 0.0, 0.0

    thr = stepsize_discard_thresh
    if thr is None:
        thr = np.inf
    thr = float(thr)
    if is_2d:
        x_seg = _as_numpy(pos_seg[0]).astype(float)
        y_seg = _as_numpy(pos_seg[1]).astype(float)
        dx = np.diff(x_seg)
        dy = np.diff(y_seg)
        step_d = np.sqrt(dx**2 + dy**2)
    else:
        x_seg = _as_numpy(pos_seg).astype(float)
        step_d = np.abs(np.diff(x_seg))

    valid_step = (step_d <= thr) & np.isfinite(step_d)
    valid_steps = step_d[valid_step]
    path_len = float(np.sum(valid_steps)) if valid_steps.size else 0.0

    if np.isfinite(thr):
        valid_pos = np.ones((n,), dtype=bool)
        valid_pos[1:] = valid_step
    else:
        valid_pos = np.ones((n,), dtype=bool)

    if is_2d:
        x_valid = x_seg[valid_pos]
        y_valid = y_seg[valid_pos]
        if x_valid.size:
            dist_from_start = np.sqrt((x_valid - x_seg[0]) ** 2 + (y_valid - y_seg[0]) ** 2)
            max_disp = float(np.max(dist_from_start))
        else:
            max_disp = 0.0

        if x_valid.size >= 2:
            dxp = x_valid[:, None] - x_valid[None, :]
            dyp = y_valid[:, None] - y_valid[None, :]
            pair = np.sqrt(dxp**2 + dyp**2)
            max_span = float(np.max(pair))
        else:
            max_span = 0.0
    else:
        x_valid = x_seg[valid_pos]
        if x_valid.size:
            max_disp = float(np.max(np.abs(x_valid - x_seg[0])))
        else:
            max_disp = 0.0

        if x_valid.size >= 2:
            max_span = float(np.max(x_valid) - np.min(x_valid))
        else:
            max_span = 0.0

    dur_sec = float(n) * float(binsize)
    avg_speed = float(path_len / dur_sec) if dur_sec > 0 else 0.0
    return path_len, max_disp, max_span, avg_speed


def _compute_replay_metrics_single(
    label_posterior_marginal,
    dynamics_posterior_marginal,
    *,
    continuous_prob_thresh=0.8,
    continuous_state_idx=0,
    binsize=None,
    min_segment_duration=0.06,
    stepsize_discard_thresh=None,
    stepsize_split_thresh=None,
    position_key=None,
    use_posterior_weighted=False,
    warn_on_position_key_fail=True,
    raise_on_position_key_fail=False,
):
    """
    Single-event metrics (internal). Returns one dict of metrics.
    """
    dyn, time_dyn = _get_dyn_array(dynamics_posterior_marginal)
    n_time = int(dyn.shape[0])
    n_dyn = int(dyn.shape[1])

    # time coordinate preference: take from label if available (xarray), else from dynamics
    time_vec = None
    if isinstance(label_posterior_marginal, dict):
        maze0 = next(iter(label_posterior_marginal.keys()))
        time_vec = _get_time_coord_from_xr(label_posterior_marginal[maze0])
    else:
        time_vec = _get_time_coord_from_xr(label_posterior_marginal) if hasattr(label_posterior_marginal, "dims") else None
    if time_vec is None:
        time_vec = time_dyn
    binsize = _infer_binsize(time_vec, binsize=binsize)

    # dynamics state fractions (posterior weighted)
    state_fraction_dict = {}
    for i in range(n_dyn):
        state_fraction_dict[f"state_{i}_fraction"] = float(np.mean(dyn[:, i])) if n_time else 0.0

    if not (0 <= int(continuous_state_idx) < n_dyn):
        raise ValueError(f"continuous_state_idx={continuous_state_idx} out of range for n_dyn={n_dyn}")

    # -------------------------------------------------------------------------
    # Find continuous segments (threshold + min duration)
    # -------------------------------------------------------------------------
    p_cont = dyn[:, int(continuous_state_idx)]
    segments_all = _find_runs_above_threshold(p_cont, continuous_prob_thresh)
    min_bins = int(math.ceil(float(min_segment_duration) / float(binsize))) if float(min_segment_duration) > 0 else 0
    segments = [(s, e) for s, e in segments_all if (e - s + 1) >= min_bins]

    # -------------------------------------------------------------------------
    # MAP label trajectory (+ winning maze per time for multi-maze)
    # -------------------------------------------------------------------------
    winning_maze = None
    map_label_idx = None
    map_label_idx_by_maze = None

    if isinstance(label_posterior_marginal, dict):
        winning_maze, map_label_idx_by_maze, map_label_idx = _get_global_map_trajectory(label_posterior_marginal)
        if int(winning_maze.shape[0]) != n_time:
            raise ValueError("label and dynamics posteriors must share time axis length.")
    else:
        label_np = _as_numpy(label_posterior_marginal)
        if label_np.ndim != 2:
            raise ValueError(f"label_posterior_marginal must be 2D (time,label_bin). Got shape {label_np.shape}")
        if int(label_np.shape[0]) != n_time:
            raise ValueError("label and dynamics posteriors must share time axis length.")
        map_label_idx, _ = _get_label_map_trajectory_single(label_np)

    # split on maze change first (hard discontinuity)
    if winning_maze is not None and winning_maze.size >= 2:
        maze_break_between = winning_maze[1:] != winning_maze[:-1]
        segments = _split_segments_on_breakpoints(segments, maze_break_between)
        segments = [(s, e) for s, e in segments if (e - s + 1) >= min_bins]

    # -------------------------------------------------------------------------
    # Split segments on large steps (done per-segment so mixed 1D/2D per maze works)
    # -------------------------------------------------------------------------
    has_any_pos = False
    pos_fail_reasons = []
    seg_split = []
    for s, e in segments:
        s = int(s)
        e = int(e)
        if e <= s:
            seg_split.append((s, e))
            continue

        maze = None
        if winning_maze is not None:
            maze = winning_maze[s]

        # resolve coordinate map for this maze
        pk = _position_key_for_maze(position_key, maze)
        if pk is None:
            seg_split.append((s, e))
            continue

        if isinstance(label_posterior_marginal, dict):
            post_maze = label_posterior_marginal.get(maze, None)
            if post_maze is None and maze is not None:
                post_maze = label_posterior_marginal.get(str(maze), None)
            if post_maze is None:
                seg_split.append((s, e))
                continue
        else:
            post_maze = label_posterior_marginal

        is_2d, pos_per_bin = _extract_positions_from_post(post_maze, pk)
        if pos_per_bin is None:
            reason = _position_key_fail_reason_from_post(post_maze, pk)
            if reason is None:
                reason = "unknown coord parsing failure"
            pos_fail_reasons.append(f"maze={maze} {reason}")
            seg_split.append((s, e))
            continue

        # build decoded pos for this segment (within a single maze)
        if use_posterior_weighted:
            post_seg = _slice_time(post_maze, s, e + 1)
            is2, pos_seg = _xr_expected_positions_from_post(post_seg, pk)
            if pos_seg is None:
                # fallback to numpy
                w = _as_numpy(post_seg)
                if is_2d:
                    x_seg = np.sum(w * pos_per_bin[:, 0][None, :], axis=1)
                    y_seg = np.sum(w * pos_per_bin[:, 1][None, :], axis=1)
                    pos_seg = (x_seg, y_seg)
                else:
                    pos_seg = np.sum(w * pos_per_bin[None, :], axis=1)
        else:
            post_seg = _slice_time(post_maze, s, e + 1)
            is2, pos_seg = _xr_map_positions_from_post(post_seg, pk)
            if pos_seg is None:
                # fallback to numpy argmax -> coord indexing
                if isinstance(label_posterior_marginal, dict):
                    idx = _as_numpy(map_label_idx_by_maze[maze][s : e + 1]).astype(int)
                else:
                    idx = _as_numpy(map_label_idx[s : e + 1]).astype(int)
                if is_2d:
                    pos_seg = (pos_per_bin[idx, 0], pos_per_bin[idx, 1])
                else:
                    pos_seg = pos_per_bin[idx]

        has_any_pos = True
        break_between = _segment_break_between_from_pos(pos_seg, stepsize_split_thresh, is_2d=is_2d)
        if break_between is None:
            seg_split.append((s, e))
            continue

        # split local indices then shift back
        local = _split_segments_on_breakpoints([(0, e - s)], break_between)
        for ls, le in local:
            seg_split.append((s + ls, s + le))

    segments = [(s, e) for s, e in seg_split if (e - s + 1) >= min_bins]

    # -------------------------------------------------------------------------
    # Aggregate segment metrics (LR-style)
    # -------------------------------------------------------------------------
    segment_durations = [int(e - s + 1) for s, e in segments]
    total_cont_bins = int(np.sum(segment_durations)) if len(segment_durations) else 0

    result = {
        "n_bins": int(n_time),
        "n_continuous_segments": int(len(segments)),
        "segment_durations": segment_durations,
        "median_segment_duration": float(np.median(segment_durations)) if len(segment_durations) else 0.0,
        "max_segment_duration": int(np.max(segment_durations)) if len(segment_durations) else 0,
        "total_continuous_bins": int(total_cont_bins),
        "frac_continuous": float(total_cont_bins / n_time) if n_time else 0.0,
        "duration_sec": float(n_time) * float(binsize),
        "binsize": float(binsize),
    }
    result.update(state_fraction_dict)

    # spatial/path metrics: compute per-segment (handles mixed 1D/2D across mazes)
    seg_path = []
    seg_max_disp = []
    seg_max_span = []
    seg_speed = []

    if position_key is not None and has_any_pos and len(segments):
        for s, e in segments:
            s = int(s)
            e = int(e)
            maze = None
            if winning_maze is not None:
                maze = winning_maze[s]

            pk = _position_key_for_maze(position_key, maze)
            if pk is None:
                seg_path.append(0.0)
                seg_max_disp.append(0.0)
                seg_max_span.append(0.0)
                seg_speed.append(0.0)
                continue

            if isinstance(label_posterior_marginal, dict):
                post_maze = label_posterior_marginal.get(maze, None)
                if post_maze is None and maze is not None:
                    post_maze = label_posterior_marginal.get(str(maze), None)
                if post_maze is None:
                    seg_path.append(0.0)
                    seg_max_disp.append(0.0)
                    seg_max_span.append(0.0)
                    seg_speed.append(0.0)
                    continue
            else:
                post_maze = label_posterior_marginal

            is_2d, pos_per_bin = _extract_positions_from_post(post_maze, pk)
            if pos_per_bin is None:
                reason = _position_key_fail_reason_from_post(post_maze, pk)
                if reason is None:
                    reason = "unknown coord parsing failure"
                pos_fail_reasons.append(f"maze={maze} {reason}")
                seg_path.append(0.0)
                seg_max_disp.append(0.0)
                seg_max_span.append(0.0)
                seg_speed.append(0.0)
                continue

            if use_posterior_weighted:
                post_seg = _slice_time(post_maze, s, e + 1)
                is2, pos_seg = _xr_expected_positions_from_post(post_seg, pk)
                if pos_seg is None:
                    w = _as_numpy(post_seg)
                    if is_2d:
                        x_seg = np.sum(w * pos_per_bin[:, 0][None, :], axis=1)
                        y_seg = np.sum(w * pos_per_bin[:, 1][None, :], axis=1)
                        pos_seg = (x_seg, y_seg)
                    else:
                        pos_seg = np.sum(w * pos_per_bin[None, :], axis=1)
            else:
                post_seg = _slice_time(post_maze, s, e + 1)
                is2, pos_seg = _xr_map_positions_from_post(post_seg, pk)
                if pos_seg is None:
                    if isinstance(label_posterior_marginal, dict):
                        idx = _as_numpy(map_label_idx_by_maze[maze][s : e + 1]).astype(int)
                    else:
                        idx = _as_numpy(map_label_idx[s : e + 1]).astype(int)
                    if is_2d:
                        pos_seg = (pos_per_bin[idx, 0], pos_per_bin[idx, 1])
                    else:
                        pos_seg = pos_per_bin[idx]

            path_len, max_disp, max_span, avg_speed = _segment_spatial_metrics_from_pos(
                pos_seg,
                binsize=binsize,
                stepsize_discard_thresh=stepsize_discard_thresh,
                is_2d=is_2d,
            )
            seg_path.append(float(path_len))
            seg_max_disp.append(float(max_disp))
            seg_max_span.append(float(max_span))
            seg_speed.append(float(avg_speed))

    result["segment_path_lengths"] = seg_path if len(seg_path) else []
    result["segment_max_displacements"] = seg_max_disp if len(seg_max_disp) else []
    result["segment_max_spans"] = seg_max_span if len(seg_max_span) else []
    result["segment_avg_speeds"] = seg_speed if len(seg_speed) else []

    if len(seg_path):
        result["total_path_length"] = float(np.sum(seg_path))
        result["max_displacement"] = float(np.max(seg_max_disp)) if len(seg_max_disp) else 0.0
        result["max_span"] = float(np.max(seg_max_span)) if len(seg_max_span) else 0.0
        result["median_segment_path_length"] = float(np.median(seg_path))
        result["median_segment_max_span"] = float(np.median(seg_max_span)) if len(seg_max_span) else 0.0
        result["median_segment_speed"] = float(np.median(seg_speed)) if len(seg_speed) else 0.0
    else:
        result["total_path_length"] = 0.0
        result["max_displacement"] = 0.0
        result["max_span"] = 0.0
        result["median_segment_path_length"] = 0.0
        result["median_segment_max_span"] = 0.0
        result["median_segment_speed"] = 0.0

    # warn/error if user asked for coordinates but we couldn't use them (anywhere or partially)
    if position_key is not None and (len(pos_fail_reasons) or (not has_any_pos)):
        msg = "[replay_metrics_supervised] position_key coord extraction issue; some/all spatial metrics may be 0. "
        if len(pos_fail_reasons):
            uniq = []
            for r in pos_fail_reasons:
                if r not in uniq:
                    uniq.append(r)
                if len(uniq) >= 8:
                    break
            msg += "Reasons (up to 8): " + "; ".join(uniq)
        if bool(raise_on_position_key_fail):
            raise ValueError(msg)
        if bool(warn_on_position_key_fail):
            print(msg)

    return result


def compute_replay_metrics(
    label_posterior_marginal,
    dynamics_posterior_marginal,
    *,
    continuous_prob_thresh=0.8,
    continuous_state_idx=0,
    binsize=None,
    min_segment_duration=0.06,
    stepsize_discard_thresh=None,
    stepsize_split_thresh=None,
    position_key=None,
    use_posterior_weighted=False,
    starts=None,
    ends=None,
    warn_on_position_key_fail=True,
    raise_on_position_key_fail=False,
):
    """
    Compute LR-style replay metrics from supervised posteriors.

    See module docstring for definitions (especially continuous segment definition).
    Returns:
    - if `starts/ends` are None and inputs are not lists: a single metrics dict
    - if `starts/ends` are provided OR inputs are lists: a dict with:
        - 'metrics_df': pd.DataFrame of scalar metrics (one row per event)
        - 'metrics': list of per-event metrics dicts (includes list-valued fields)
        - 'summary': pooled summary dict (median-style; minimal)
    """
    # list inputs => per-event
    if isinstance(label_posterior_marginal, list) or isinstance(dynamics_posterior_marginal, list):
        label_l = label_posterior_marginal if isinstance(label_posterior_marginal, list) else [label_posterior_marginal]
        dyn_l = dynamics_posterior_marginal if isinstance(dynamics_posterior_marginal, list) else [dynamics_posterior_marginal]
        n = min(len(label_l), len(dyn_l))
        metrics = []
        for i in range(n):
            m = _compute_replay_metrics_single(
                label_l[i],
                dyn_l[i],
                continuous_prob_thresh=continuous_prob_thresh,
                continuous_state_idx=continuous_state_idx,
                binsize=binsize,
                min_segment_duration=min_segment_duration,
                stepsize_discard_thresh=stepsize_discard_thresh,
                stepsize_split_thresh=stepsize_split_thresh,
                position_key=position_key,
                use_posterior_weighted=use_posterior_weighted,
                warn_on_position_key_fail=warn_on_position_key_fail,
                raise_on_position_key_fail=raise_on_position_key_fail,
            )
            m["event_i"] = int(i)
            metrics.append(m)
        return _metrics_list_to_df_and_summary(metrics)

    # starts/ends => slice a time-concat decode into per-event metrics
    if starts is not None and ends is not None:
        starts = _as_numpy(starts).astype(int)
        ends = _as_numpy(ends).astype(int)
        metrics = []
        for i, (s, e) in enumerate(zip(starts, ends)):
            lab_i = _slice_label(label_posterior_marginal, s, e)
            dyn_i = _slice_time(dynamics_posterior_marginal, s, e)
            m = _compute_replay_metrics_single(
                lab_i,
                dyn_i,
                continuous_prob_thresh=continuous_prob_thresh,
                continuous_state_idx=continuous_state_idx,
                binsize=binsize,
                min_segment_duration=min_segment_duration,
                stepsize_discard_thresh=stepsize_discard_thresh,
                stepsize_split_thresh=stepsize_split_thresh,
                position_key=position_key,
                use_posterior_weighted=use_posterior_weighted,
                warn_on_position_key_fail=warn_on_position_key_fail,
                raise_on_position_key_fail=raise_on_position_key_fail,
            )
            m["event_i"] = int(i)
            metrics.append(m)
        return _metrics_list_to_df_and_summary(metrics)

    return _compute_replay_metrics_single(
        label_posterior_marginal,
        dynamics_posterior_marginal,
        continuous_prob_thresh=continuous_prob_thresh,
        continuous_state_idx=continuous_state_idx,
        binsize=binsize,
        min_segment_duration=min_segment_duration,
        stepsize_discard_thresh=stepsize_discard_thresh,
        stepsize_split_thresh=stepsize_split_thresh,
        position_key=position_key,
        use_posterior_weighted=use_posterior_weighted,
        warn_on_position_key_fail=warn_on_position_key_fail,
        raise_on_position_key_fail=raise_on_position_key_fail,
    )


def _metrics_list_to_df_and_summary(metrics):
    """
    Convert list[dict] metrics to (metrics_df, summary) similar to LR compute_intervals_metrics.
    """
    try:
        import pandas as pd
    except Exception:
        pd = None

    list_cols = {
        "segment_durations",
        "segment_path_lengths",
        "segment_max_displacements",
        "segment_max_spans",
        "segment_avg_speeds",
    }

    if pd is not None:
        scalar_metrics = [{k: v for k, v in m.items() if k not in list_cols} for m in metrics]
        metrics_df = pd.DataFrame(scalar_metrics)
    else:
        metrics_df = None

    if len(metrics):
        summary = {
            "n_events": int(len(metrics)),
            "total_continuous_segments": int(np.sum([m.get("n_continuous_segments", 0) for m in metrics])),
            "median_frac_continuous": float(np.median([m.get("frac_continuous", 0.0) for m in metrics])),
            "median_total_path_length": float(np.median([m.get("total_path_length", 0.0) for m in metrics])),
        }
    else:
        summary = {"n_events": 0}

    return {
        "metrics_df": metrics_df,
        "metrics": metrics,
        "summary": summary,
    }