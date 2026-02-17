import numpy as np
import pynapple as nap
import pandas as pd
import scipy.spatial.distance as scipy_distance
import scipy.linalg
import matplotlib.pyplot as plt



def get_ang_vel_tsd(position_tsdf: nap.TsdFrame, *, smooth_win: float = 0.1, min_speed: float = 1e-6) -> nap.Tsd:
    """
    Angular velocity (rad/s) from 2D position using kinematics:
        omega = (v_x a_y - v_y a_x) / (v_x^2 + v_y^2)
    This avoids atan2 unwrap artifacts; still unstable when speed is ~0 (handled by min_speed -> NaN).
    """
    vel_tsdf = position_tsdf[["x", "y"]].derivative().smooth(smooth_win)
    acc_tsdf = vel_tsdf.derivative().smooth(smooth_win)

    vx = vel_tsdf["x"].d
    vy = vel_tsdf["y"].d
    ax = acc_tsdf["x"].d
    ay = acc_tsdf["y"].d
    denom = vx * vx + vy * vy
    omega = (vx * ay - vy * ax) / denom
    omega[denom <= (min_speed * min_speed)] = np.nan

    ang_vel_tsd = nap.Tsd(t=vel_tsdf.t, d=omega, time_support=vel_tsdf.time_support)
    return ang_vel_tsd

def get_dist_to_maze(position_tsdf, method='roman_tmaze_projected', *, lin_round=0.1):
    # position_tsdf: nap.TsdFrame with columns ['x','y'] (and optionally 'lin' if method='from_lin')
    if method == 'from_lin':
        maze_xy_sampled = get_maze_xy_sampled(position_tsdf[["x", "y", "lin"]], method=method, lin_round=lin_round)
    else:
        maze_xy_sampled = get_maze_xy_sampled(position_tsdf[["x", "y"]], method=method, lin_round=lin_round)

    xy = position_tsdf[["x", "y"]].to_numpy()
    dist_to_maze = np.min(scipy_distance.cdist(xy, maze_xy_sampled), axis=1)

    dist_to_maze_tsd = nap.Tsd(t=position_tsdf.t, d=dist_to_maze)
    return dist_to_maze_tsd

def detect_offmaze_ep(
    position_tsdf: nap.TsdFrame,
    *,
    maze_xy_sampled: np.ndarray | None = None,
    dist_to_maze_tsd: nap.Tsd | None = None,
    ang_vel_tsd: nap.Tsd | None = None,
    method: str = "roman_tmaze_projected",
    lin_round: float = 0.1,
    off_maze_thresh: float = 5.0,
    on_maze_thresh: float = 1.0,
    ang_vel_thresh: float | None = 6.,
    smooth_win: float = 0.1,
    start_end_dist_thresh: float | None = 15.0,
) -> dict:
    """
    Detect off-maze epochs from distance-to-maze time series using hysteresis-like logic:
      - candidate windows: dist > on_maze_thresh
      - keep only candidates that contain any time with dist >= off_maze_thresh
      - optionally (if ang_vel_thresh is not None): candidate windows must also contain any time with |ang_vel| > ang_vel_thresh
      - optionally (if start_end_dist_thresh is not None): candidate windows must have start-end xy distance <= start_end_dist_thresh
    """
    if position_tsdf is None:
        raise ValueError("position_tsdf must be provided (nap.TsdFrame with ['x','y'] and 'lin' if method='from_lin').")
    if off_maze_thresh <= on_maze_thresh:
        raise ValueError(f"Require off_maze_thresh > on_maze_thresh; got {off_maze_thresh} <= {on_maze_thresh}")

    if maze_xy_sampled is None:
        if method == 'from_lin':
            maze_xy_sampled = get_maze_xy_sampled(position_tsdf[["x", "y", "lin"]], method=method, lin_round=lin_round)
        else:
            maze_xy_sampled = get_maze_xy_sampled(position_tsdf[["x", "y"]], method=method, lin_round=lin_round)
    if dist_to_maze_tsd is None:
        xy = position_tsdf[["x", "y"]].to_numpy()
        dist_to_maze = np.min(scipy_distance.cdist(xy, maze_xy_sampled), axis=1)
        dist_to_maze_tsd = nap.Tsd(t=position_tsdf.t, d=dist_to_maze)

    if ang_vel_tsd is None:
        ang_vel_tsd = get_ang_vel_tsd(position_tsdf, smooth_win=smooth_win)
    ang_vel_abs_tsd = nap.Tsd(t=ang_vel_tsd.t, d=np.abs(ang_vel_tsd.d), time_support=ang_vel_tsd.time_support)

    # windows where animal is away from maze (low threshold)
    above_on = dist_to_maze_tsd.threshold(on_maze_thresh, method="above")
    ep_above_on = above_on.time_support

    # times where animal is clearly off maze (high threshold)
    above_off = dist_to_maze_tsd.threshold(off_maze_thresh, method="above")
    ep_above_off = above_off.time_support

    if ang_vel_thresh is None:
        ep_high_ang_vel = None
    else:
        above_ang = ang_vel_abs_tsd.threshold(ang_vel_thresh, method="above")
        ep_high_ang_vel = above_ang.time_support

    if len(ep_above_on) == 0 or len(ep_above_off) == 0:
        ep_offmaze = nap.IntervalSet(start=np.array([]), end=np.array([]))
        out = {
            "dist_to_maze_tsd": dist_to_maze_tsd,
            "maze_xy_sampled": maze_xy_sampled,
            "ang_vel_tsd": ang_vel_tsd,
            "ep_above_on": ep_above_on,
            "ep_above_off": ep_above_off,
            "ep_high_ang_vel": ep_high_ang_vel,
            "ep_offmaze": ep_offmaze,
        }
        return out

    df_on = ep_above_on.as_dataframe()
    starts = []
    ends = []
    for s, e in zip(df_on["start"].to_numpy(), df_on["end"].to_numpy()):
        win = nap.IntervalSet(start=np.array([s]), end=np.array([e]))
        has_off = len(ep_above_off.intersect(win)) > 0
        has_ang = True if ep_high_ang_vel is None else (len(ep_high_ang_vel.intersect(win)) > 0)
        if start_end_dist_thresh is None:
            has_small_se = True
        else:
            pos_win = position_tsdf.restrict(win)
            if len(pos_win) == 0:
                has_small_se = False
            else:
                df_pos = pos_win.as_dataframe()
                dx = float(df_pos["x"].iloc[-1] - df_pos["x"].iloc[0])
                dy = float(df_pos["y"].iloc[-1] - df_pos["y"].iloc[0])
                start_end_dist = float(np.sqrt(dx * dx + dy * dy))
                has_small_se = start_end_dist <= start_end_dist_thresh

        if has_off and has_ang and has_small_se:
            starts.append(s)
            ends.append(e)

    if len(starts) == 0:
        ep_offmaze = nap.IntervalSet(start=np.array([]), end=np.array([]))
        out = {
            "dist_to_maze_tsd": dist_to_maze_tsd,
            "maze_xy_sampled": maze_xy_sampled,
            "ang_vel_tsd": ang_vel_tsd,
            "ep_above_on": ep_above_on,
            "ep_above_off": ep_above_off,
            "ep_high_ang_vel": ep_high_ang_vel,
            "ep_offmaze": ep_offmaze,
        }
        return out

    ep_offmaze = nap.IntervalSet(start=np.asarray(starts), end=np.asarray(ends))
    out = {
        "dist_to_maze_tsd": dist_to_maze_tsd,
        "maze_xy_sampled": maze_xy_sampled,
        "ang_vel_tsd": ang_vel_tsd,
        "ep_above_on": ep_above_on,
        "ep_above_off": ep_above_off,
        "ep_high_ang_vel": ep_high_ang_vel,
        "ep_offmaze": ep_offmaze,
    }
    return out

def get_behavior_ep_d(
    position_tsdf: nap.TsdFrame,
    speed_tsd: nap.Tsd,
    *,
    offmaze_method: str = "roman_tmaze_projected",
    speed_immo_thresh: float = 2.5,
    speed_loco_thresh: float = 5.0,
    **offmaze_kwargs,
) -> dict:
    """
    Get behavior epochs used in lr_7/lr_8:
      - immobility: speed < speed_immo_thresh
      - locomotion: speed > speed_loco_thresh AND not offmaze
      - offmaze: from detect_offmaze_ep
    """
    offmaze_res = detect_offmaze_ep(position_tsdf, method=offmaze_method, **offmaze_kwargs)
    ep_offmaze = offmaze_res["ep_offmaze"]
    ep_immobility = speed_tsd.threshold(speed_immo_thresh, method="below").time_support
    ep_locomotion = speed_tsd.threshold(speed_loco_thresh, method="above").time_support.set_diff(ep_offmaze)
    ep_d = {"immobility": ep_immobility, "offmaze": ep_offmaze, "locomotion": ep_locomotion}
    return {"ep_d": ep_d, "offmaze_res": offmaze_res}

def plot_offmaze(position_tsdf,ep_offmaze,fig=None,ax=None,figsize=(3,3)):
    if fig is None or ax is None:
        fig,ax=plt.subplots(figsize=figsize, constrained_layout=True)
    ax.plot(position_tsdf['x'],position_tsdf['y'],color='grey',alpha=0.5)
    for ep in ep_offmaze:
        position_offmaze = position_tsdf.restrict(ep)
        ax.plot(position_offmaze['x'],position_offmaze['y'],color='red')
    plt.tight_layout()
    return fig,ax


def debug_offmaze_res(
    position_tsdf: nap.TsdFrame,
    *,
    maze_xy_sampled: np.ndarray | None = None,
    dist_to_maze_tsd: nap.Tsd | None = None,
    ang_vel_tsd: nap.Tsd | None = None,
    method: str = "roman_tmaze_projected",
    lin_round: float = 0.1,
    off_maze_thresh: float = 5.0,
    on_maze_thresh: float = 1.0,
    ang_vel_thresh: float | None = None,
    smooth_win: float = 0.1,
    start_end_dist_thresh: float | None = 10.0,
) -> dict:
    """
    Debug helper: for each candidate window (dist > on_maze_thresh), report which gate(s) fail:
      - must contain some dist > off_maze_thresh
      - optional: must contain some |ang_vel| > ang_vel_thresh
      - optional: start-end xy distance <= start_end_dist_thresh

    Returns dict with:
      - df: pd.DataFrame (one row per candidate window)
      - ep_candidates: nap.IntervalSet (dist > on_maze_thresh)
      - ep_above_off: nap.IntervalSet (dist > off_maze_thresh)
      - ep_high_ang_vel: nap.IntervalSet (|ang_vel| > ang_vel_thresh) or None
    """
    if maze_xy_sampled is None:
        if method == 'from_lin':
            maze_xy_sampled = get_maze_xy_sampled(position_tsdf[["x", "y", "lin"]], method=method, lin_round=lin_round)
        else:
            maze_xy_sampled = get_maze_xy_sampled(position_tsdf[["x", "y"]], method=method, lin_round=lin_round)
    if dist_to_maze_tsd is None:
        xy = position_tsdf[["x", "y"]].to_numpy()
        dist_to_maze = np.min(scipy_distance.cdist(xy, maze_xy_sampled), axis=1)
        dist_to_maze_tsd = nap.Tsd(t=position_tsdf.t, d=dist_to_maze)
    if ang_vel_tsd is None:
        ang_vel_tsd = get_ang_vel_tsd(position_tsdf, smooth_win=smooth_win)

    ang_vel_abs_tsd = nap.Tsd(t=ang_vel_tsd.t, d=np.abs(ang_vel_tsd.d), time_support=ang_vel_tsd.time_support)

    ep_candidates = dist_to_maze_tsd.threshold(on_maze_thresh, method="above").time_support
    ep_above_off = dist_to_maze_tsd.threshold(off_maze_thresh, method="above").time_support

    if ang_vel_thresh is None:
        ep_high_ang_vel = None
    else:
        ep_high_ang_vel = ang_vel_abs_tsd.threshold(ang_vel_thresh, method="above").time_support

    rows = []
    df_c = ep_candidates.as_dataframe()
    for i, (s, e) in enumerate(zip(df_c["start"].to_numpy(), df_c["end"].to_numpy())):
        win = nap.IntervalSet(start=np.array([s]), end=np.array([e]))

        has_off = len(ep_above_off.intersect(win)) > 0
        has_ang = True if ep_high_ang_vel is None else (len(ep_high_ang_vel.intersect(win)) > 0)

        pos_win = position_tsdf.restrict(win)
        if len(pos_win) == 0:
            start_end_dist = np.nan
        else:
            df_pos = pos_win.as_dataframe()
            dx = float(df_pos["x"].iloc[-1] - df_pos["x"].iloc[0])
            dy = float(df_pos["y"].iloc[-1] - df_pos["y"].iloc[0])
            start_end_dist = float(np.sqrt(dx * dx + dy * dy))

        if start_end_dist_thresh is None:
            has_small_se = True
        else:
            has_small_se = np.isfinite(start_end_dist) and (start_end_dist <= start_end_dist_thresh)

        dist_win = dist_to_maze_tsd.restrict(win)
        dist_max = float(np.nanmax(dist_win.d)) if len(dist_win) > 0 else np.nan

        ang_win = ang_vel_abs_tsd.restrict(win)
        ang_vel_abs_max = float(np.nanmax(ang_win.d)) if len(ang_win) > 0 else np.nan

        rows.append(
            dict(
                cand_i=i,
                start=float(s),
                end=float(e),
                dur=float(e - s),
                dist_max=dist_max,
                ang_vel_abs_max=ang_vel_abs_max,
                start_end_dist=start_end_dist,
                pass_off=bool(has_off),
                pass_ang=bool(has_ang),
                pass_start_end=bool(has_small_se),
                pass_all=bool(has_off and has_ang and has_small_se),
            )
        )

    df = pd.DataFrame(rows)
    out = {
        "df": df,
        "ep_candidates": ep_candidates,
        "ep_above_off": ep_above_off,
        "ep_high_ang_vel": ep_high_ang_vel,
    }
    return out


def get_maze_xy_sampled(
    xy: np.ndarray | nap.TsdFrame | None,
    *,
    method: str = "roman_tmaze_projected",
    place_bin_size: float = 1.0,
    lin_round: float = 0.1,
    do_plot: bool = False,
    edge_sample_step: float = 1.0,
) -> np.ndarray:
    """
    Get **on-maze samples** (maze geometry) from tracked xy, using the same idea as
    `poor_gplvm` / `poor_man_gplvm`: linearize onto a track graph and then use the
    *projected* xy positions, uniquely sampled along `linear_position`.
    
    Parameters
    ----------
    xy : np.ndarray | nap.TsdFrame | None
        Array of shape (n, 2) with [x, y] positions, or nap.TsdFrame with 'x', 'y' columns
        If None, will generate samples directly from the (hard-coded) maze geometry.
    method : str
        - 'roman_tmaze_projected': (**default**) uses a hard-coded Roman tmaze track graph, then calls
          `track_linearization.get_linearized_position(...)` and samples unique projected xy.
        - 'from_lin': uses `position_tsdf['lin']` (if available). Samples unique linearized
          position bins (via rounding) and uses the median x/y for each bin.
        - 'roman_tmaze_edges': purely geometric sampling along the hard-coded Roman tmaze
          edges (no linearization); mainly for debugging / fallback when you really don't
          have `lin`, but beware coordinate mismatches across datasets.
    place_bin_size : float
        Passed to `replay_trajectory_classification.Environment(...)` only for plotting
        or if you later reuse the env. Not required for sampling itself.
    lin_round : float
        Uniqueness resolution in *linear_position* space. In your other pipeline you used
        `linear_position.round(1)`, so default is 0.1.
    do_plot : bool
        If True and dependencies are installed, plot the track graph (debug).
    edge_sample_step : float
        Only used when `xy is None`. Step size (in the same xy units) for sampling points
        along each maze edge.
    
    Returns
    -------
    np.ndarray
        Array of shape (m, 2) containing sampled maze positions [x, y]
    """
    # Extract xy array (optional)
    if xy is None:
        xy_arr = None
    elif isinstance(xy, nap.TsdFrame):
        xy_arr = xy[["x", "y"]].to_numpy()
    else:
        xy_arr = np.asarray(xy)
    if xy_arr is not None:
        if xy_arr.ndim != 2 or xy_arr.shape[1] != 2:
            raise ValueError(f"xy must be (n, 2); got {xy_arr.shape}")
        xy_arr = xy_arr[np.isfinite(xy_arr).all(axis=1)]

    if method == "from_lin":
        if not isinstance(xy, nap.TsdFrame):
            raise ValueError("method='from_lin' requires `xy` to be a nap.TsdFrame with columns ['x','y','lin']")
        df = xy.as_dataframe()
        if not all(k in df.columns for k in ("x", "y", "lin")):
            raise ValueError(f"method='from_lin' requires columns ['x','y','lin']; got cols={list(df.columns)}")
        lin = df["lin"].to_numpy()
        x = df["x"].to_numpy()
        y = df["y"].to_numpy()
        m = np.isfinite(lin) & np.isfinite(x) & np.isfinite(y)
        lin = lin[m]
        x = x[m]
        y = y[m]
        if lin_round <= 0:
            raise ValueError(f"lin_round must be > 0; got {lin_round}")
        decimals = int(np.round(-np.log10(lin_round)))
        lin_r = np.round(lin, decimals)
        # unique along lin bins, take median x/y per bin
        uniq = np.unique(lin_r)
        xy_out = np.full((len(uniq), 2), np.nan, dtype=float)
        for i, v in enumerate(uniq):
            mm = lin_r == v
            xy_out[i, 0] = np.nanmedian(x[mm])
            xy_out[i, 1] = np.nanmedian(y[mm])
        xy_out = xy_out[np.isfinite(xy_out).all(axis=1)]
        return xy_out

    if method == "roman_tmaze_edges":
        # Uses the hard-coded Roman tmaze geometry; coordinate mismatch is possible across datasets.
        # Optional deps not needed here.
        if edge_sample_step <= 0:
            raise ValueError(f"edge_sample_step must be > 0; got {edge_sample_step}")
        node_positions = np.asarray([
            (81.5, 45.8),  # 0-home
            (8.2, 45.5),   # 1-T
            (11.0, 10.0),  # 2-left reward
            (82.4, 12.6),  # 3-left return
            (9.9, 84.9),   # 4-right reward
            (82.5, 81.0),  # 5-right return
        ], dtype=float)
        edges = np.asarray([
            (0, 1),
            (1, 2),
            (2, 3),
            (3, 0),
            (1, 4),
            (4, 5),
            (5, 0),
        ], dtype=int)
        pts = []
        for a, b in edges:
            p0 = node_positions[int(a)]
            p1 = node_positions[int(b)]
            seg_len = float(np.linalg.norm(p1 - p0))
            n = max(2, int(np.ceil(seg_len / edge_sample_step)) + 1)
            t = np.linspace(0.0, 1.0, n)
            pts.append(p0[None, :] * (1 - t)[:, None] + p1[None, :] * t[:, None])
        return np.unique(np.vstack(pts), axis=0)

    if method != "roman_tmaze_projected":
        raise ValueError(f"Unknown method={method!r}")

    # Optional deps (kept local so this module doesn't hard-require them)
    try:
        import replay_trajectory_classification as rtc
        import track_linearization as track_linearization
    except Exception as e:
        raise ImportError(
            "maze_xy_sampled(method='roman_tmaze_projected') requires "
            "`replay_trajectory_classification` and `track_linearization`."
        ) from e
    make_track_graph = getattr(rtc, "make_track_graph", None)
    plot_track_graph = getattr(rtc, "plot_track_graph", None)
    Environment = getattr(rtc, "Environment", None)
    if make_track_graph is None or Environment is None:
        raise ImportError(
            "Could not access `make_track_graph` / `Environment` on `replay_trajectory_classification`. "
            "Please check your installed version."
        )

    # Roman tmaze geometry (copied from your poor_gplvm get_environment.get_roman_tmaze)
    node_positions = [
        (81.5, 45.8),  # 0-home
        (8.2, 45.5),   # 1-T
        (11.0, 10.0),  # 2-left reward
        (82.4, 12.6),  # 3-left return
        (9.9, 84.9),   # 4-right reward
        (82.5, 81.0),  # 5-right return
    ]
    edges = [
        (0, 1),
        (1, 2),
        (2, 3),
        (3, 0),
        (1, 4),
        (4, 5),
        (5, 0),
    ]
    edge_order = [
        (0, 1),
        (1, 2),
        (2, 3),
        (3, 0),
        (1, 4),
        (4, 5),
        (5, 0),
    ]
    edge_spacing = (0, 0, 0, 50, 0, 0)

    track_graph = make_track_graph(node_positions, edges)
    if do_plot:
        fig, ax = plt.subplots(constrained_layout=True)
        if plot_track_graph is None:
            raise ImportError(
                "do_plot=True requires `replay_trajectory_classification.plot_track_graph` to exist "
                "(not found in your installed version)."
            )
        plot_track_graph(track_graph, ax=ax)
        plt.tight_layout()

    if xy_arr is None:
        raise ValueError("method='roman_tmaze_projected' requires xy (tracked positions). If you lack xy, use method='roman_tmaze_edges'.")

    # Data-driven sampling: linearize & get projected xy along the graph (matches poor_gplvm)
    pos_df = track_linearization.get_linearized_position(
        xy_arr,
        track_graph,
        edge_order=edge_order,
        edge_spacing=edge_spacing,
        use_HMM=False,
    )

    # (Optional) env creation kept for parity/debug; not used for sampling
    _ = Environment(
        place_bin_size=place_bin_size,
        track_graph=track_graph,
        edge_order=edge_order,
        edge_spacing=edge_spacing,
    )

    if "linear_position" not in pos_df.columns:
        raise ValueError(f"Expected 'linear_position' in linearized position df; got cols={list(pos_df.columns)}")
    if "projected_x_position" not in pos_df.columns or "projected_y_position" not in pos_df.columns:
        raise ValueError(
            "Expected projected_x_position / projected_y_position in linearized position df; "
            f"got cols={list(pos_df.columns)}"
        )

    # Match your other pipeline: unique along linear_position.round(1) by default
    if lin_round <= 0:
        raise ValueError(f"lin_round must be > 0; got {lin_round}")
    decimals = int(np.round(-np.log10(lin_round)))
    _, uniq_ind = np.unique(pos_df["linear_position"].round(decimals), return_index=True)
    maze_coord_df = pos_df.iloc[uniq_ind].reset_index(drop=True)
    xy_sampled_all = maze_coord_df[["projected_x_position", "projected_y_position"]].to_numpy()

    return xy_sampled_all


def get_behavior_type_transmat(
    bc_ep_d,
    time_stamps,
    speed=None,
    unknown_label=None,
    include_unknown=False,
):
    """
    Assign behavior category to each time stamp via bc_ep_d intervals, then compute
    transition probability matrix (row-stochastic). Optionally apply matrix exponential
    with scalar speed: P_out = expm(speed * logm(P)).

    Parameters
    ----------
    bc_ep_d : dict[str, nap.IntervalSet]
        Behavior name -> IntervalSet (e.g. from get_behavior_ep_d: immobility, offmaze, locomotion).
    time_stamps : array-like or nap.Tsd
        Time points to label (seconds).
    speed : float or None
        If not None, apply matrix exponential: P_out = expm(speed * logm(P)).
        Requires P to be embeddable (logm can fail on some matrices).
    unknown_label : str or None
        Label for times not in any interval. If None, use "unknown".
    include_unknown : bool
        If True, include unknown as a row/column when present. If False (default), exclude
        unknown and renormalize rows so matrix is row-stochastic over known categories only.

    Returns
    -------
    pd.DataFrame
        Transition probability matrix, index/columns = behavior types.
    """
    t = np.asarray(time_stamps).ravel()
    if hasattr(time_stamps, "t"):
        t = np.asarray(time_stamps.t).ravel()
    categories = list(bc_ep_d.keys())
    ts_obj = nap.Ts(t)
    # assign label: first match in categories order
    labels = np.full(len(t), unknown_label if unknown_label is not None else "unknown", dtype=object)
    for cat in categories:
        in_cat = bc_ep_d[cat].in_interval(ts_obj)
        d = in_cat.d if hasattr(in_cat, "d") else np.asarray(in_cat)
        mask = np.isfinite(d)
        labels[mask] = cat
    unk = unknown_label if unknown_label is not None else "unknown"
    if include_unknown and np.any(labels == unk):
        use_cats = list(categories) + [unk]
    else:
        use_cats = list(categories)
    cat_to_ix = {c: i for i, c in enumerate(use_cats)}
    n = len(use_cats)
    counts = np.zeros((n, n))
    for i in range(len(labels) - 1):
        a, b = labels[i], labels[i + 1]
        if a in cat_to_ix and b in cat_to_ix:
            counts[cat_to_ix[a], cat_to_ix[b]] += 1
    row_sum = counts.sum(axis=1, keepdims=True)
    row_sum[row_sum == 0] = 1
    P = counts / row_sum
    if speed is not None:
        try:
            Q = scipy.linalg.logm(P)
            P = scipy.linalg.expm(float(speed) * Q)
            # re-normalize rows (numerical)
            row_sum = P.sum(axis=1, keepdims=True)
            row_sum[row_sum == 0] = 1
            P = P / row_sum
        except Exception as e:
            raise ValueError("matrix exponential failed (P may not be embeddable); try speed=None") from e
    return pd.DataFrame(P, index=use_cats, columns=use_cats)
