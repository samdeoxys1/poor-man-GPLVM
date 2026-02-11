'''
"lazy" because the input to the functions here tend to be high level, aggregated data
More for my own convenience and harder to use for others...
'''

import numpy as np
import textwrap

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.colors as mcolors
import matplotlib.transforms as mtransforms

import pynapple as nap

import poor_man_gplvm.plot_helper as ph
import mpl_toolkits.axes_grid1.inset_locator as inset_locator


def _sec_to_hms_str(t_s, *, decimals=2):
    t_s = float(t_s)
    if not np.isfinite(t_s):
        return f'{t_s}'
    sign = '-' if t_s < 0 else ''
    t_s = abs(t_s)
    h = int(t_s // 3600.0)
    m = int((t_s % 3600.0) // 60.0)
    s = t_s % 60.0
    fmt = f'{{s:0{2 + (1 + int(decimals)) if decimals else 0}.{int(decimals)}f}}'
    return f'{sign}{h:02d}:{m:02d}:' + fmt.format(s=s)


def _get_event_window_start_end(replay_res, event_i):
    if replay_res is None:
        return None, None
    pbe_res = replay_res.get('pbe_res', None)
    if pbe_res is None:
        return None, None
    ev = pbe_res.get('event_windows', None)
    if ev is None:
        return None, None
    i = int(event_i)
    try:
        st = float(np.asarray(ev.start)[i])
        ed = float(np.asarray(ev.end)[i])
        return st, ed
    except Exception:
        try:
            df = ev.as_dataframe().reset_index(drop=True)
            st = float(df.loc[i, 'start'])
            ed = float(df.loc[i, 'end'])
            return st, ed
        except Exception:
            return None, None


def _get_df_row_value(df, event_i, col, default=None):
    if df is None:
        return default
    try:
        return df.loc[int(event_i), col]
    except Exception:
        try:
            return df.iloc[int(event_i)][col]
        except Exception:
            return default


def _flatten_axs(axs):
    if axs is None:
        return None
    if hasattr(axs, 'plot') and hasattr(axs, 'imshow'):
        return [axs]
    if isinstance(axs, (list, tuple)):
        out = []
        for a in axs:
            out.extend(_flatten_axs(a))
        return out
    try:
        arr = np.asarray(axs, dtype=object).ravel()
        return [a for a in arr.tolist() if a is not None]
    except Exception:
        return [axs]


def _ensure_tsdframe(arr, *, t=None, columns=None):
    if arr is None:
        return None
    if isinstance(arr, (nap.Tsd, nap.TsdFrame)):
        return arr
    a = np.asarray(arr)
    if a.ndim == 1:
        if t is None:
            t = np.arange(a.shape[0], dtype=float)
        return nap.Tsd(t=np.asarray(t), d=a)
    if a.ndim == 2:
        if t is None:
            t = np.arange(a.shape[0], dtype=float)
        if columns is None:
            columns = np.arange(a.shape[1])
        return nap.TsdFrame(t=np.asarray(t), d=a, columns=np.asarray(columns))
    return None


def _robust_vmin_vmax(arr2d, *, q_low=0.01, q_high=0.99):
    a = np.asarray(arr2d, dtype=float)
    if a.size == 0:
        return 0.0, 1.0
    try:
        vmin = float(np.nanquantile(a, float(q_low)))
        vmax = float(np.nanquantile(a, float(q_high)))
    except Exception:
        vmin = float(np.nanmin(a))
        vmax = float(np.nanmax(a))
    if not np.isfinite(vmin):
        vmin = 0.0
    if not np.isfinite(vmax):
        vmax = 1.0
    if vmax <= vmin:
        vmax = vmin + 1e-6
    return vmin, vmax


def _plot_posterior_heatmap(
    ax,
    tsdf,
    *,
    title=None,
    cmap='viridis',
    vmin_q=0.01,
    vmax_q=0.99,
    add_scatter_map=False,
    scatter_kwargs=None,
    yticks=None,
    yticklabels=None,
    ylabel=None,
):
    tsdf = _ensure_tsdframe(tsdf)
    if tsdf is None:
        return None

    t = np.asarray(tsdf.t)
    d = np.asarray(tsdf.d, dtype=float)
    if d.ndim == 1:
        d = d[:, None]

    vmin, vmax = _robust_vmin_vmax(d, q_low=vmin_q, q_high=vmax_q)
    im = ax.imshow(
        d.T,
        aspect='auto',
        origin='lower',
        interpolation='none',
        extent=[float(t.min()), float(t.max()), 0.0, float(d.shape[1])],
        cmap=cmap,
        norm=mcolors.Normalize(vmin=vmin, vmax=vmax),
    )

    if add_scatter_map:
        mk = np.nanargmax(d, axis=1).astype(int)
        sk = dict(s=4, c='yellow')
        if scatter_kwargs is not None:
            sk.update(dict(scatter_kwargs))
        ax.scatter(t, mk + 0.5, **sk)

    if yticks is not None:
        ax.set_yticks(yticks)
    if yticklabels is not None:
        ax.set_yticklabels(yticklabels)
    if ylabel is not None:
        ax.set_ylabel(str(ylabel))
    if title is not None:
        ax.set_title(str(title))
    return im


def _pick_dynamics_column(dyn_tsdf, dynamics_col, state_names_fallback):
    dyn_tsdf = _ensure_tsdframe(dyn_tsdf)
    if dyn_tsdf is None:
        return None, None, None

    cols = getattr(dyn_tsdf, 'columns', None)
    if dynamics_col is None:
        return dyn_tsdf, None, None

    if isinstance(dynamics_col, str):
        name = str(dynamics_col)
        if cols is not None:
            try:
                cols_l = list(cols)
                if name in cols_l:
                    j = int(cols_l.index(name))
                else:
                    j = int(name)  # allow "0"/"1" as str
            except Exception:
                j = int(name)
        else:
            j = int(name)
    else:
        j = int(dynamics_col)
        name = None

    d = np.asarray(dyn_tsdf.d)
    if d.ndim == 1:
        d = d[:, None]
    j = int(np.clip(j, 0, d.shape[1] - 1))

    if name is None:
        if cols is not None:
            try:
                name = str(list(cols)[j])
            except Exception:
                name = None
    if name is None:
        if state_names_fallback is not None and j < len(state_names_fallback):
            name = str(state_names_fallback[j])
        else:
            name = str(j)

    p = nap.Tsd(t=np.asarray(dyn_tsdf.t), d=np.asarray(d[:, j], dtype=float))
    return dyn_tsdf, p, name


def _plot_dynamics_panel(
    ax,
    dyn_tsdf,
    *,
    dynamics_col=None,
    state_names=None,
    title=None,
    heatmap_cmap='magma',
    line_kwargs=None,
    heatmap_vmin_q=0.01,
    heatmap_vmax_q=0.99,
):
    dyn_tsdf, p, name = _pick_dynamics_column(dyn_tsdf, dynamics_col, state_names)
    if dyn_tsdf is None:
        return dict(mode=None, dyn_tsdf=None, p=None, name=None)

    if dynamics_col is None:
        n_state = int(np.asarray(dyn_tsdf.d).shape[1]) if np.ndim(dyn_tsdf.d) == 2 else 1
        yticks = None
        yticklabels = None
        if state_names is not None and n_state == len(state_names):
            yticks = (np.arange(n_state) + 0.5).tolist()
            yticklabels = list(state_names)
        _plot_posterior_heatmap(
            ax,
            dyn_tsdf,
            title=title,
            cmap=heatmap_cmap,
            vmin_q=heatmap_vmin_q,
            vmax_q=heatmap_vmax_q,
            yticks=yticks,
            yticklabels=yticklabels,
        )
        return dict(mode='heatmap', dyn_tsdf=dyn_tsdf, p=None, name=None)

    lk = dict(color='k', lw=1.5)
    if line_kwargs is not None:
        lk.update(dict(line_kwargs))
    ax.plot(p.t, p.d, **lk)
    ax.set_ylim(-0.02, 1.02)
    ax.set_ylabel(f'P("{name}")')
    if title is not None:
        ax.set_title(str(title))
    return dict(mode='line', dyn_tsdf=dyn_tsdf, p=p, name=name)


def plot_replay_sup_unsup_event(
    replay_res_sup=None,
    replay_res_unsup=None,
    event_i=0,
    *,
    position_tsdf=None,
    maze='familiar',
    plot_sup_trajectory=True,
    plot_sup_dynamics=True,
    plot_unsup_latent=True,
    plot_unsup_dynamics=True,
    layout='two_col',
    dynamics_col=None,
    time_panel_order=None,
    fig=None,
    axs=None,
    figsize=(10, 6),
    width_ratios=(1.25, 1.0),
    height_ratios=None,
    traj_box_aspect=1.0,
    traj_inset=True,
    traj_inset_frac=0.92,
    title_fontsize=10,
    title_wrap_width=110,
    sup_2d_kwargs=None,
    unsup_heatmap_kwargs=None,
    heatmap_add_scatter_latent=False,
    heatmap_add_scatter_dynamics=False,
    latent_cmap='Greys',
    dyn_cmap_sup='viridis',
    dyn_cmap_unsup='viridis',
    latent_vmin_q=0.01,
    latent_vmax_q=0.99,
    dynamics_vmin_q=0.01,
    dynamics_vmax_q=0.99,
    dynamics_line_kwargs=None,
    sup_2d_plot_colorbar=True,
    sup_2d_cbar_bbox=(0, 0, 1, 0.05),
    sup_2d_cbar_thickness=None,
    sup_2d_cbar_tie_time_axis=False,
    time_scalebar=True,
):
    """
    Plot replay for a single event from supervised + unsupervised pipelines.

    Panels (toggleable)
    - Decoded spatial trajectory (supervised): 2D posterior + MAP overlay.
    - Decoded dynamics (supervised): dynamics posterior (heatmap if dynamics_col=None; otherwise a single prob trace).
    - Decoded latent (unsupervised): latent posterior heatmap (best transition kernel).
    - Decoded dynamics (unsupervised): dynamics posterior (best transition kernel).

    Layout
    - layout='two_col': 2 columns x 2 rows
      left: supervised trajectory, supervised dynamics
      right: unsup latent, unsup dynamics
    - layout='time_col': 2 columns with independent widths
      left: supervised trajectory only (2D; square-ish)
      right: all time-based panels stacked (same width)
    - layout='one_col': 1 column stacked panels
      unsup latent, unsup dynamics, sup trajectory, sup dynamics

    Progress colorbar (2D trajectory)
    - sup_2d_cbar_bbox: (x0, y0, w, h) in trajectory axis axes coords. Default (0, -0.04, 1, 0.10).
      y0 = vertical position (lower = further below). w = span along axis (1 = full width). h = thickness if sup_2d_cbar_thickness not set.
    - sup_2d_cbar_thickness: height (thickness) of the horizontal bar in axes coords, e.g. 0.06 or 0.12. Overrides bbox[3] when set.
    - sup_2d_cbar_tie_time_axis: if True, progress bar uses same horizontal position/width as time panels and xlim(st, ed) so it aligns with time axis; requires at least one time panel.

    Returns
    -------
    fig, axs, out_dict
      axs: list of matplotlib Axes in plotting order
        (sup_traj, sup_dyn, unsup_latent, unsup_dyn; some may be missing)
      out_dict: dict with extracted info (start/end time, best_kernel, probs, flags).

    Cluster Jupyter example

```python
import poor_man_gplvm._plot_helper_lazy as phl

fig, axs, out = phl.plot_replay_sup_unsup_event(
    replay_res_sup=replay_res_sup,
    replay_res_unsup=replay_res_unsup,
    event_i=10,
    position_tsdf=prep_res['position_tsdf'],  # required if plot_sup_trajectory=True
    plot_sup_trajectory=True,
    plot_sup_dynamics=True,
    plot_unsup_latent=True,
    plot_unsup_dynamics=True,
    dynamics_col=None,  # None -> heatmap; 0/1 -> plot P("state") trace
    layout='two_col',   # or 'one_col'
    figsize=(12, 6),
    width_ratios=(1.4, 1.0),
    sup_2d_kwargs=dict(cmap_name='plasma', binarize_thresh=0.01),
    dynamics_line_kwargs=dict(lw=2, color='k'),
)
```
    """
    mpl.rcParams['pdf.fonttype'] = 42
    mpl.rcParams['svg.fonttype'] = 'none'

    sup_2d_kwargs_ = {} if sup_2d_kwargs is None else dict(sup_2d_kwargs)
    # progress colorbar drawn below 2D traj in lazy helper; do not use ph's top colorbar
    sup_2d_kwargs_.setdefault('plot_colorbar', False)
    unsup_heatmap_kwargs_ = {} if unsup_heatmap_kwargs is None else dict(unsup_heatmap_kwargs)  # kept for backward compat

    event_i = int(event_i)

    has_sup_traj = bool(plot_sup_trajectory) and (replay_res_sup is not None)
    has_sup_dyn = bool(plot_sup_dynamics) and (replay_res_sup is not None)
    has_unsup_lat = bool(plot_unsup_latent) and (replay_res_unsup is not None)
    has_unsup_dyn = bool(plot_unsup_dynamics) and (replay_res_unsup is not None)
    n_panels_req = int(has_sup_traj) + int(has_sup_dyn) + int(has_unsup_lat) + int(has_unsup_dyn)

    if has_sup_traj and (position_tsdf is None):
        raise ValueError('position_tsdf must be provided when plot_sup_trajectory=True and replay_res_sup is not None.')

    # ---- event time (prefer supervised, else unsupervised) ----
    st_sup, ed_sup = _get_event_window_start_end(replay_res_sup, event_i)
    st_uns, ed_uns = _get_event_window_start_end(replay_res_unsup, event_i)
    st = st_sup if st_sup is not None else st_uns
    ed = ed_sup if ed_sup is not None else ed_uns
    dur = None if (st is None or ed is None) else float(ed - st)

    # ---- supervised stats ----
    sup_sig = None
    sup_max_disp = None
    if replay_res_sup is not None:
        rm = replay_res_sup.get('replay_metrics_df', None)
        if rm is not None:
            if 'is_sig_overall' in rm.columns:
                v = _get_df_row_value(rm, event_i, 'is_sig_overall', default=None)
                if v is not None and not (isinstance(v, float) and np.isnan(v)):
                    sup_sig = bool(v)
            sup_max_disp = _get_df_row_value(rm, event_i, 'max_displacement', default=None)
            try:
                sup_max_disp = None if sup_max_disp is None else float(sup_max_disp)
            except Exception:
                pass

    # ---- unsupervised stats + best kernel ----
    unsup_sig = None
    best_kernel = None
    probs = None
    post_latent_best = None
    post_dyn_best = None
    if replay_res_unsup is not None:
        ev_df = replay_res_unsup.get('event_df_joint', None)
        if ev_df is not None and ('is_sig_overall' in ev_df.columns):
            v = _get_df_row_value(ev_df, event_i, 'is_sig_overall', default=None)
            if v is not None and not (isinstance(v, float) and np.isnan(v)):
                unsup_sig = bool(v)

        cmp = replay_res_unsup.get('pbe_compare_transition_res', None)
        if cmp is not None:
            prob_df = cmp.get('prob_category_per_event_df', None)
            if prob_df is not None:
                try:
                    probs = prob_df.iloc[event_i]
                    best_kernel = str(probs.idxmax())
                except Exception:
                    probs = None
                    best_kernel = None

            if best_kernel is not None:
                post_lat_d = cmp.get('posterior_latent_marg_per_event', None)
                post_dyn_d = cmp.get('posterior_dynamics_marg_per_event', None)
                try:
                    post_latent_best = post_lat_d[best_kernel][event_i] if post_lat_d is not None else None
                except Exception:
                    post_latent_best = None
                try:
                    post_dyn_best = post_dyn_d[best_kernel][event_i] if post_dyn_d is not None else None
                except Exception:
                    post_dyn_best = None
                post_latent_best = _ensure_tsdframe(post_latent_best)
                post_dyn_best = _ensure_tsdframe(post_dyn_best)

    # ---- supervised posteriors per event ----
    sup_dyn_one = None
    if replay_res_sup is not None:
        decode_res_sup = replay_res_sup.get('decode_res', None)
        if decode_res_sup is not None:
            try:
                sup_dyn_one = decode_res_sup.get('posterior_dynamics_marg_per_event', None)
                if isinstance(sup_dyn_one, list):
                    sup_dyn_one = sup_dyn_one[event_i]
            except Exception:
                sup_dyn_one = None
    sup_dyn_one = _ensure_tsdframe(sup_dyn_one)

    # Binsize from data (t[1]-t[0]) for label; empirical block width (t.max()-t.min())/n for bar length
    def _binsize_and_block_from_ts(ts):
        if ts is None:
            return None, None
        t = np.asarray(getattr(ts, 't', ts) if hasattr(ts, 't') else ts)
        n = getattr(t, 'size', len(t))
        if n < 2:
            return None, None
        binsize = float(t[1] - t[0])
        block = (float(np.nanmax(t)) - float(np.nanmin(t))) / max(1, n)
        return binsize, block
    binsize_sec = None
    block_sec = None
    for ts in (sup_dyn_one, post_latent_best, post_dyn_best):
        b, blk = _binsize_and_block_from_ts(ts)
        if b is not None and b > 0:
            binsize_sec = b
            block_sec = blk if blk and blk > 0 else b
            break
    binsize_ms = int(round(binsize_sec * 1000.0)) if binsize_sec and binsize_sec > 0 else None

    # ---- axes allocation ----
    axs_flat = _flatten_axs(axs)
    out_axs = []
    ax_sup_traj_container = None
    bottom_time_axs = []

    if axs_flat is not None:
        need = int(has_sup_traj) + int(has_sup_dyn) + int(has_unsup_lat) + int(has_unsup_dyn)
        if len(axs_flat) < need:
            raise ValueError('Provided axs does not have enough axes for requested panels.')

        idx = 0
        ax_sup_traj = axs_flat[idx] if has_sup_traj else None
        idx += int(has_sup_traj)
        ax_sup_dyn = axs_flat[idx] if has_sup_dyn else None
        idx += int(has_sup_dyn)
        ax_uns_lat = axs_flat[idx] if has_unsup_lat else None
        idx += int(has_unsup_lat)
        ax_uns_dyn = axs_flat[idx] if has_unsup_dyn else None

        if fig is None:
            fig = (ax_sup_traj or ax_sup_dyn or ax_uns_lat or ax_uns_dyn).figure
    else:
        if fig is None:
            layout_ = str(layout).lower()
            if height_ratios is None:
                if layout_ == 'two_col':
                    height_ratios_use = (1.0, 1.0)
                elif layout_ == 'time_col':
                    height_ratios_use = (1.0, 1.0)
                elif layout_ == 'one_col':
                    height_ratios_use = tuple([1.0] * max(1, n_panels_req))
                else:
                    height_ratios_use = (1.0, 1.0)
            else:
                height_ratios_use = tuple(height_ratios)

            if n_panels_req == 0:
                fig, ax0 = plt.subplots(figsize=figsize, constrained_layout=True)
                ax0.text(0.5, 0.5, 'No panels to plot', ha='center', va='center', transform=ax0.transAxes)
                ax0.set_axis_off()
                ax_sup_traj = ax0
                ax_sup_dyn = None
                ax_uns_lat = None
                ax_uns_dyn = None
            elif layout_ == 'one_col':
                fig = plt.figure(figsize=figsize, constrained_layout=True)
                gs = gridspec.GridSpec(
                    nrows=max(1, n_panels_req),
                    ncols=1,
                    figure=fig,
                    height_ratios=height_ratios_use,
                )
                rr = 0
                ax_uns_lat = fig.add_subplot(gs[rr, 0]) if has_unsup_lat else None
                rr += int(has_unsup_lat)
                ax_uns_dyn = fig.add_subplot(gs[rr, 0], sharex=ax_uns_lat) if has_unsup_dyn else None
                rr += int(has_unsup_dyn)
                # NOTE: do NOT share axes between spatial (x/y) and time-based panels.
                ax_sup_traj = fig.add_subplot(gs[rr, 0]) if has_sup_traj else None
                rr += int(has_sup_traj)
                # dynamics is time-based; share with other time-based panels if present
                time_ref = ax_uns_lat or ax_uns_dyn
                ax_sup_dyn = fig.add_subplot(gs[rr, 0], sharex=time_ref) if has_sup_dyn else None
                # one_col: lowest row = last time panel
                if ax_sup_dyn is not None:
                    bottom_time_axs = [ax_sup_dyn]
                elif ax_uns_dyn is not None:
                    bottom_time_axs = [ax_uns_dyn]
                elif ax_uns_lat is not None:
                    bottom_time_axs = [ax_uns_lat]
            else:
                # two_col + time_col layouts
                fig = plt.figure(figsize=figsize, constrained_layout=True)
                if layout_ == 'time_col':
                    # left column: 2D trajectory (independent width, square-ish)
                    # right column: all time panels stacked (same width)
                    gs_outer = gridspec.GridSpec(
                        nrows=1,
                        ncols=2,
                        figure=fig,
                        width_ratios=tuple(width_ratios),
                    )
                    ax_sup_traj = fig.add_subplot(gs_outer[0, 0]) if has_sup_traj else None

                    # decide time panel order
                    if time_panel_order is None:
                        time_panel_order_use = ('sup_dyn', 'unsup_lat', 'unsup_dyn')
                    else:
                        time_panel_order_use = tuple(time_panel_order)

                    # map keys -> required axes
                    key_to_on = {
                        'sup_dyn': bool(has_sup_dyn),
                        'unsup_lat': bool(has_unsup_lat),
                        'unsup_dyn': bool(has_unsup_dyn),
                    }
                    keys = [k for k in time_panel_order_use if key_to_on.get(k, False)]
                    if len(keys) == 0:
                        # still create an empty time axis so layout is stable
                        keys = ['_empty_time']

                    if height_ratios is None:
                        height_ratios_time = [1.0] * len(keys)
                    else:
                        height_ratios_time = list(height_ratios)
                        if len(height_ratios_time) != len(keys):
                            height_ratios_time = [1.0] * len(keys)

                    gs_time = gridspec.GridSpecFromSubplotSpec(
                        nrows=len(keys),
                        ncols=1,
                        subplot_spec=gs_outer[0, 1],
                        height_ratios=height_ratios_time,
                    )

                    ax_time_ref = None
                    ax_sup_dyn = None
                    ax_uns_lat = None
                    ax_uns_dyn = None
                    for ii, k in enumerate(keys):
                        sharex = ax_time_ref if ax_time_ref is not None else None
                        ax_i = fig.add_subplot(gs_time[ii, 0], sharex=sharex)
                        if ax_time_ref is None:
                            ax_time_ref = ax_i
                        if k == 'sup_dyn':
                            ax_sup_dyn = ax_i
                        elif k == 'unsup_lat':
                            ax_uns_lat = ax_i
                        elif k == 'unsup_dyn':
                            ax_uns_dyn = ax_i
                        else:
                            ax_i.set_axis_off()
                    # time_col: lowest = last in stack
                    last_ax = ax_sup_dyn or ax_uns_lat or ax_uns_dyn
                    if last_ax is not None:
                        bottom_time_axs = [last_ax]
                else:
                    # two_col default: 2x2 grid
                    gs = gridspec.GridSpec(
                        nrows=2,
                        ncols=2,
                        figure=fig,
                        width_ratios=tuple(width_ratios),
                        height_ratios=height_ratios_use if len(height_ratios_use) == 2 else (1.0, 1.0),
                    )
                    if has_sup_traj and bool(traj_inset):
                        # container axis fills the cell; inset axis is square-ish and centered
                        ax_sup_traj_container = fig.add_subplot(gs[0, 0])
                        ax_sup_traj_container.set_axis_off()
                        frac = float(traj_inset_frac)
                        frac = float(np.clip(frac, 0.1, 1.0))
                        ax_sup_traj = inset_locator.inset_axes(
                            ax_sup_traj_container,
                            width=f"{100.0 * frac:.1f}%",
                            height=f"{100.0 * frac:.1f}%",
                            loc='center',
                            borderpad=0.0,
                        )
                    else:
                        ax_sup_traj = fig.add_subplot(gs[0, 0]) if has_sup_traj else None
                    # pick a time reference axis for sharing among time panels only
                    ax_uns_lat = fig.add_subplot(gs[0, 1]) if has_unsup_lat else None
                    ax_uns_dyn = fig.add_subplot(gs[1, 1], sharex=ax_uns_lat) if has_unsup_dyn else None
                    time_ref = ax_uns_lat or ax_uns_dyn
                    ax_sup_dyn = fig.add_subplot(gs[1, 0], sharex=time_ref) if has_sup_dyn else None
                    # two_col: lowest row = both bottom panels
                    bottom_time_axs = [a for a in (ax_sup_dyn, ax_uns_dyn) if a is not None]
        else:
            ax_sup_traj = None
            ax_sup_dyn = None
            ax_uns_lat = None
            ax_uns_dyn = None

    # ---- plot supervised trajectory ----
    if has_sup_traj and ax_sup_traj is not None:
        decode_res = replay_res_sup.get('decode_res', None)
        if decode_res is not None:
            ph.plot_decode_res_posterior_and_map_2d(
                decode_res,
                maze,
                event_i,
                position_tsdf,
                fig=fig,
                ax=ax_sup_traj,
                **sup_2d_kwargs_,
            )
        # Title on container (keeps inset axis square-ish). Fallback to plotting axis.
        if ax_sup_traj_container is not None:
            ax_sup_traj_container.set_title('Decoded spatial trajectory\n(supervised)')
        else:
            ax_sup_traj.set_title('Decoded spatial trajectory\n(supervised)')
        # make it look natural / close to square (independent of time axes)
        try:
            ax_sup_traj.set_aspect('equal', adjustable='box')
        except Exception:
            pass
        try:
            if traj_box_aspect is not None:
                ax_sup_traj.set_box_aspect(float(traj_box_aspect))
        except Exception:
            pass
        out_axs.append(ax_sup_traj)

    # ---- plot supervised dynamics ----
    sup_dyn_plot_info = dict(mode=None, name=None)
    if has_sup_dyn and (ax_sup_dyn is not None) and (sup_dyn_one is not None):
        sup_dyn_plot_info = _plot_dynamics_panel(
            ax_sup_dyn,
            sup_dyn_one,
            dynamics_col=dynamics_col,
            state_names=['continuous', 'fragmented'],
            title='Decoded dynamics (supervised)',
            heatmap_cmap=dyn_cmap_sup,
            line_kwargs=dynamics_line_kwargs,
            heatmap_vmin_q=dynamics_vmin_q,
            heatmap_vmax_q=dynamics_vmax_q,
        )
        out_axs.append(ax_sup_dyn)
    elif has_sup_dyn and (ax_sup_dyn is not None):
        ax_sup_dyn.set_title('Decoded dynamics (supervised)')
        out_axs.append(ax_sup_dyn)

    # ---- plot unsupervised latent ----
    if has_unsup_lat and (ax_uns_lat is not None) and (post_latent_best is not None):
        sk = dict(s=4, c='yellow')
        sk.update({k: v for k, v in unsup_heatmap_kwargs_.items() if k.startswith('heatmap_scatter_')})
        _plot_posterior_heatmap(
            ax_uns_lat,
            post_latent_best,
            title='Decoded latent (unsupervised)',
            cmap=latent_cmap,
            vmin_q=latent_vmin_q,
            vmax_q=latent_vmax_q,
            add_scatter_map=bool(heatmap_add_scatter_latent),
            scatter_kwargs=sk,
            ylabel='latent',
        )
        out_axs.append(ax_uns_lat)
    elif has_unsup_lat and (ax_uns_lat is not None):
        ax_uns_lat.set_title('Decoded latent (unsupervised)')
        out_axs.append(ax_uns_lat)

    # ---- plot unsupervised dynamics ----
    unsup_dyn_plot_info = dict(mode=None, name=None)
    if has_unsup_dyn and (ax_uns_dyn is not None) and (post_dyn_best is not None):
        unsup_dyn_plot_info = _plot_dynamics_panel(
            ax_uns_dyn,
            post_dyn_best,
            dynamics_col=dynamics_col,
            state_names=['consistent', 'inconsistent'],
            title='Decoded dynamics (unsupervised)',
            heatmap_cmap=dyn_cmap_unsup,
            line_kwargs=dynamics_line_kwargs,
            heatmap_vmin_q=dynamics_vmin_q,
            heatmap_vmax_q=dynamics_vmax_q,
        )
        out_axs.append(ax_uns_dyn)
    elif has_unsup_dyn and (ax_uns_dyn is not None):
        ax_uns_dyn.set_title('Decoded dynamics (unsupervised)')
        out_axs.append(ax_uns_dyn)

    # ---- hide x ticks and labels on all time panels; scalebar (unit) on lowest row only ----
    all_time_axs = [a for a in (ax_sup_dyn, ax_uns_lat, ax_uns_dyn) if a is not None]
    for ax in all_time_axs:
        try:
            ax.tick_params(axis='x', which='both', length=0, labelbottom=False)
        except Exception:
            pass

    # ---- progress colorbar below 2D traj (drawn after time panels so we can use their xlim when tying) ----
    if bool(sup_2d_plot_colorbar) and (ax_sup_traj is not None):
        try:
            cmap_name = sup_2d_kwargs_.get('cmap_name', 'plasma')
            ref_time_ax = ax_sup_dyn if (ax_sup_dyn is not None) else (ax_uns_lat if (ax_uns_lat is not None) else ax_uns_dyn)
            tie_time = bool(sup_2d_cbar_tie_time_axis) and (ref_time_ax is not None)
            if tie_time:
                # same position/width as time panel and same xlim so bar aligns with plotting part of time axis
                pos_time = ref_time_ax.get_position()
                t_lo, t_hi = ref_time_ax.get_xlim()
                pos_traj = ax_sup_traj.get_position()
                thickness = 0.015 if sup_2d_cbar_thickness is None else float(sup_2d_cbar_thickness)
                gap = 0.008
                y0 = pos_traj.y0 - thickness - gap
                cax = fig.add_axes([pos_time.x0, y0, pos_time.width, thickness])
                cax.set_xlim(t_lo, t_hi)
                cax.set_ylim(0, 1)
                grad = np.linspace(0, 1, 256).reshape(1, -1)
                cax.imshow(grad, aspect='auto', cmap=plt.get_cmap(cmap_name), extent=[t_lo, t_hi, 0, 1], interpolation=None)
                cax.set_yticks([])
                cax.set_xticks([t_lo, t_hi])
                cax.set_xticklabels(['early', 'late'])
                cax.tick_params(axis='x', which='both', length=0)
                cax.spines[:].set_visible(False)
            else:
                norm = mpl.colors.Normalize(0, 1)
                sm = mpl.cm.ScalarMappable(cmap=plt.get_cmap(cmap_name), norm=norm)
                sm.set_array([])
                bbox = list(tuple(sup_2d_cbar_bbox))
                if len(bbox) != 4:
                    bbox = [0, -0.04, 1, 0.10]
                if sup_2d_cbar_thickness is not None:
                    bbox[3] = float(sup_2d_cbar_thickness)
                cax = inset_locator.inset_axes(
                    ax_sup_traj,
                    width='100%',
                    height='100%',
                    loc='lower center',
                    bbox_to_anchor=tuple(bbox),
                    bbox_transform=ax_sup_traj.transAxes,
                    borderpad=0,
                )
                cbar = fig.colorbar(sm, cax=cax, orientation='horizontal')
                cbar.set_ticks([0.0, 1.0])
                cbar.set_ticklabels(['early', 'late'])
                cbar.ax.tick_params(axis='x', which='both', length=0)
                cbar.outline.set_visible(False)
        except Exception:
            pass
    if bool(time_scalebar) and block_sec is not None and block_sec > 0 and binsize_ms is not None and len(bottom_time_axs) > 0:
        trans_blend = mtransforms.blended_transform_factory
        y_bar = -0.05
        y_label = -0.10
        for ax in bottom_time_axs:
            try:
                x0, x1 = ax.get_xlim()
                trans = trans_blend(ax.transData, ax.transAxes)
                ax.plot([x0, x0 + float(block_sec)], [y_bar, y_bar], color='k', lw=1.5, transform=trans, clip_on=False)
                ax.text(x0 + float(block_sec) / 2.0, y_label, f'{binsize_ms} ms', transform=trans, ha='center', va='top', fontsize=8, clip_on=False)
            except Exception:
                pass

    # ---- title ----
    title_l = []
    if st is not None:
        t_str = _sec_to_hms_str(st, decimals=2)
        if dur is None:
            title_l.append(f'event={event_i}  start={t_str}')
        else:
            title_l.append(f'event={event_i}  start={t_str}  dur={dur:.3g}s')
    else:
        title_l.append(f'event={event_i}')

    if replay_res_sup is not None:
        parts = []
        if sup_sig is not None:
            parts.append(f'sig={bool(sup_sig)}')
        if sup_max_disp is not None:
            parts.append(f'max_disp={sup_max_disp:.3g}')
        if len(parts):
            title_l.append('SUP: ' + ', '.join(parts))

    if replay_res_unsup is not None:
        parts = []
        if unsup_sig is not None:
            parts.append(f'sig={bool(unsup_sig)}')
        if best_kernel is not None:
            parts.append(f'best={best_kernel}')
        if probs is not None:
            try:
                prob_str = ', '.join([f'{k}:{float(v):.2f}' for k, v in probs.items()])
                prob_str = textwrap.fill(prob_str, width=int(title_wrap_width), subsequent_indent='  ')
                parts.append('prob=' + prob_str)
            except Exception:
                pass
        if len(parts):
            title_l.append('UNSUP: ' + ', '.join(parts))

    title = '\n'.join(title_l)
    try:
        fig.suptitle(title, fontsize=float(title_fontsize), x=0.01, ha='left', y=1.02)
    except Exception:
        pass

    try:
        plt.tight_layout()
    except Exception:
        pass

    out = dict(
        event_i=int(event_i),
        start_time=st,
        end_time=ed,
        duration=dur,
        sup_sig=sup_sig,
        sup_max_displacement=sup_max_disp,
        unsup_sig=unsup_sig,
        best_kernel=best_kernel,
        probs=None if probs is None else dict(probs),
        dynamics_col=dynamics_col,
        sup_dynamics_plot=sup_dyn_plot_info,
        unsup_dynamics_plot=unsup_dyn_plot_info,
    )

    return fig, out_axs, out

