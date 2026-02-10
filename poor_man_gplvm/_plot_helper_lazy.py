'''
"lazy" because the input to the functions here tend to be high level, aggregated data
More for my own convenience and harder to use for others...
'''

import numpy as np
import textwrap

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

import pynapple as nap

import poor_man_gplvm.plot_helper as ph


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


def plot_replay_sup_unsup_event(
    replay_res_sup=None,
    replay_res_unsup=None,
    event_i=0,
    *,
    position_tsdf=None,
    maze='familiar',
    plot_sup_trajectory=True,
    plot_unsup_latent=True,
    plot_unsup_dynamics=True,
    fig=None,
    axs=None,
    figsize=(10, 4),
    title_fontsize=10,
    title_wrap_width=110,
    sup_2d_kwargs=None,
    unsup_heatmap_kwargs=None,
    heatmap_add_scatter_latent=True,
    heatmap_add_scatter_dynamics=False,
):
    """
    Plot replay for a single event from supervised + unsupervised pipelines.

    - Supervised: 2D posterior + MAP trajectory overlay (requires `position_tsdf` if enabled).
    - Unsupervised: 1D heatmaps of latent posterior and dynamics posterior for the **best** transition-kernel
      (argmax prob per event).

    Returns
    -------
    fig, axs, out_dict
      axs: list of matplotlib Axes in plotting order (sup, unsup_latent, unsup_dynamics; some may be missing)
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
    plot_unsup_latent=True,
    plot_unsup_dynamics=True,
    sup_2d_kwargs=dict(cmap_name='plasma', binarize_thresh=0.01),
    unsup_heatmap_kwargs=dict(heatmap_scatter_s=2, heatmap_scatter_c='yellow'),
)
```
    """
    mpl.rcParams['pdf.fonttype'] = 42
    mpl.rcParams['svg.fonttype'] = 'none'

    sup_2d_kwargs_ = {} if sup_2d_kwargs is None else dict(sup_2d_kwargs)
    unsup_heatmap_kwargs_ = {} if unsup_heatmap_kwargs is None else dict(unsup_heatmap_kwargs)

    event_i = int(event_i)

    has_sup = bool(plot_sup_trajectory) and (replay_res_sup is not None)
    has_unsup = (replay_res_unsup is not None) and (bool(plot_unsup_latent) or bool(plot_unsup_dynamics))
    n_unsup_panel = int(bool(plot_unsup_latent) and (replay_res_unsup is not None)) + int(bool(plot_unsup_dynamics) and (replay_res_unsup is not None))

    if has_sup and (position_tsdf is None):
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

    # ---- axes allocation ----
    axs_flat = _flatten_axs(axs)
    created_fig = False
    out_axs = []

    if axs_flat is not None:
        idx = 0
        ax_sup = None
        ax_lat = None
        ax_dyn = None
        if has_sup:
            if idx >= len(axs_flat):
                raise ValueError('Provided axs does not have enough axes for requested panels.')
            ax_sup = axs_flat[idx]
            idx += 1
        if bool(plot_unsup_latent) and (replay_res_unsup is not None):
            if idx >= len(axs_flat):
                raise ValueError('Provided axs does not have enough axes for requested panels.')
            ax_lat = axs_flat[idx]
            idx += 1
        if bool(plot_unsup_dynamics) and (replay_res_unsup is not None):
            if idx >= len(axs_flat):
                raise ValueError('Provided axs does not have enough axes for requested panels.')
            ax_dyn = axs_flat[idx]
            idx += 1

        if fig is None:
            fig = (ax_sup or ax_lat or ax_dyn).figure
    else:
        if fig is None:
            created_fig = True
            if has_sup and n_unsup_panel > 0:
                nrows = int(max(1, n_unsup_panel))
                fig = plt.figure(figsize=figsize, constrained_layout=True)
                gs = gridspec.GridSpec(nrows=nrows, ncols=2, figure=fig, width_ratios=[1.35, 1.0])
                ax_sup = fig.add_subplot(gs[:, 0])
                ax_lat = None
                ax_dyn = None
                r = 0
                if bool(plot_unsup_latent) and (replay_res_unsup is not None):
                    ax_lat = fig.add_subplot(gs[r, 1])
                    r += 1
                if bool(plot_unsup_dynamics) and (replay_res_unsup is not None):
                    if ax_lat is None:
                        ax_dyn = fig.add_subplot(gs[r, 1])
                    else:
                        ax_dyn = fig.add_subplot(gs[r, 1], sharex=ax_lat)
            elif has_sup:
                fig, ax_sup = plt.subplots(figsize=figsize, constrained_layout=True)
                ax_lat = None
                ax_dyn = None
            elif n_unsup_panel > 0:
                fig, ax_arr = plt.subplots(nrows=n_unsup_panel, ncols=1, figsize=figsize, constrained_layout=True, sharex=True)
                ax_arr = np.atleast_1d(ax_arr).ravel()
                ax_sup = None
                ax_lat = None
                ax_dyn = None
                k = 0
                if bool(plot_unsup_latent) and (replay_res_unsup is not None):
                    ax_lat = ax_arr[k]
                    k += 1
                if bool(plot_unsup_dynamics) and (replay_res_unsup is not None):
                    ax_dyn = ax_arr[k]
            else:
                fig, ax0 = plt.subplots(figsize=figsize, constrained_layout=True)
                ax0.text(0.5, 0.5, 'No panels to plot', ha='center', va='center', transform=ax0.transAxes)
                ax0.set_axis_off()
                ax_sup = ax0
                ax_lat = None
                ax_dyn = None
        else:
            created_fig = False
            ax_sup = None
            ax_lat = None
            ax_dyn = None

    if has_sup and ax_sup is not None:
        decode_res = replay_res_sup.get('decode_res', None)
        if decode_res is not None:
            ph.plot_decode_res_posterior_and_map_2d(
                decode_res,
                maze,
                event_i,
                position_tsdf,
                fig=fig,
                ax=ax_sup,
                **sup_2d_kwargs_,
            )
        out_axs.append(ax_sup)

    # Unsupervised heatmaps (best kernel)
    if replay_res_unsup is not None:
        data_dict = {}
        add_scatter = {}
        if bool(plot_unsup_latent) and (post_latent_best is not None) and (ax_lat is not None):
            data_dict['unsup_latent_posterior'] = post_latent_best
            add_scatter['unsup_latent_posterior'] = bool(heatmap_add_scatter_latent)
        if bool(plot_unsup_dynamics) and (post_dyn_best is not None) and (ax_dyn is not None):
            data_dict['unsup_dynamics_posterior'] = post_dyn_best
            add_scatter['unsup_dynamics_posterior'] = bool(heatmap_add_scatter_dynamics)

        if len(data_dict):
            axs_use = []
            if 'unsup_latent_posterior' in data_dict:
                axs_use.append(ax_lat)
                out_axs.append(ax_lat)
            if 'unsup_dynamics_posterior' in data_dict:
                axs_use.append(ax_dyn)
                out_axs.append(ax_dyn)

            ph.plot_pynapple_data_mpl(
                data_dict,
                plot_title=True,
                add_scatter_to_heatmap=add_scatter,
                fig=fig,
                axs=axs_use,
                **unsup_heatmap_kwargs_,
            )

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
    )

    return fig, out_axs, out

