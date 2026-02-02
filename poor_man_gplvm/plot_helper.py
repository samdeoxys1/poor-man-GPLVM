'''
helper functions for plotting
'''

import numpy as np
import matplotlib.pyplot as plt
import pynapple as nap
import sys,os
import pandas as pd

import plotly.graph_objects as go
import plotly.io as pio
pio.renderers.default='notebook'
from plotly.subplots import make_subplots
from scipy.stats import wilcoxon
import seaborn as sns


plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['svg.fonttype'] = 'none'
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['image.interpolation'] = 'nearest'



def save_fig(fig,fig_name,fig_dir='./figs',fig_format=['png','svg'],dpi=300,do_close=False,bbox_inches=None, text_to_save=None):
    '''
    save figure to fig_dir
    '''
    # clean fig_name to avoid special characters
    fig_name = fig_name.replace(' ','_')
    fig_name = fig_name.replace('.','__')
    if not os.path.exists(fig_dir):
        os.makedirs(fig_dir)
    for fmt in fig_format:
        fig.savefig(os.path.join(fig_dir,fig_name+f'.{fmt}'),dpi=dpi,bbox_inches=bbox_inches,transparent=False)
        print(f'saved {fig_name}.{fmt} to {fig_dir}')
    if text_to_save is not None:
        try:
            if not isinstance(text_to_save, str):
                import pprint
                text_to_save = pprint.pformat(text_to_save, width=120, compact=False, sort_dicts=False)
            txt_path = os.path.join(fig_dir, fig_name + '.txt')
            with open(txt_path, 'w') as f:
                f.write(text_to_save)
            print(f'saved {fig_name}.txt to {fig_dir}')
        except Exception as e:
            print(f'failed to save {fig_name}.txt to {fig_dir}: {e}')
    if do_close:
        plt.close(fig)

def save_fig_plotly(fig,fig_name,fig_dir='./figs',fig_format=['png','svg'],scale_png=10,scale_svg=0.15):
    '''
    save figure to fig_dir
    '''
    fig_name = fig_name.replace(' ','_')
    fig_name = fig_name.replace('.','__')
    if not os.path.exists(fig_dir):
        os.makedirs(fig_dir)
    for fmt in fig_format:
        if fmt =='html':
            fig.write_html(os.path.join(fig_dir,fig_name+f'.{fmt}'),include_mathjax='cdn')
        else:
            scale = scale_png if fmt == 'png' else scale_svg    
            fig.write_image(os.path.join(fig_dir,fig_name+f'.{fmt}'),scale=scale)
        print(f'saved {fig_name}.{fmt} to {fig_dir}')
    return fig

def plot_mean_error_plot(data,error_type='std',mean_axis=0,fig=None,ax=None,**kwargs):
    '''
    plt the mean and error of the data
    data: pd.DataFrame or np.ndarray
    error_type: 'ci' or 'std'
    mean_axis: axis to take the mean of, same for error; plot the other axis
    '''
    if ax is None:
        fig,ax = plt.subplots()
    if error_type == 'ci':
        mean = np.mean(data,axis=mean_axis)
        error = np.std(data,axis=mean_axis) / np.sqrt(data.shape[mean_axis])
    elif error_type == 'std':
        mean = np.mean(data,axis=mean_axis)
        error = np.std(data,axis=mean_axis)
    else:
        raise ValueError(f'error_type {error_type} not supported')
    ax.plot(mean,**kwargs)
    if isinstance(data,pd.DataFrame):
        xs = mean.index
    else:
        xs = np.arange(len(mean))
    ax.fill_between(xs,mean-error,mean+error,alpha=0.5,**kwargs)
    return fig,ax

import numpy as np
import pynapple as nap
import plotly.graph_objects as go
from plotly.subplots import make_subplots

def _deep_update(base: dict, extra: dict):
    """Recursively update nested dicts (for kwargs like line/marker/etc.)."""
    for k, v in (extra or {}).items():
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            _deep_update(base[k], v)
        else:
            base[k] = v
    return base

def _per_key(value, key):
    if isinstance(value, dict):
        return value.get(key, None)
    return value


def _compute_tickvals(requested, vmin, vmax):
    if requested is None:
        return None
    if isinstance(requested, int):
        n = max(1, requested)
        if not np.isfinite(vmin) or not np.isfinite(vmax):
            return None
        if vmin == vmax:
            return [vmin]
        return list(np.linspace(vmin, vmax, n))
    return requested

def plot_pynapple_data_plotly(
    data_dict: dict,
    reference_time_key=None,
    width=900,
    heights=200,                 # int OR list of pixel heights (one per subplot)
    global_scale = 1., # scale all size by the same factor
    vertical_spacing=0.04,       # increase a bit to avoid title overlap
    styles: dict | None = None,  # {key: {kwargs for go.Scatter/Heatmap}}
    x_nticks: int | None = None, # or dict: {key: int}
    y_nticks: int | None = None, # or dict: {key: int}
    tickformat: str | None = None,   # e.g. '%H:%M:%S' for time axes
    y_lim_quantile: tuple[float, float] | dict[str, tuple[float, float] | None] | None = (0.01, 0.99),
    y_lim: tuple[float, float] | dict[str, tuple[float, float] | None] | None = None, # only effective if y_lim_quantile is not None / not provided
    ylabel: str | dict[str, str] | None = None,
    xlabel: str | dict[str, str] | None = None,
    tickvals: list[float] | int | dict[str, list[float] | int | None] | None = None,
    ticktext: list[str] | dict[str, list[str] | None] | None = None,
    ylabel_standoff: float | dict[str, float] | None = None,
    xlabel_standoff: float | dict[str, float] | None = None,
    title_top_margin=70,         # extra top space so titles never clip
    annotation_yshift=8,         # lift subplot titles a bit (pixels)
    shared_vlines: list[float] | None = None,  # x positions
    showlegend=False,
    font_size=12,
):
    """
    Plot dict of pynapple objects 
    1D -> line; 2D -> heatmap (time on x, rows on y).

    reference_time_key: key of the data_dict to use as reference time; all other data will be restricted to this interval
    """
    # --- pick a common interval ---
    if reference_time_key is not None:
        ref = data_dict[reference_time_key]
        common_interval = nap.IntervalSet([ref.t[0], ref.t[-1]])
    else:
        st = max(np.min(arr.t) for arr in data_dict.values())
        ed = min(np.max(arr.t) for arr in data_dict.values())
        common_interval = nap.IntervalSet([st, ed])

    # restrict
    data = {k: v.restrict(common_interval) for k, v in data_dict.items()}
    keys = list(data.keys())
    n = len(keys)

    if isinstance(heights, list):
        heights = np.array(heights)
    heights = heights * global_scale
    width = width * global_scale
    vertical_spacing = vertical_spacing * global_scale
    title_top_margin = title_top_margin * global_scale
    annotation_yshift = annotation_yshift * global_scale

    # --- per-row heights (pixels -> relative row_heights) & total height ---
    if isinstance(heights, (list, tuple, np.ndarray)):
        fig_height = int(np.sum(heights))
        row_heights = [h / float(np.sum(heights)) for h in heights]
        if len(row_heights) != n:
            raise ValueError("len(heights) must equal number of subplots.")
    else:
        fig_height = int(n * heights)
        row_heights = [1/n] * n

    # --- build subplots ---
    fig = make_subplots(
        rows=n, cols=1, shared_xaxes=True, vertical_spacing=vertical_spacing,
        subplot_titles=keys, row_heights=row_heights
    )

    # --- draw traces ---
    for i, k in enumerate(keys, start=1):
        arr = data[k]
        t = arr.t
        d = arr.d

        # empty pane still needs a dummy trace so the title shows up
        if d.size == 0:
            fig.add_trace(go.Scatter(x=[], y=[], mode='lines', name=k), row=i, col=1)
            continue

        # Defaults, then per-key overrides via styles[k]
        sty = (styles or {}).get(k, {})
        if d.ndim == 1:
            defaults = dict(mode='lines', name=k)
            _deep_update(defaults, sty)
            fig.add_trace(go.Scatter(x=t, y=d, **defaults), row=i, col=1)
        elif d.ndim == 2:
            d_plot = d.T
            # robust zmin/zmax if not supplied
            defaults = dict(
                x=t,
                y=getattr(arr, "columns", np.arange(d_plot.shape[0])),
                z=d_plot,
                colorscale='Viridis',
                showscale=False,
                zauto=False
            )
            if "zmin" not in sty:
                defaults["zmin"] = float(np.nanquantile(d_plot, 0.01))
            if "zmax" not in sty:
                defaults["zmax"] = float(np.nanquantile(d_plot, 0.99))
            _deep_update(defaults, sty)
            fig.add_trace(go.Heatmap(**defaults), row=i, col=1)
        else:
            raise ValueError(f"Unsupported dim for key '{k}': {d.ndim}")

    # --- axis ranges & tick control ---
    for i, k in enumerate(keys, start=1):
        arr = data[k]
        y_min_used = None
        y_max_used = None
        if arr.d.size and arr.d.ndim == 1:
            y = arr.d
            # Determine quantile config for this subplot
            q_cfg = y_lim_quantile
            if isinstance(y_lim_quantile, dict):
                q_cfg = y_lim_quantile.get(k, (0.01, 0.99))
            if q_cfg is not None:
                q_low, q_high = q_cfg
                ymin = float(np.nanquantile(y, q_low))
                ymax = float(np.nanquantile(y, q_high))
                if not np.isfinite(ymin) or not np.isfinite(ymax) or ymin == ymax:
                    ymin = float(np.nanmin(y))
                    ymax = float(np.nanmax(y))
                    if ymin == ymax:
                        eps = 1e-6 if ymin == 0 else abs(ymin) * 1e-6
                        ymin -= eps
                        ymax += eps
                fig.update_yaxes(range=[ymin, ymax], row=i, col=1)
                y_min_used, y_max_used = ymin, ymax
            else:
                # Try explicit y_lim next
                lim_cfg = y_lim
                if isinstance(y_lim, dict):
                    lim_cfg = y_lim.get(k, None)
                if lim_cfg is not None:
                    ymin, ymax = lim_cfg
                    # validate
                    if not (np.isfinite(ymin) and np.isfinite(ymax)) or ymin == ymax:
                        ymin = float(np.nanmin(y))
                        ymax = float(np.nanmax(y))
                        if ymin == ymax:
                            eps = 1e-6 if ymin == 0 else abs(ymin) * 1e-6
                            ymin -= eps
                            ymax += eps
                    fig.update_yaxes(range=[float(ymin), float(ymax)], row=i, col=1)
                    y_min_used, y_max_used = float(ymin), float(ymax)
                else:
                    # no override -> fall back to raw data extent for tick computations
                    y_min_used = float(np.nanmin(y))
                    y_max_used = float(np.nanmax(y))

        # ticks: global int OR per-key dict
        if isinstance(x_nticks, dict):
            nt = x_nticks.get(k, None)
            if nt: fig.update_xaxes(nticks=nt, row=i, col=1)
        elif isinstance(x_nticks, int):
            fig.update_xaxes(nticks=x_nticks, row=i, col=1)

        if isinstance(y_nticks, dict):
            nt = y_nticks.get(k, None)
            if nt: fig.update_yaxes(nticks=nt, row=i, col=1)
        elif isinstance(y_nticks, int):
            fig.update_yaxes(nticks=y_nticks, row=i, col=1)

        if tickformat:
            fig.update_xaxes(tickformat=tickformat, row=i, col=1)

        # --- labels, tickvals/ticktext, standoff ---
        # y-axis
        ylab = _per_key(ylabel, k)
        ystandoff = _per_key(ylabel_standoff, k)
        y_tick_req = _per_key(tickvals, k)
        y_tick_text = _per_key(ticktext, k)
        y_update = {}
        if ylab is not None:
            y_update["title"] = ylab
        if ystandoff is not None:
            y_update["title_standoff"] = ystandoff
        if y_tick_req is not None:
            # If no prior range calc, use data extents when available
            if y_min_used is None or y_max_used is None:
                if arr.d.size and arr.d.ndim == 1:
                    y_min_used = float(np.nanmin(arr.d))
                    y_max_used = float(np.nanmax(arr.d))
            tv = _compute_tickvals(y_tick_req, y_min_used, y_max_used)
            if tv is not None:
                y_update["tickmode"] = "array"
                y_update["tickvals"] = tv
                if y_tick_text is not None:
                    y_update["ticktext"] = y_tick_text
        if y_update:
            fig.update_yaxes(**y_update, row=i, col=1)

        # x-axis
        xlab = _per_key(xlabel, k)
        xstandoff = _per_key(xlabel_standoff, k)
        x_update = {}
        if xlab is not None:
            x_update["title"] = xlab
        if xstandoff is not None:
            x_update["title_standoff"] = xstandoff
        if x_update:
            fig.update_xaxes(**x_update, row=i, col=1)

    # hide x tick labels except bottom row (cleaner stacked look)
    for i in range(1, n):
        fig.update_xaxes(showticklabels=False, row=i, col=1)

    # --- shared vertical lines across all subplots ---
    if shared_vlines:
        for x0 in shared_vlines:
            # In recent Plotly, row="all" works; fallback draws one per row.
            try:
                fig.add_vline(x=x0, row="all", col=1, line_dash="dash", line_width=1)
            except TypeError:
                for i in range(1, n+1):
                    fig.add_vline(x=x0, row=i, col=1, line_dash="dash", line_width=1)

    # --- layout & titles/annotations spacing ---
    fig.update_layout(
        width=width,
        height=fig_height,
        showlegend=showlegend,
        margin=dict(t=title_top_margin, r=10, b=10, l=10),
    )
    # lift subplot titles slightly so they never collide with plots
    fig.for_each_annotation(lambda a: a.update(yshift=annotation_yshift))

    set_plotly_fonts(fig,base=font_size)

    return fig

# --- saving ---
# fig = plot_pynapple_data_plotly(...)
# 1) Static images (SVG/PNG) -> needs kaleido installed: pip install -U kaleido
# fig.write_image("figure.svg", scale=1)    # vector, publication-ready
# fig.write_image("figure.png", scale=2)    # raster, higher DPI via scale
# 2) HTML for interactive exploration:
# fig.write_html("figure.html", include_mathjax="cdn")
# 3) Show as non-interactive in notebook:
# fig.show(config={"staticPlot": True})

def add_vertical_shades(fig,intvl_l,ep=None,*,exclude=None,fillcolor="red",opacity=0.25,line_width=0,line_dash=None,layer="above",**vrect_kwargs):
    '''
    shade the intervals in the figure

    fig: plotly figure
    intvl_l: nap.IntervalSet, the intervals to be shaded
    ep: nap.IntervalSet, with one row, to restrict the intvl_l

    exclude: iterable of (row, col) pairs (1-based) specifying subplot coordinates to NOT shade
    fillcolor, opacity, line_width, line_dash, layer: styling controls for the shaded rectangles
    Additional keyword arguments are passed through to fig.add_vrect
    '''

    # restrict intervals to epoch if provided
    if ep is None:
        intvl_l_sub = intvl_l
    else:
        intv_ts=nap.Ts(intvl_l['start'])
        ma = np.logical_not(np.isnan(ep.in_interval(intv_ts)))
        intvl_l_sub = intvl_l[ma]

    # infer subplot grid shape
    row_l,col_l = fig._get_subplot_rows_columns() # ranges

    # normalize exclude list into a set of tuples
    exclude_set = set()
    if exclude is not None:
        exclude_set = { (int(rc[0]), int(rc[1])) for rc in exclude }

    # draw a vrect per interval on subplots not in exclude_set
    for intv in intvl_l_sub:
        for i in row_l:
            for j in col_l:
                if (i, j) in exclude_set:
                    continue
                args = dict(
                    x0=intv['start'][0],
                    x1=intv['end'][0],
                    row=i,
                    col=j,
                    fillcolor=fillcolor,
                    opacity=opacity,
                    line_width=line_width,
                    layer=layer,
                )
                if line_dash is not None:
                    args['line_dash'] = line_dash
                args.update(vrect_kwargs)
                fig.add_vrect(**args)
    return fig


def add_vertical_shades_mpl(fig,intvl_l,ep=None,*,exclude=None,color="red",alpha=0.25,linewidth=0,linestyle=None,zorder=0,mode='span',**span_kwargs):
    '''
    Shade intervals on a Matplotlib figure (all subplots by default).

    fig: matplotlib.figure.Figure
    intvl_l: nap.IntervalSet, the intervals to be shaded
    ep: nap.IntervalSet, with one row, to restrict the intvl_l

    exclude: iterable of (row, col) pairs (1-based) specifying subplot coordinates to NOT shade
    color, alpha, linewidth, linestyle, zorder: Matplotlib styling controls (passed to axvspan/axvline)
    mode: 'span' (default) for colored rectangles, 'lines' for vertical lines at interval boundaries
    Additional keyword arguments are passed through to ax.axvspan or ax.axvline
    '''

    # restrict intervals to epoch if provided
    if ep is None:
        intvl_l_sub = intvl_l
    else:
        intv_ts=nap.Ts(intvl_l['start'])
        ma = np.logical_not(np.isnan(ep.in_interval(intv_ts))) # careful ep can have multiple rows!!!
        intvl_l_sub = intvl_l[ma]

    # collect (x0, x1) pairs
    intervals = []
    for intv in intvl_l_sub:
        x0 = float(intv['start'][0])
        x1 = float(intv['end'][0])
        intervals.append([x0, x1])

    # normalize exclude list into a set of tuples
    exclude_set = set()
    if exclude is not None:
        exclude_set = { (int(rc[0]), int(rc[1])) for rc in exclude }

    # find axes to shade; respect subplot positions when available
    axes_to_shade = []
    all_axes = fig.get_axes()
    for ax in all_axes:
        spec = getattr(ax, 'get_subplotspec', None)
        if not callable(spec):
            # No get_subplotspec method: skip (likely not a standard subplot)
            continue
        ss = ax.get_subplotspec()
        if ss is None:
            # No subplotspec: skip (likely colorbar or other auxiliary axis)
            continue
        # Check if this subplot is in the exclude set
        rows = range(ss.rowspan.start + 1, ss.rowspan.stop + 1)
        cols = range(ss.colspan.start + 1, ss.colspan.stop + 1)
        skip = False
        for r in rows:
            for c in cols:
                if (r, c) in exclude_set:
                    skip = True
                    break
            if skip:
                break
        if not skip:
            axes_to_shade.append(ax)

    if len(intervals) == 0 or len(axes_to_shade) == 0:
        return fig

    # draw spans or lines on selected axes
    for ax in axes_to_shade:
        for x0, x1 in intervals:
            if mode == 'lines':
                # Draw vertical lines at interval boundaries
                lw = linewidth if linewidth else 1
                ls = linestyle if linestyle else ':'
                ax.axvline(x0, color=color, alpha=alpha, linewidth=lw, linestyle=ls, zorder=zorder, **span_kwargs)
                if x0 != x1:
                    ax.axvline(x1, color=color, alpha=alpha, linewidth=lw, linestyle=ls, zorder=zorder, **span_kwargs)
            else:  # mode == 'span'
                if x0 == x1:
                    ax.axvline(x0, color=color, alpha=alpha, linewidth=max(1, linewidth or 1), linestyle=linestyle, zorder=zorder, **span_kwargs)
                else:
                    ax.axvspan(x0, x1, color=color, alpha=alpha, linewidth=linewidth, linestyle=linestyle, zorder=zorder, **span_kwargs)

    return fig


import plotly.graph_objects as go

def set_plotly_fonts(fig,
    base=12,              # global default (ticks if not overridden)
    title=None,           # figure title
    subplot_title=None,   # titles from make_subplots (annotations)
    axis_title=None,      # x/y axis titles
    ticks=None,           # x/y tick labels
    legend=None,
    colorbar_title=None,
    colorbar_ticks=None,
    hover=None
):
    # sensible defaults relative to base
    title           = title           or int(base*1.2)
    subplot_title   = subplot_title   or int(base*1.1)
    axis_title      = axis_title      or int(base*1.0)
    ticks           = ticks           or int(base*0.9)
    legend          = legend          or int(base*0.9)
    colorbar_title  = colorbar_title  or int(base*0.9)
    colorbar_ticks  = colorbar_ticks  or int(base*0.85)
    hover           = hover           or int(base*0.9)

    # global default for any text without its own font set
    fig.update_layout(font=dict(size=base))

    # figure title
    fig.update_layout(title_font=dict(size=title))

    # all axis ticks & titles at once
    fig.update_xaxes(tickfont=dict(size=ticks), title_font=dict(size=axis_title), row="all", col="all")
    fig.update_yaxes(tickfont=dict(size=ticks), title_font=dict(size=axis_title), row="all", col="all")

    # subplot titles (are annotations)
    if getattr(fig.layout, "annotations", None):
        fig.update_annotations(font=dict(size=subplot_title))

    # legend
    fig.update_layout(legend=dict(font=dict(size=legend)))

    # colorbars on traces (heatmap/contour/hist2d/etc.)
    for tr in fig.data:
        cb = getattr(tr, "colorbar", None)
        if cb is not None:
            tr.update(colorbar=dict(
                titlefont=dict(size=colorbar_title),
                tickfont=dict(size=colorbar_ticks)
            ))

    # hover labels (if you keep interactivity)
    fig.update_layout(hoverlabel=dict(font=dict(size=hover)))

    return fig

def _round_to_first_distinguishing_digit(a, b):
    """
    Round two numbers to the precision of the first non-zero digit
    that distinguishes them.

    Examples:
    - a=0.0123, b=0.0345 -> precision 2 -> (0.01, 0.03)
    - a=1234, b=5678 -> precision -3 -> (1000, 6000)
    """
    try:
        a = float(a)
        b = float(b)
    except Exception:
        return a, b

    if not (np.isfinite(a) and np.isfinite(b)):
        return a, b

    diff = abs(b - a)
    if diff == 0:
        return a, b

    order = np.floor(np.log10(diff))
    precision = int(-order)
    # Python round supports negative precision for rounding to tens/hundreds/etc.
    ra = round(a, precision)
    rb = round(b, precision)
    return ra, rb

# for small plots, only keep two ticks 
def set_two_ticks(axis, xlim=None, ylim=None, do_int=False, apply_to='y'):
    """
    Set exactly two ticks on the specified axis/axes.

    apply_to: 'x', 'y', or 'both'
    """
    def _compute_two(lim, getlim):
        explicit = lim is not None
        if lim is None:
            lim = getlim()
        lo, hi = lim
        if do_int:
            lo_i = int(np.floor(lo))
            hi_i = int(np.ceil(hi))
            if lo_i == hi_i:
                hi_i = lo_i + 1
            return [lo_i, hi_i]
        else:
            if explicit:
                return [lo, hi]
            if lo == hi:
                eps = 1e-6 if lo == 0 else abs(lo) * 1e-6
                lo -= eps
                hi += eps
            # ensure lo <= hi before rounding
            if lo > hi:
                lo, hi = hi, lo
            lo_r, hi_r = _round_to_first_distinguishing_digit(lo, hi)
            if lo_r == hi_r:
                eps = 1e-6 if lo_r == 0 else abs(lo_r) * 1e-6
                lo_r -= eps
                hi_r += eps
            return [lo_r, hi_r]

    if apply_to in ('y', 'both'):
        y_ticks = _compute_two(ylim, axis.get_ylim)
        axis.set_yticks([y_ticks[0], y_ticks[1]])
        if ylim is not None:
            axis.set_ylim(ylim)
    if apply_to in ('x', 'both'):
        x_ticks = _compute_two(xlim, axis.get_xlim)
        axis.set_xticks([x_ticks[0], x_ticks[1]])
        if xlim is not None:
            axis.set_xlim(xlim)
    return axis

# New: symmetric ticks around 0 (e.g., [-M, 0, M])
def set_symmetric_ticks(axis, xlim=None, ylim=None, do_int=False, apply_to='y'):
    """
    Set symmetric ticks around 0 on the specified axis/axes: [-M, 0, M].

    - Limits are taken from xlim/ylim if provided, otherwise from the current axis limits
    - M is max(abs(lo), abs(hi)) (ceil if do_int)
    - apply_to: 'x', 'y', or 'both'
    """
    def _compute_three(lim, getlim):
        explicit = lim is not None
        if lim is None:
            lim = getlim()
        lo, hi = lim
        if do_int:
            M = int(np.floor(min(abs(lo), abs(hi))))
            if M == 0:
                M = 1
            return [-M, 0, M]
        else:
            M = float(min(abs(lo), abs(hi)))
            if explicit:
                M_r = M
            else:
                # Round to the first distinguishing digit of limit ticks (-M, M)
                ml, mh = _round_to_first_distinguishing_digit(-M, M)
                M_r = max(abs(ml), abs(mh))
                if M_r == 0:
                    eps = 1e-6 if lo == 0 else abs(lo) * 1e-6
                    M_r = eps
            return [-M_r, 0.0, M_r]

    if apply_to in ('y', 'both'):
        y_ticks = _compute_three(ylim, axis.get_ylim)
        axis.set_yticks(y_ticks)
        if ylim is not None:
            axis.set_ylim(ylim)
    if apply_to in ('x', 'both'):
        x_ticks = _compute_three(xlim, axis.get_xlim)
        axis.set_xticks(x_ticks)
        if xlim is not None:
            axis.set_xlim(xlim)
    return axis

# for plotting the distribution of the shuffle data and the data itself
def plot_shuffle_data_dist_with_thresh(shuffle,data,bins=20,alpha=0.025,fig=None,ax=None,lw=4,plot_ci_high=True,plot_ci_low=False,figsize=(2,1.3)):
    thresh_high=np.quantile(shuffle,(1-alpha))
    percentile_high = (1-alpha) * 100
    # if plot_ci_low:
    thresh_low=np.quantile(shuffle,alpha)
    percentile_low = alpha * 100
    if ax is None:
        fig,ax=plt.subplots(figsize=figsize)
    ax.hist(shuffle,bins=bins,alpha=0.5)
    ax.axvline(data,label='data',linewidth=lw)
    if plot_ci_low:
        ax.axvline(thresh_low,label=f'{percentile_low:.02f} percentile',linestyle=':',linewidth=lw)
    if plot_ci_high:
        ax.axvline(thresh_high,label=f'{percentile_high:.02f} percentile',linestyle=':',linewidth=lw)
    ax.legend()
    return fig,ax


def subplots_wrapper(nplots,return_axs=True,basewidth=6,baseheight=4,figsize=None,**kwargs):
    nrows = int(np.sqrt(nplots))
    ncols = int(nplots // nrows)
    if nplots%nrows !=0:
        ncols+=1
    if figsize is None:
        figsize=(ncols*basewidth,nrows*baseheight)
    if return_axs:
        
        fig,axs = plt.subplots(nrows,ncols,figsize=figsize,**kwargs)
        return fig,axs
    else:
        fig = plt.figure(figsize=figsize)
        return fig, nrows, ncols


def plot_paired_line_median(
    df: pd.DataFrame,
    ticklabels: list | tuple | None = None,
    line_colors: list | tuple | str | None = None,
    line_styles: list | tuple | str | None = '-',
    line_widths: list | tuple | float | None = 1.5,
    line_alphas: list | tuple | float | None = 0.8,
    *,
    bar_color: str = '0.6',
    bar_edgecolor: str = 'k',
    bar_alpha: float = 0.5,
    bar_width: float = 0.5,
    show_sig: bool = True,
    sig_level: float = 0.05,
    alternative: str = 'two-sided',  # 'two-sided', 'greater', or 'less'
    fig=None,
    ax=None,
    figsize=(3, 3),
    text_coord = (0.02,0.98),
    text_fontsize: int = 10,
):
    """
    Plot paired data (n x 2 DataFrame): each row as a line connecting the two columns,
    and a bar for the median of each column. Also performs Wilcoxon signed-rank test,
    computes Cohen's d for paired differences, annotates significance stars if significant,
    and writes p-value and effect size on the figure.

    Parameters
    - df: n x 2 pandas DataFrame.
    - ticklabels: optional [label_col1, label_col2]; defaults to DataFrame column names.
    - line_colors, line_styles, line_widths, line_alphas: styling for connecting lines.
      Can be a single value (applied to all lines) or a list of length n. If colors is None,
      Matplotlib's default color cycle will be used for each line.
    - bar_color, bar_edgecolor, bar_alpha, bar_width: styling for the median bars.
    - show_sig: if True, draw significance bracket and stars when p < sig_level.
    - sig_level: alpha threshold for significance.
    - alternative: passed to scipy.stats.wilcoxon.
    - fig, ax, figsize: Matplotlib figure/axes or figure size if creating new.
    - text_fontsize: font size for the p-value/effect-size annotation.

    Returns
    - fig, ax, result_dict
      result_dict contains: { 'n': int, 'statistic': float, 'pvalue': float,
                              'effect_size': float, 'median_col1': float,
                              'median_col2': float, 'median_diff': float,
                              'stars': str }
    """
    if not isinstance(df, pd.DataFrame) or df.shape[1] != 2:
        raise ValueError('df must be a pandas DataFrame with exactly 2 columns')

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize,constrained_layout=True)

    # X positions for the two conditions
    x_positions = np.array([0.0, 1.0])

    # Tick labels
    if ticklabels is None:
        ticklabels = list(df.columns)
    if len(ticklabels) != 2:
        raise ValueError('ticklabels must have length 2')

    # Values and medians
    values = df.values.astype(float)
    num_rows = values.shape[0]
    medians = np.nanmedian(values, axis=0)

    # Helper: normalize styling input to per-row list
    def _to_list(value, default):
        if value is None:
            return [default] * num_rows
        if isinstance(value, (list, tuple, np.ndarray, pd.Series)):
            if len(value) == 0:
                return [default] * num_rows
            if len(value) == 1:
                return [value[0]] * num_rows
            if len(value) != num_rows:
                raise ValueError('When providing per-line styling, length must equal number of rows in df')
            return list(value)
        return [value] * num_rows

    # Build per-line styles
    color_list = None if line_colors is None else _to_list(line_colors, None)
    style_list = _to_list(line_styles, '-')
    width_list = _to_list(line_widths, 1.5)
    alpha_list = _to_list(line_alphas, 0.8)

    # Draw median as horizontal black line segments at each column position
    seg_half = bar_width/2.0
    for xi, mi in zip(x_positions, medians):
        ax.hlines(y=mi, xmin=xi-seg_half, xmax=xi+seg_half, colors='k', linewidth=2.0, zorder=3)

    # Draw paired lines for each row
    for i in range(num_rows):
        # When colors are None, let Matplotlib auto-cycle colors
        color_i = None if color_list is None else color_list[i]
        ax.plot(x_positions, values[i, :],
                color=color_i,
                linestyle=style_list[i],
                linewidth=width_list[i],
                alpha=alpha_list[i],
                zorder=2)

    # Axes cosmetics
    ax.set_xticks(x_positions)
    ax.set_xticklabels(ticklabels)

    # Compute y-limits with margin to make room for stars text
    y_min = float(np.nanmin(values)) if np.isfinite(np.nanmin(values)) else 0.0
    y_max = float(np.nanmax(values)) if np.isfinite(np.nanmax(values)) else 1.0
    if not np.isfinite(y_min) or not np.isfinite(y_max) or y_min == y_max:
        y_min, y_max = (0.0, 1.0)
    y_range = y_max - y_min
    top_margin = 0.18 * y_range
    bottom_margin = 0.05 * y_range
    ax.set_ylim(y_min - bottom_margin, y_max + top_margin)

    # Wilcoxon signed-rank test (drop NaNs and zeros for 'wilcox' zero_method)
    x = df.iloc[:, 0].to_numpy(dtype=float)
    y = df.iloc[:, 1].to_numpy(dtype=float)
    mask_finite = np.isfinite(x) & np.isfinite(y)
    x = x[mask_finite]
    y = y[mask_finite]
    diffs = y - x
    nonzero_mask = diffs != 0
    x_nz = x[nonzero_mask]
    y_nz = y[nonzero_mask]
    n_eff = int(len(x_nz))

    if n_eff >= 1:
        res = wilcoxon(x_nz, y_nz, alternative=alternative, zero_method='wilcox', correction=False, mode='auto')
        statistic = float(res.statistic)
        pvalue = float(res.pvalue)
        # Cohen's d for paired samples: mean(d) / sd(d)
        diffs_nz = y_nz - x_nz
        sd = np.nanstd(diffs_nz, ddof=1) if len(diffs_nz) > 1 else np.nan
        effect_size = float(np.nanmean(diffs_nz) / sd) if (sd is not None and sd > 0) else np.nan
    else:
        statistic = np.nan
        pvalue = 1.0
        effect_size = np.nan

    # Stars for significance
    def _p_to_stars(p):
        if p < 1e-4:
            return '****'
        if p < 1e-3:
            return '***'
        if p < 1e-2:
            return '**'
        if p < 5e-2:
            return '*'
        return ''

    stars = _p_to_stars(pvalue)

    # Draw bracket and stars if significant
    if show_sig and pvalue < sig_level:
        bracket_y = y_max + 0.08 * y_range
        tick_h = 0.03 * y_range
        # vertical ticks
        ax.plot([x_positions[0], x_positions[0]], [bracket_y - tick_h, bracket_y], color='k', linewidth=1, zorder=3)
        ax.plot([x_positions[1], x_positions[1]], [bracket_y - tick_h, bracket_y], color='k', linewidth=1, zorder=3)
        # horizontal line
        ax.plot([x_positions[0], x_positions[1]], [bracket_y, bracket_y], color='k', linewidth=1, zorder=3)
        # stars text centered
        ax.text(np.mean(x_positions), bracket_y + 0.01 * y_range, stars,
                ha='center', va='bottom', fontsize=text_fontsize, color='k')

    # Text annotation for p-value and effect size in axes coords (top-left)
    median_diff = float(np.nanmedian(df.iloc[:, 1] - df.iloc[:, 0]))
    text_str = f"p={pvalue:.3g}\nES={effect_size:.2f}\nn={n_eff}"
    ax.text(text_coord[0], text_coord[1], text_str, transform=ax.transAxes, ha='left', va='top', fontsize=text_fontsize)

    result = {
        'n': n_eff,
        'statistic': statistic,
        'pvalue': pvalue,
        'effect_size': effect_size,
        'median_col1': float(medians[0]),
        'median_col2': float(medians[1]),
        'median_diff': median_diff,
        'stars': stars,
    }

    sns.despine()
    set_two_ticks(ax,apply_to='y',do_int=False)
    plt.tight_layout()
    return fig, ax, result


import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import pynapple as nap
from matplotlib.colors import Normalize

def plot_pynapple_data_mpl(data_dict,  height_per_plot=3,width_per_plot=6,height_ratios=None,plot_title=False,add_scatter_to_heatmap=False,heatmap_scatter_s=0.05,heatmap_scatter_c='yellow',fig=None,axs=None):
    """
    Plot a dictionary of pynapple objects using matplotlib.
    
    Parameters:
        data_dict (dict): Dictionary where each key maps to a pynapple object.
                          Each object must have attributes `.t` (time) and `.d` (data)
                          and a method `.restrict(interval)`.
        figsize (tuple): Figure size (width, height) in inches.
        height_ratios (list): Optional list of relative heights for each subplot.
                              If None, equal heights are used.
        
        if add_scatter_to_heatmap:
            add scatter to heatmap on the max row within each column
            can be a dict
        
    Returns:
        fig (matplotlib.figure.Figure): The generated figure.
        axs (list): List of subplot axes.
    """
    # Compute common time range across all arrays
    min_times = [np.min(arr.t) for arr in data_dict.values() if not isinstance(arr,tuple)]
    max_times = [np.max(arr.t) for arr in data_dict.values() if not isinstance(arr,tuple)]
    
    st = np.max(min_times)
    ed = np.min(max_times)
    
    # Create a common time interval
    common_interval = nap.IntervalSet([st, ed])
    
    # Restrict each array to the common time range
    for key, arr in data_dict.items():
        if not isinstance(arr,tuple):
            data_dict[key] = arr.restrict(common_interval)
        else:
            tind,uind,c_l = arr
            ma = (tind < ed) & (tind > st)
            tind = tind[ma]
            uind = uind[ma]
            c_l = c_l[ma]
            data_dict[key] = (tind,uind,c_l)
    
    n_plots = len(data_dict)
    if isinstance(add_scatter_to_heatmap,bool):
        add_scatter_to_heatmap = {key:add_scatter_to_heatmap for key in data_dict.keys()}
    
    # Set up height ratios if not provided
    if height_ratios is None:
        height_ratios = [1] * n_plots
    
    # Create figure and subplots with gridspec for flexible heights
    if axs is None:
        fig = plt.figure(figsize=(width_per_plot,height_per_plot*n_plots),constrained_layout=True)
        gs = gridspec.GridSpec(n_plots, 1, height_ratios=height_ratios)
        axs = []
        need_to_create_ax=True
    else:
        need_to_create_ax=False
        
    for i, (key, arr) in enumerate(data_dict.items()):
        if need_to_create_ax:
            if i==0:
                ax = fig.add_subplot(gs[i])
            else:
                ax = fig.add_subplot(gs[i],sharex=axs[0])
            
            axs.append(ax)
        else:
            ax=axs[i]
        
        if isinstance(arr,tuple): # then it's for raster plot
            tind,uind,c_l = arr
            ax.scatter(tind,uind,c=c_l,cmap='Spectral_r',s=5)
        # Extract time and data
        else:
            t = arr.t
            d = arr.d
            
            # Check dimension of data and plot accordingly
            if d.ndim == 1:
                # 1D: Plot a line plot
                ax.plot(t, d, label=key)
                
                # Set robust y range by excluding outliers
                mu = np.nanmean(d)
                sigma = np.nanstd(d)
                if sigma > 0:
                    z = (d - mu) / sigma
                    threshold = 5  # adjust threshold as needed
                    filtered = d[np.abs(z) <= threshold]
                    if len(filtered) > 0:
                        ymin, ymax = np.min(filtered), np.max(filtered)
                        ax.set_ylim(ymin, ymax)
                
            elif d.ndim == 2:
                # 2D: Create a heatmap
                # For pynapple data, time is along the rows
                d_plot = d.T
                
                # Calculate robust color limits
                zmin = np.nanquantile(d_plot, 0.01)
                zmax = np.nanquantile(d_plot, 0.99)
                
                # Create heatmap
                im = ax.imshow(d_plot, aspect='auto', origin='lower', 
                            interpolation='none', 
                            extent=[np.min(t), np.max(t), 0, d_plot.shape[0]], 
                            norm=Normalize(vmin=zmin, vmax=zmax))
                
                # if add scatter to heatmap
                if add_scatter_to_heatmap.get(key,False):
                    map_ = d_plot.argmax(axis=0)
                    ax.scatter(t,map_,s=heatmap_scatter_s,c=heatmap_scatter_c)
                
            # Add colorbar
            # plt.colorbar(im, ax=ax)
            
            
            else:
                ax.text(0.5, 0.5, f"Unsupported data dimension: {d.ndim}",
                    horizontalalignment='center', verticalalignment='center',
                    transform=ax.transAxes)
            
        # Set title and labels
        if plot_title:
            ax.set_title(key)
        
        # Hide x-labels and ticks for all but the bottom subplot
        if i < n_plots - 1:
            plt.setp(ax.get_xticklabels(), visible=False)
    
    # Set common x-label on the bottom subplot
    axs[-1].set_xlabel('Time')
    
    # Adjust layout
    plt.tight_layout()
    
    return fig, axs,common_interval

# Example usage:
# fig, axs = plot_pynapple_data_mpl(data_dict, figsize=(12, 8))
# plt.show() 

import numpy as np
from collections.abc import Iterable

def shade_intervals(axs, intervals, *, sort_bounds=True, unique=True, **kwargs):
    """
    Shade vertical bands on one axis or a list/array of axes.

    Parameters
    ----------
    axs : matplotlib.axes.Axes or iterable of Axes
        Axis (or axes) to draw on.
    intervals : array-like, shape (n, 2)
        Each row is [x_min, x_max]. Can be unsorted; NaNs are ignored.
    sort_bounds : bool, default True
        If True, ensures left <= right for each interval.
    unique : bool, default True
        If True, merges duplicate rows (exact matches) after sorting.
    **kwargs :
        Passed through to `axvspan` (e.g., alpha=0.2, color='C0', zorder=0).

    Returns
    -------
    spans : list[list[matplotlib.patches.Polygon]]
        For each axis, the list of created span patches (outer list is aligned
        with `axs` order; if a single axis was passed, a single inner list).
    """
    intervals = np.asarray(intervals, dtype=float)

    if intervals.ndim != 2 or intervals.shape[1] != 2:
        raise ValueError("`intervals` must be an array of shape (n, 2).")

    # Drop rows with NaNs
    mask = ~np.isnan(intervals).any(axis=1)
    intervals = intervals[mask]

    if sort_bounds and len(intervals):
        left = np.minimum(intervals[:, 0], intervals[:, 1])
        right = np.maximum(intervals[:, 0], intervals[:, 1])
        intervals = np.column_stack([left, right])

    if unique and len(intervals):
        intervals = np.unique(intervals, axis=0)

    # Normalize axs to a list
    if hasattr(axs, "plot"):  # single Axes
        axes = [axs]
    elif isinstance(axs, Iterable):
        axes = list(axs)
        if not axes:
            raise ValueError("`axs` iterable is empty.")
        # Basic check: each must look like an Axes (has axvspan)
        for a in axes:
            if not hasattr(a, "axvspan"):
                raise TypeError("All items in `axs` must be Matplotlib Axes.")
    else:
        raise TypeError("`axs` must be an Axes or an iterable of Axes.")

    # Defaults
    if "alpha" not in kwargs and "facecolor" not in kwargs and "color" not in kwargs:
        kwargs.setdefault("alpha", 0.15)

    spans_all = []
    for ax in axes:
        spans = []
        for x0, x1 in intervals:
            if x0 == x1:  # zero-width: draw a thin line instead
                line = ax.axvline(x0, **{**kwargs, "alpha": kwargs.get("alpha", 0.15)})
                spans.append(line)
            else:
                patch = ax.axvspan(x0, x1, **kwargs)
                spans.append(patch)
        spans_all.append(spans)

    return spans_all

def pre_post_1d_timeseries_plot(tsd,pre_ep,post_ep,fig=None,ax=None):
    '''
    plot 1d timeseries in time; mark pre post with verticle lines and ticks
    pre_ep,post_ep: nap.IntervalSet
    tsd: nap.Tsd
    '''
    if ax is None:
        fig,ax=plt.subplots()
    ax.plot(tsd.t,tsd.d)
    ax.axvline(pre_ep.end[0],color='k',linestyle=':')
    ax.axvline(post_ep.start[0],color='k',linestyle=':')
    pre_mid = (pre_ep.end[0] + pre_ep.start[0]) / 2
    post_mid = (post_ep.end[0] + post_ep.start[0]) / 2
    behavior_mid = (pre_ep.end[0]+post_ep.start[0])/2
    ax.set_xticks([pre_mid,behavior_mid,post_mid])
    ax.set_xticklabels(['PRE','Beh.','POST'])
    return fig,ax

def median_plot(**kwargs):
    kwargs_hide=dict(
            boxprops=dict(visible=False),
            whiskerprops=dict(visible=False),
            capprops=dict(visible=False),
            flierprops=dict(visible=False)
            )
    kwargs.update(kwargs_hide)
    g=sns.boxplot(**kwargs)
    return g


def plot_trajectories_on_maze_mark_events(position_tsdf,x_peri_jump,y_peri_jump,fig=None,ax=None,ds=5,start_marker='<',end_marker='o',midpoint_marker='x',start_color='C0',end_color='C1',midpoint_color='red',trajectory_color='C0',trajectory_alpha=0.4,midpoint_label='jump',marker_size=5,marker_alpha=0.5,midpoint_only=False):
    '''
    position_tsdf: nap.TsdFrame, contain x and y columns
    x_peri_jump,y_peri_jump: array, shape (n_time,n_trajectory)

    mark start, end, midpoint
    '''
    if ax is None:
        fig,ax=plt.subplots()
    fig,ax=plot_maze_background(position_tsdf,ds=ds,fig=fig,ax=ax)
    
    for ind in range(x_peri_jump.shape[1]):

        midpt = x_peri_jump.shape[0]//2
        
        st = x_peri_jump[0,ind],y_peri_jump[0,ind]
        # Only add labels on first iteration to avoid duplicates in legend
        start_label = 'start' if ind == 0 else None
        end_label = 'end' if ind == 0 else None
        mid_label = midpoint_label if ind == 0 else None
        
        ax.scatter([x_peri_jump[midpt,ind]],[y_peri_jump[midpt,ind]],marker=midpoint_marker,c=midpoint_color,zorder=3,label=mid_label,s=marker_size,alpha=marker_alpha)
        if not midpoint_only:
            ax.plot(x_peri_jump[:,ind],y_peri_jump[:,ind],c=trajectory_color,alpha=trajectory_alpha,zorder=0)
            ax.scatter([st[0]],[st[1]],marker=start_marker,c=start_color,label=start_label,s=marker_size,alpha=marker_alpha,zorder=1)
            ax.scatter([x_peri_jump[-1,ind]],[y_peri_jump[-1,ind]],marker=end_marker,c=end_color,label=end_label,s=marker_size,alpha=marker_alpha,zorder=2)
        
    ax.legend(bbox_to_anchor=[1.05,1],frameon=False)
    
    return fig,ax

def plot_maze_background(spk_beh_df,ds=10,fig=None,ax=None,mode='line',**kwargs):
    kwargs_ = dict(c='grey',alpha=0.5)
    kwargs_.update(kwargs)
    if ax is None:
        fig,ax=plt.subplots()
    if mode=='line':
        ax.plot(spk_beh_df['x'].values[::ds],spk_beh_df['y'].values[::ds],**kwargs_)
    elif mode=='scatter':
        ax.scatter(spk_beh_df['x'].values[::ds],spk_beh_df['y'].values[::ds],s=1,**kwargs_)
    sns.despine(ax=ax,bottom=True,left=True)
    ax.axis('off')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    return fig,ax

def plot_data_shuffle_time_series(data, shuffle, align_at='middle', fig=None, ax=None, figsize=(6, 4), data_label='data', shuffle_label='null', data_color='C0', shuffle_color='C0', shuffle_alpha=0.3, data_lw=2,marker='o',marker_size=2):
    '''
    plot data and shuffle time series; data is a line and shuffle is a filled area for middle 95% quantile
    data: n_time
    shuffle: n_time x n_shuffle

    align_at: if not None: if 'middle', align each shuffle and data at the middle time point (this way to look at the change at the aligned time)
    '''
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    if isinstance(data,nap.Tsd):
        t_l = data.t
    else:
        t_l = np.arange(len(data))
    data = np.asarray(data)
    shuffle = np.asarray(shuffle)
    
    # Align each shuffle to data if requested
    if align_at == 'middle':
        mid_idx = len(data) // 2
        # Calculate offset for each shuffle individually
        shuffle_aligned = shuffle.copy()
        for i in range(shuffle.shape[1]):
            offset = data[mid_idx] - shuffle[mid_idx, i]
            shuffle_aligned[:, i] += offset
    else:
        shuffle_aligned = shuffle
    
    # Calculate shuffle quantiles and mean from aligned shuffles
    shuffle_mean = np.nanmean(shuffle_aligned, axis=1)
    shuffle_lower = np.nanquantile(shuffle_aligned, 0.025, axis=1)
    shuffle_upper = np.nanquantile(shuffle_aligned, 0.975, axis=1)
    
    # Time axis
    time_axis = t_l
    
    # Plot shuffle as filled area
    ax.fill_between(time_axis, shuffle_lower, shuffle_upper, 
                     alpha=shuffle_alpha, color=shuffle_color, label=shuffle_label)
    
    # Plot data as line
    ax.plot(time_axis, data, color=data_color, linewidth=data_lw, label=data_label,marker=marker,markersize=marker_size)
    
    ax.legend(bbox_to_anchor=[1.05,1],frameon=False)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Value')
    set_symmetric_ticks(ax,apply_to='x',do_int=True,)
    
    sns.despine(ax=ax)
    
    return fig, ax
    
from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar
def add_scalebar(ax, x, y, length, label=None, 
                 orientation='horizontal', 
                 linewidth=2, color='k', fontsize=10, 
                 zorder=10, text_offset=None, 
                 coord_system='axes', **kwargs):
    """
    Add a scale bar to a matplotlib Axes.

    Parameters
    ----------
    ax : matplotlib Axes
        The axes to draw on.
    x, y : float
        Starting location. Interpretation depends on coord_system.
    length : float
        Length of the scale bar. Interpretation depends on coord_system.
    label : str, optional
        Text label for the bar.
    orientation : str, 'horizontal' or 'vertical'
        Orientation of the scale bar.
    linewidth : float
        Width of the line.
    color : str
        Line and text color.
    fontsize : int
        Size of the label text.
    zorder : int
        Drawing order (higher values are drawn on top).
    text_offset : float, optional
        Custom offset for text positioning. If None, uses adaptive offset.
        For 'axes' coord_system, this should be in axes fraction (0-1).
        For 'data' coord_system, this should be in data units.
    coord_system : str, 'data' or 'axes'
        Coordinate system to use:
        - 'data': x, y, length in data coordinates (default)
        - 'axes': x, y, length as fractions of axes (0-1)
    kwargs : dict
        Passed to ax.plot (for extra styling).
    """
    # Determine transform based on coord_system
    if coord_system == 'axes':
        transform = ax.transAxes
    elif coord_system == 'data':
        transform = ax.transData
    else:
        raise ValueError(f"coord_system must be 'axes' or 'data', got {coord_system}")
    
    # Convert (x, y) to a location string for AnchoredSizeBar
    # Map approximate positions to location codes
    # Note: AnchoredSizeBar uses preset locations, not arbitrary (x, y)
    # We'll use a simple heuristic to map x, y to the nearest location
    if coord_system == 'axes':
        if y < 0.33:
            if x < 0.33:
                loc = 'lower left'
            elif x > 0.67:
                loc = 'lower right'
            else:
                loc = 'lower center'
        elif y > 0.67:
            if x < 0.33:
                loc = 'upper left'
            elif x > 0.67:
                loc = 'upper right'
            else:
                loc = 'upper center'
        else:
            if x < 0.33:
                loc = 'center left'
            elif x > 0.67:
                loc = 'center right'
            else:
                loc = 'center'
    else:
        # For data coordinates, default to lower center
        loc = 'lower center'
    
    # Handle orientation - AnchoredSizeBar is horizontal by default
    # For vertical bars, swap size and size_vertical
    if orientation == 'horizontal':
        size_horizontal = length
        size_vertical_val = linewidth / 100.0  # Small vertical thickness
    elif orientation == 'vertical':
        size_horizontal = linewidth / 100.0  # Small horizontal thickness
        size_vertical_val = length
    else:
        raise ValueError(f"orientation must be 'horizontal' or 'vertical', got {orientation}")
    
    # Create font properties for label
    import matplotlib.font_manager as fm
    fontprops = fm.FontProperties(size=fontsize)
    
    # Determine label position (top or bottom of bar)
    label_top = (orientation == 'horizontal' and text_offset is not None and text_offset > 0) or False
    
    # Set up keyword arguments for AnchoredSizeBar
    asb_kwargs = {
        'pad': 0.1,
        'borderpad': 0.1,
        'sep': text_offset if text_offset is not None else 2,
        'frameon': False,
        'size_vertical': size_vertical_val,
        'color': color,
        'label_top': label_top,
        'fontproperties': fontprops,
    }
    # Merge with any additional kwargs
    asb_kwargs.update(kwargs)
    
    # Create the AnchoredSizeBar
    scalebar = AnchoredSizeBar(
        transform,
        size_horizontal,
        label if label is not None else '',
        loc,
        **asb_kwargs
    )
    
    # Add to axes
    ax.add_artist(scalebar)
    
    return scalebar

def plot_brain_state_intervals(interval_dict,color_dict={'REM':'magenta','NREM':'blue','Awake':'black'},order=['REM','NREM','Awake'],gap=0.2,fig=None,ax=None):
    '''
    different brain state intervals are plotted as axvspan on different y coordinates and different colors
    '''
    if ax is None:
        fig,ax=plt.subplots()
    ymin=0
    ymax=ymin+gap
    yticks = []
    yticklabels=[]
    
    if order is None: # if order if provided then follow this order, which will determine the height of each interval
        order = list(interval_dict.keys())

    for state in order:
        yticks.append((ymin+ymax)/2)
        yticklabels.append(state)
        color = color_dict[state]
        interval = interval_dict[state]
        for ii,intv in enumerate(interval):
            if ii==0:
                label=state
            else:
                label=None
            ax.fill_between([intv['start'][0],intv['end'][0]], ymin, ymax, color=color, alpha=0.3, label=label)
            
        ymin=ymin+gap
        ymax=ymax+gap
    ax.set_yticks(yticks)
    ax.set_yticklabels(yticklabels)
    ax.set_ylim(0,ymax)
    ax.tick_params(axis='y',length=0)
    return fig,ax

from scipy.stats import ks_2samp

def plot_cdf_and_ks_test(sample1, sample2, alpha=0.05,fig=None,ax=None,label1='sample1',label2='sample2',xlabel='Value',title=None,
                        c_l  = ['C0','C1'],linestyle_l=['-','-'],alternative='two-sided',do_legend=True,
                        bins='auto',
                    ):
    _,bin_edges = np.histogram(np.concatenate([np.array(sample1),np.array(sample2)]),bins=bins,density=True)
    # Compute CDF for sample1
    # hist1, bin_edges1 = np.histogram(sample1, bins='auto', density=True)
    hist1, bin_edges1 = np.histogram(sample1, bins=bin_edges, density=True)
    cdf1 = np.cumsum(hist1) / np.sum(hist1)
    
    # Compute CDF for sample2
    # hist2, bin_edges2 = np.histogram(sample2, bins='auto', density=True)
    hist2, bin_edges2 = np.histogram(sample2, bins=bin_edges, density=True)
    cdf2 = np.cumsum(hist2) / np.sum(hist2)
    if ax is None:
        fig,ax=plt.subplots()
    # Plot CDFs
    ax.plot(bin_edges1[1:], cdf1, label=label1,c=c_l[0],linestyle=linestyle_l[0])
    ax.plot(bin_edges2[1:], cdf2, label=label2,c=c_l[1],linestyle=linestyle_l[1])
    
    # KS test
    ks_stat, p_value = ks_2samp(sample1, sample2,alternative=alternative)
    
    common_bins = np.union1d(bin_edges1, bin_edges2)
    cdf1_interp = np.interp(common_bins, bin_edges1[1:], cdf1)
    cdf2_interp = np.interp(common_bins, bin_edges2[1:], cdf2)
    max_diff_idx = np.argmax(np.abs(cdf1_interp - cdf2_interp))
    max_diff_x = common_bins[max_diff_idx]
    max_diff_y = max(cdf1_interp[max_diff_idx], cdf2_interp[max_diff_idx])
    
    
    if p_value < alpha:
        title_ = f"KS Test: p-value = {p_value:.3f} *\nstat={ks_stat:.3f}"
    else:
        title_ = f"KS Test: p-value = {p_value:.3f}\nstat={ks_stat:.3f}"
    
    if title is None:
        title=title_
    ax.set_title(title)
    
    if p_value < alpha:
        if 0.05 >= p_value > 0.01:
            star = "*"
        elif 0.01 >= p_value >0.001:
            star="**"
        elif 0.001 >= p_value >0.0001:
            star = "***"
        elif 0.0001 >= p_value:
            star = "****"


        # ax.annotate(f"* p={p_value:.3f}", (max_diff_x, max_diff_y + 0.05), 
        #              ha='center', va='bottom', color='k')
        ax.annotate(f"{star}", (max_diff_x, max_diff_y + 0.05), 
                     ha='center', va='bottom', color='k')
    else:
        ax.annotate(f"n.s.", (max_diff_x, max_diff_y + 0.05), 
                     ha='center', va='bottom', color='k')
    if do_legend:
        ax.legend(bbox_to_anchor=[1.05,1])
    ax.set_xlabel(xlabel)
    ax.set_ylabel('CDF')
    ax.grid(False)
    sns.despine()
#     plt.show()

    return fig,ax

# mark events in time (for heatmaps)
import numpy as np
import matplotlib.transforms as mtransforms

def mark_times_top_triangles(t_l, ax, c='red', y=1.02, s=60, **kwargs):
    """
    Draw downward-pointing triangles above the top of `ax`, pointing into an imshow.

    Parameters
    ----------
    t_l : float or array-like
        X positions (times) in the same units as ax's x-axis.
    ax : matplotlib.axes.Axes
        Axes containing the imshow.
    c : color
        Triangle color (can be overridden by kwargs['color']).
    y : float
        Vertical position in axes-fraction coords (1.0 = top edge). Use >1 to put above.
    s : float
        Marker size for scatter (points^2).
    **kwargs :
        Passed to ax.scatter (e.g., alpha, edgecolors, linewidths, zorder).

    Returns
    -------
    matplotlib.collections.PathCollection or None
    """
    t = np.atleast_1d(t_l).astype(float)

    # keep markers within current view
    x0, x1 = ax.get_xlim()
    lo, hi = (x0, x1) if x0 <= x1 else (x1, x0)
    t = t[(t >= lo) & (t <= hi)]
    if t.size == 0:
        return None

    blend = mtransforms.blended_transform_factory(ax.transData, ax.transAxes)

    kwargs.setdefault('color', c)
    kwargs.setdefault('marker', 'v')       # downward triangle
    kwargs.setdefault('clip_on', False)    # let it sit above the axes
    kwargs.setdefault('zorder', 10)

    return ax.scatter(t, np.full_like(t, y), transform=blend, s=s, **kwargs)


#=============plot decoded posterior 2D=============#
def plot_replay_posterior_trajectory_2d(posterior_position_one,position_tsdf,start_time=None,duration=None,fig=None,ax=None,despine=True,figsize=(2,2),plot_colorbar=False,start_time_x=0.1,start_time_y=0.95,duration_x=0.9,duration_y=0.95,fontsize=10,cmap_name='plasma',maze_alpha=0.5,x_key='x',y_key='y'):
    x = posterior_position_one[x_key]
    y=posterior_position_one[y_key]
    dx = (x[-1] - x[0]) / (len(x) - 1) if len(x) > 1 else 1.0
    dy = (y[-1] - y[0]) / (len(y) - 1) if len(y) > 1 else 1.0

    extent = [x[0] - dx/2,x[-1] + dx/2, y[0] - dy/2, y[-1] + dy/2]

    binarize_thresh=0.01
    toplot=np.zeros((*posterior_position_one.shape[1:],4))
    cmap = plt.colormaps.get_cmap(cmap_name)
    for tt in range(posterior_position_one.shape[0]):
        tt_normalized=tt/posterior_position_one.shape[0]
        c=cmap(tt_normalized)
        ma = posterior_position_one[tt] > binarize_thresh
        toplot[ma] = np.array(c)
    toplot=toplot.swapaxes(0,1)# make x the horizontal in imshow
    if ax is None:
        fig,ax=plt.subplots(figsize=figsize,constrained_layout=True)
    # Keep posterior (imshow) visually on top of the maze trajectory
    ax.imshow(toplot,aspect='auto',extent=extent,origin='lower', zorder=2)
    ax.plot(position_tsdf['x'],position_tsdf['y'],c='grey',alpha=maze_alpha, zorder=1)
    if despine:
        sns.despine(ax=ax,left=True,bottom=True)
        ax.set_xticks([])
        ax.set_yticks([])
    
    if start_time is not None:
        hour=start_time / 60
        minute=start_time % 60
        ax.text(start_time_x,start_time_y,f'{int(hour)}:{int(minute)}',fontsize=fontsize,transform=ax.transAxes)
    if duration is not None:
        ax.text(duration_x,duration_y,f'{duration:.02f}s',fontsize=fontsize,transform=ax.transAxes)
    if plot_colorbar:
        # generic time-progression colorbar (independent of RGBA imshow)
        norm = plt.Normalize(0, 1)
        sm = plt.cm.ScalarMappable(cmap=plt.colormaps.get_cmap(cmap_name), norm=norm)
        sm.set_array([])
        try:
            cbar = fig.colorbar(
                sm, ax=ax, orientation='horizontal',
                location='top', pad=0.02, fraction=0.08
            )
        except Exception:
            # fallback for older matplotlib / layout engines
            from mpl_toolkits.axes_grid1.inset_locator import inset_axes
            cax = inset_axes(
                ax, width="80%", height="6%", loc='upper center',
                bbox_to_anchor=(0.5, 1.08, 0, 0),
                bbox_transform=ax.transAxes, borderpad=0
            )
            cbar = fig.colorbar(sm, cax=cax, orientation='horizontal')
        cbar.set_ticks([0.0, 1.0])
        cbar.set_ticklabels(['early', 'late'])
        try:
            cbar.ax.xaxis.set_ticks_position('top')
        except Exception:
            pass
        # hide tick marks but keep labels
        try:
            cbar.ax.tick_params(axis='x', which='both', length=0)
        except Exception:
            pass
        try:
            cbar.outline.set_visible(False)
        except Exception:
            pass
        return fig, ax, cbar
    try:
        plt.tight_layout()
    except Exception:
        pass
    return fig, ax