'''
helper functions for plotting
'''

import numpy as np
import matplotlib.pyplot as plt
import pynapple as nap
import sys,os
import pandas as pd

import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.stats import wilcoxon
import seaborn as sns

plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['svg.fonttype'] = 'none'

def save_fig(fig,fig_name,fig_dir='./figs',fig_format=['png','svg'],dpi=300):
    '''
    save figure to fig_dir
    '''
    # clean fig_name to avoid special characters
    fig_name = fig_name.replace(' ','_')
    fig_name = fig_name.replace('.','__')
    if not os.path.exists(fig_dir):
        os.makedirs(fig_dir)
    for fmt in fig_format:
        fig.savefig(os.path.join(fig_dir,fig_name+f'.{fmt}'),dpi=dpi,bbox_inches='tight')
        print(f'saved {fig_name}.{fmt} to {fig_dir}')
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

def plot_mean_error_plot(data,error_type='ci',mean_axis=0,fig=None,ax=None,**kwargs):
    '''
    plt the mean and error of the data
    data: pd.DataFrame or np.ndarray
    error_type: 'ci' or 'std'
    mean_axis: axis to take the mean of, same for error; plot the other axis
    '''
    if fig is None:
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
        ma = (intvl_l['start'] >= ep['start'][0]) & (intvl_l['end'] <= ep['end'][0])
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
        fig, ax = plt.subplots(figsize=figsize)

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
    text_str = f"Wilcoxon p={pvalue:.3g}, effect size={effect_size:.2f}, n={n_eff}"
    ax.text(0.02, 0.98, text_str, transform=ax.transAxes, ha='left', va='top', fontsize=text_fontsize)

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
    return fig, ax, result

