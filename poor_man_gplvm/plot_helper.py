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

plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['svg.fonttype'] = 'none'

def save_fig(fig,fig_name,fig_dir='./figs',fig_format=['png','svg'],dpi=300):
    '''
    save figure to fig_dir
    '''
    if not os.path.exists(fig_dir):
        os.makedirs(fig_dir)
    for fmt in fig_format:
        fig.savefig(os.path.join(fig_dir,fig_name+f'.{fmt}'),dpi=dpi,bbox_inches='tight')
        print(f'saved {fig_name}.{fmt} to {fig_dir}')
    plt.close(fig)

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
    title_top_margin=70,         # extra top space so titles never clip
    annotation_yshift=8,         # lift subplot titles a bit (pixels)
    shared_vlines: list[float] | None = None,  # x positions
    showlegend=False
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
        if arr.d.size and arr.d.ndim == 1:
            y = arr.d
            mu, sd = float(np.nanmean(y)), float(np.nanstd(y))
            if np.isfinite(sd) and sd > 0:
                z = (y - mu) / sd
                filt = y[np.abs(z) <= 3.0]
                ymin, ymax = (np.min(filt), np.max(filt)) if filt.size else (np.min(y), np.max(y))
            else:
                ymin, ymax = float(np.nanmin(y)), float(np.nanmax(y))
            fig.update_yaxes(range=[ymin, ymax], row=i, col=1)

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
        margin=dict(t=title_top_margin, r=10, b=10, l=10)
    )
    # lift subplot titles slightly so they never collide with plots
    fig.for_each_annotation(lambda a: a.update(yshift=annotation_yshift))

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
