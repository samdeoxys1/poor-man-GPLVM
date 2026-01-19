'''
helper functions for animation
'''

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.animation import FuncAnimation
from matplotlib.colors import Normalize
import pynapple as nap

# Illustrator-friendly font settings
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42
plt.rcParams['svg.fonttype'] = 'none'


def animate_pynapple_data_mpl(
    toplot,
    position_tsdf,
    filename='anim.mp4',
    fps=10,
    dpi=100,
    height_per_plot=2,
    width_per_plot=4,
    height_ratios=None,
    plot_title=False,
    suptitle=None,
    annotation_text=None,
    # Aesthetics kwargs
    maze_kwargs=None,
    traj_kwargs=None,
    pos_marker_kwargs=None,
    line_kwargs=None,
    heatmap_kwargs=None,
    raster_kwargs=None,
    show=False,
):
    """
    Animate pynapple data with a two-column layout:
    - Left column: maze background (full session in grey) + animated trajectory from position_tsdf
    - Right column: animated time series (1D line, 2D heatmap, or raster)

    Parameters
    ----------
    toplot : dict
        Dictionary where each key maps to a pynapple object (Tsd, TsdFrame) or
        a raster tuple (tind, uind, c_l). Same contract as plot_pynapple_data_mpl.
    position_tsdf : nap.TsdFrame
        Must have columns 'x' and 'y' for position data. Full session used for
        maze background; restricted to common_interval for animated trajectory.
    filename : str
        Output filename (default: 'anim.mp4').
    fps : int
        Frames per second (default: 10).
    dpi : int
        DPI for saved animation (default: 100).
    height_per_plot : float
        Height per subplot row in inches.
    width_per_plot : float
        Width per column in inches.
    height_ratios : list or None
        Relative heights for each row. If None, equal heights.
    plot_title : bool
        Whether to show plot titles.
    suptitle : str or None
        Optional figure suptitle (e.g., "jump 5  t=123.456s").
    annotation_text : str or None
        Optional text to render in the blank left-column space below the maze.
    maze_kwargs : dict or None
        Kwargs for maze background plot (default: color='0.5', alpha=0.5, lw=0.5).
    traj_kwargs : dict or None
        Kwargs for animated trajectory line (default: color='C0', lw=1).
    pos_marker_kwargs : dict or None
        Kwargs for current position marker (default: c='red', s=50, zorder=10).
    line_kwargs : dict or None
        Kwargs for 1D line plots (default: lw=1).
    heatmap_kwargs : dict or None
        Kwargs for 2D heatmaps (default: cmap='viridis', interpolation='none').
    raster_kwargs : dict or None
        Kwargs for raster scatter (default: cmap='Spectral_r', s=5).
    show : bool
        Whether to display the animation (default: False).

    Returns
    -------
    fig : matplotlib.figure.Figure
    (ax_maze, axs_right) : tuple
        ax_maze is the maze axis, axs_right is list of right column axes.
    out_dict : dict
        Contains 'anim', 'common_interval', 'save_path', 't_frames'.

    Example
    -------
    >>> fig, (ax_maze, axs), out = animate_pynapple_data_mpl(toplot, position_tsdf)
    >>> # In notebook:
    >>> from IPython.display import Video
    >>> Video(out['save_path'], embed=True)
    """
    # --- Default kwargs ---
    maze_kwargs = maze_kwargs or {}
    maze_kwargs.setdefault('color', '0.5')
    maze_kwargs.setdefault('alpha', 0.5)
    maze_kwargs.setdefault('lw', 0.5)

    traj_kwargs = traj_kwargs or {}
    traj_kwargs.setdefault('color', 'C0')
    traj_kwargs.setdefault('lw', 1)

    pos_marker_kwargs = pos_marker_kwargs or {}
    pos_marker_kwargs.setdefault('c', 'red')
    pos_marker_kwargs.setdefault('s', 50)
    pos_marker_kwargs.setdefault('zorder', 10)

    line_kwargs = line_kwargs or {}
    line_kwargs.setdefault('lw', 1)

    heatmap_kwargs = heatmap_kwargs or {}
    heatmap_kwargs.setdefault('cmap', 'viridis')
    heatmap_kwargs.setdefault('interpolation', 'none')
    heatmap_kwargs.setdefault('aspect', 'auto')
    heatmap_kwargs.setdefault('origin', 'lower')

    raster_kwargs = raster_kwargs or {}
    raster_kwargs.setdefault('cmap', 'Spectral_r')
    raster_kwargs.setdefault('s', 5)

    # --- Compute common interval from toplot (NOT including full position_tsdf) ---
    data_dict = dict(toplot)  # copy to avoid mutating input
    min_times = [np.min(arr.t) for arr in data_dict.values() if not isinstance(arr, tuple)]
    max_times = [np.max(arr.t) for arr in data_dict.values() if not isinstance(arr, tuple)]

    st = np.max(min_times)
    ed = np.min(max_times)
    common_interval = nap.IntervalSet([st, ed])

    # --- Restrict data to common interval ---
    for key, arr in data_dict.items():
        if not isinstance(arr, tuple):
            data_dict[key] = arr.restrict(common_interval)
        else:
            tind, uind, c_l = arr
            ma = (tind < ed) & (tind > st)
            data_dict[key] = (tind[ma], uind[ma], c_l[ma])

    # Full-session maze background (not restricted)
    x_bg = position_tsdf['x'].d
    y_bg = position_tsdf['y'].d

    # Animated trajectory uses restricted position
    pos_restricted = position_tsdf.restrict(common_interval)
    x_anim = pos_restricted['x'].d
    y_anim = pos_restricted['y'].d
    t_pos = pos_restricted.t

    # --- Time frames ---
    t_frames = np.arange(st, ed, 1 / fps)
    n_frames = len(t_frames)

    # --- Layout setup ---
    n_plots = len(data_dict)
    if height_ratios is None:
        height_ratios = [1] * n_plots

    total_height = height_per_plot * n_plots
    total_width = width_per_plot * 2  # two columns

    fig = plt.figure(figsize=(total_width, total_height), constrained_layout=True)
    gs = gridspec.GridSpec(n_plots, 2, height_ratios=height_ratios, width_ratios=[1, 1.5])

    # Suptitle if provided
    if suptitle:
        fig.suptitle(suptitle, fontsize=12)

    # Left column: maze axis spanning multiple rows for larger size
    # Use ~60% of rows for maze, rest for annotation
    maze_rows = max(1, int(n_plots * 0.6))
    ax_maze = fig.add_subplot(gs[:maze_rows, 0])
    ax_maze.set_aspect('equal', adjustable='datalim')
    # Hide box/spines
    ax_maze.axis('off')

    # Create annotation axis spanning remaining left-column rows
    ax_annot = None
    if maze_rows < n_plots:
        ax_annot = fig.add_subplot(gs[maze_rows:, 0])
        ax_annot.axis('off')
        if annotation_text:
            ax_annot.text(
                0.05, 0.95, annotation_text,
                transform=ax_annot.transAxes,
                fontsize=10, va='top', ha='left',
                family='monospace'
            )
    elif annotation_text:
        # If maze spans all rows, put annotation inside maze plot
        ax_maze.text(
            0.05, 0.05, annotation_text,
            transform=ax_maze.transAxes,
            fontsize=9, va='bottom', ha='left',
            family='monospace',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8)
        )

    # Right column: stacked subplots
    axs_right = []
    for i in range(n_plots):
        if i == 0:
            ax = fig.add_subplot(gs[i, 1])
        else:
            ax = fig.add_subplot(gs[i, 1], sharex=axs_right[0])
        axs_right.append(ax)
        if i < n_plots - 1:
            plt.setp(ax.get_xticklabels(), visible=False)
    axs_right[-1].set_xlabel('Time')

    # --- Prepare maze artists ---
    # Background: full session maze in grey
    ax_maze.plot(x_bg, y_bg, **maze_kwargs)
    # Animated trajectory line
    traj_line, = ax_maze.plot([], [], **traj_kwargs)
    # Current position marker
    pos_marker = ax_maze.scatter([], [], **pos_marker_kwargs)

    # --- Prepare right column artists and store metadata ---
    artists_info = []  # list of dicts with artist and metadata
    for i, (key, arr) in enumerate(data_dict.items()):
        ax = axs_right[i]
        info = {'key': key, 'ax': ax}

        if isinstance(arr, tuple):
            # Raster plot
            tind, uind, c_l = arr
            info['type'] = 'raster'
            info['tind'] = tind
            info['uind'] = uind
            info['c_l'] = c_l
            # Initial empty scatter
            scat = ax.scatter([], [], c=[], **raster_kwargs)
            info['artist'] = scat
            # Set axis limits from full data
            if len(tind) > 0:
                ax.set_xlim(st, ed)
                ax.set_ylim(np.min(uind) - 0.5, np.max(uind) + 0.5)
        else:
            t = arr.t
            d = arr.d
            info['t'] = t
            info['d'] = d

            if d.ndim == 1:
                info['type'] = '1d'
                line, = ax.plot([], [], label=key, **line_kwargs)
                info['artist'] = line
                # Set y limits from full data (robust)
                mu = np.nanmean(d)
                sigma = np.nanstd(d)
                if sigma > 0:
                    z = (d - mu) / sigma
                    filtered = d[np.abs(z) <= 5]
                    if len(filtered) > 0:
                        ymin, ymax = np.min(filtered), np.max(filtered)
                        ax.set_ylim(ymin, ymax)
                ax.set_xlim(st, ed)

            elif d.ndim == 2:
                info['type'] = '2d'
                d_plot = d.T
                info['d_plot'] = d_plot
                n_rows = d_plot.shape[0]
                info['n_rows'] = n_rows

                # Robust color limits
                zmin = np.nanquantile(d_plot, 0.01)
                zmax = np.nanquantile(d_plot, 0.99)
                norm = Normalize(vmin=zmin, vmax=zmax)
                info['norm'] = norm

                # Initial image with single column
                initial_data = d_plot[:, :1] if d_plot.shape[1] > 0 else np.zeros((n_rows, 1))
                hm_kw = {k: v for k, v in heatmap_kwargs.items()}
                im = ax.imshow(
                    initial_data,
                    extent=[st, st + 0.001, 0, n_rows],
                    norm=norm,
                    **hm_kw
                )
                info['artist'] = im
                ax.set_xlim(st, ed)
                ax.set_ylim(0, n_rows)
            else:
                info['type'] = 'unsupported'
                ax.text(0.5, 0.5, f"Unsupported dim: {d.ndim}",
                        ha='center', va='center', transform=ax.transAxes)

        if plot_title:
            ax.set_title(key)

        artists_info.append(info)

    # --- Animation update function ---
    def update(frame_idx):
        t_now = t_frames[frame_idx]

        # Update maze trajectory (animated part)
        mask_pos = t_pos <= t_now
        if np.any(mask_pos):
            x_show = x_anim[mask_pos]
            y_show = y_anim[mask_pos]
            traj_line.set_data(x_show, y_show)
            pos_marker.set_offsets([[x_show[-1], y_show[-1]]])
        else:
            traj_line.set_data([], [])
            pos_marker.set_offsets(np.empty((0, 2)))

        # Update right column artists
        for info in artists_info:
            atype = info.get('type')

            if atype == 'raster':
                tind = info['tind']
                uind = info['uind']
                c_l = info['c_l']
                mask = tind <= t_now
                if np.any(mask):
                    offsets = np.column_stack([tind[mask], uind[mask]])
                    info['artist'].set_offsets(offsets)
                    info['artist'].set_array(c_l[mask])
                else:
                    info['artist'].set_offsets(np.empty((0, 2)))

            elif atype == '1d':
                t = info['t']
                d = info['d']
                mask = t <= t_now
                info['artist'].set_data(t[mask], d[mask])

            elif atype == '2d':
                t = info['t']
                d_plot = info['d_plot']
                n_rows = info['n_rows']
                mask = t <= t_now
                k = np.sum(mask)
                if k > 0:
                    info['artist'].set_data(d_plot[:, :k])
                    info['artist'].set_extent([t[0], t[k - 1], 0, n_rows])

        return [traj_line, pos_marker] + [info.get('artist') for info in artists_info if info.get('artist')]

    # --- Create animation ---
    anim = FuncAnimation(fig, update, frames=n_frames, interval=1000 / fps, blit=True)

    # --- Save ---
    save_path = os.path.abspath(filename)
    os.makedirs(os.path.dirname(save_path) or '.', exist_ok=True)
    anim.save(save_path, writer='ffmpeg', fps=fps, dpi=dpi)
    print(f"Animation saved to: {save_path}")

    # --- Tight layout (wrapped for safety) ---
    try:
        plt.tight_layout()
    except RuntimeError:
        pass

    # Always close figure to avoid accumulating open figures in cluster notebooks
    if not show:
        plt.close(fig)

    out_dict = {
        'anim': anim,
        'common_interval': common_interval,
        'save_path': save_path,
        't_frames': t_frames,
    }

    return fig, (ax_maze, axs_right), out_dict
