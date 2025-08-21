'''
analysis of distance
'''

import numpy as np
import pandas as pd

import numpy as np
from scipy.spatial.distance import cdist
import statsmodels.api as sm

from scipy.spatial.distance import pdist, squareform


def compute_distance_lag(
    X,
    *,
    metric='euclidean',
    label_d=None,
    do_plot=False,
    max_index_lag=None,
    label_bins=None,
    bin_count=20,
    random_state=None,
    ax=None
):
    """
    Compute pairwise distances for an (n_time x n_feature) matrix, derive laged pairs
    (index-based and optional label-based), summarize mean/std/sem by lag, and
    optionally plot distance vs lag with error shading.

    Parameters
    ----------
    X : (n_time, n_feature) array_like
        Observations in time order.
    metric : str or callable
        Metric passed to scipy.spatial.distance.pdist.
    label_d : (n_time,) array_like or None
        Optional labels aligned to rows of X for label-based lags. If provided,
        label lag is |label[j] - label[i]| for i<j.
    do_plot : bool
        If True, produce a simple plot of distance vs index lag with shaded error.
        If label_d is provided, also plots distance vs label lag (binned if needed).
    max_index_lag : int or None
        If given, restrict summaries/plots to pairs with index_lag <= max_index_lag.
    label_bins : array_like or None
        Optional explicit bin edges for label lag aggregation. If None and label_d
        is provided, bins are chosen automatically when label lag is continuous.
    bin_count : int
        Number of bins to use when auto-binning label lag.
    random_state : int or None
        Unused currently; kept for API stability if downsampling is added later.
    ax : matplotlib Axes or None
        Optional axes to draw the index-lag plot on. If None and do_plot=True,
        a new figure/axes is created.

    Returns
    -------
    dict
        {
          'D': (n,n) distance matrix,
          'pairs_df': DataFrame of upper-tri pairs with columns:
              ['i','j','dist','index_lag', ('label_lag' if label_d provided)],
          'by_index_lag': summary DataFrame with ['index_lag','n','mean','std','sem'],
          'by_label_lag': summary DataFrame or None,
          'figs': dict with optional matplotlib figures/axes {'index': (fig, ax), 'label': (fig, ax)}
        }
    """
    X = np.asarray(X, dtype=float)
    if X.ndim != 2:
        raise ValueError("X must be 2D (n_time, n_feature)")

    # Pairwise distances and square matrix
    cond_dists = pdist(X, metric=metric)
    D = squareform(cond_dists)

    n_time = X.shape[0]
    iu, ju = np.triu_indices(n_time, k=1)
    dist_vals = D[iu, ju]
    index_lag = (ju - iu).astype(int)

    data = {
        'i': iu,
        'j': ju,
        'dist': dist_vals,
        'index_lag': index_lag,
    }

    by_label_lag = None
    label_vals = None
    if label_d is not None:
        label_vals = np.asarray(label_d)
        if label_vals.shape[0] != n_time:
            raise ValueError("label_d must have length n_time")
        label_lag = np.abs(label_vals[ju] - label_vals[iu])
        data['label_lag'] = label_lag

    pairs_df = pd.DataFrame(data)

    if max_index_lag is not None:
        pairs_df = pairs_df[pairs_df['index_lag'] <= int(max_index_lag)].copy()

    # Summaries by index lag
    by_index = (
        pairs_df.groupby('index_lag')['dist']
        .agg(n='count', mean='mean', std='std')
        .reset_index()
    )
    by_index['sem'] = by_index['std'] / np.sqrt(by_index['n'].where(by_index['n'] > 0, np.nan))

    # Summaries by label lag, if provided
    if label_d is not None:
        ll = pairs_df['label_lag'].to_numpy()
        # Decide whether to bin: if many unique values, bin; else group directly
        unique_vals = np.unique(ll[np.isfinite(ll)])
        if label_bins is not None:
            bins = np.asarray(label_bins, dtype=float)
            labels = 0.5 * (bins[:-1] + bins[1:])
            cats = pd.cut(ll, bins=bins, include_lowest=True)
            tmp = pairs_df.copy()
            tmp['label_lag_bin'] = cats
            by_label = (
                tmp.groupby('label_lag_bin')['dist']
                .agg(n='count', mean='mean', std='std')
                .reset_index()
            )
            # bin centers for plotting
            centers = by_label['label_lag_bin'].apply(lambda iv: iv.mid if pd.notnull(iv) else np.nan)
            by_label.insert(1, 'label_lag', centers.astype(float))
        elif unique_vals.size <= 50:
            by_label = (
                pairs_df.groupby('label_lag')['dist']
                .agg(n='count', mean='mean', std='std')
                .reset_index()
            )
        else:
            # quantile bins for continuous label lag
            qs = np.linspace(0, 1, bin_count + 1)
            bins = np.unique(np.quantile(ll, qs))
            if bins.size < 2:
                # degenerate; fallback to direct grouping
                by_label = (
                    pairs_df.groupby('label_lag')['dist']
                    .agg(n='count', mean='mean', std='std')
                    .reset_index()
                )
            else:
                cats = pd.cut(ll, bins=bins, include_lowest=True)
                tmp = pairs_df.copy()
                tmp['label_lag_bin'] = cats
                by_label = (
                    tmp.groupby('label_lag_bin')['dist']
                    .agg(n='count', mean='mean', std='std')
                    .reset_index()
                )
                centers = by_label['label_lag_bin'].apply(lambda iv: iv.mid if pd.notnull(iv) else np.nan)
                by_label.insert(1, 'label_lag', centers.astype(float))

        by_label['sem'] = by_label['std'] / np.sqrt(by_label['n'].where(by_label['n'] > 0, np.nan))
        by_label_lag = by_label

    figs = {}
    if do_plot:
        # Lazy imports so the module does not require plotting libs unless used
        import matplotlib.pyplot as plt  # noqa: WPS433
        try:
            import seaborn as sns  # noqa: WPS433
            _ = sns.set_style('whitegrid')
        except Exception:
            pass

        # Plot index-lag curve
        if ax is None:
            fig_idx, ax_idx = plt.subplots(1, 1, figsize=(6, 4))
        else:
            fig_idx = ax.figure
            ax_idx = ax
        x = by_index['index_lag'].to_numpy()
        m = by_index['mean'].to_numpy()
        e = by_index['sem'].to_numpy()
        ax_idx.plot(x, m, color='C0', label='Index lag')
        ax_idx.fill_between(x, m - e, m + e, color='C0', alpha=0.2)
        ax_idx.set_xlabel('Index lag')
        ax_idx.set_ylabel('Distance')
        ax_idx.set_title('Distance vs index lag')
        ax_idx.legend(loc='best')
        figs['index'] = (fig_idx, ax_idx)

        # Plot label-lag curve if available
        if by_label_lag is not None:
            fig_lab, ax_lab = plt.subplots(1, 1, figsize=(6, 4))
            if 'label_lag' in by_label_lag.columns:
                x2 = by_label_lag['label_lag'].to_numpy()
            else:
                x2 = by_label_lag['label_lag'].index.to_numpy()
            m2 = by_label_lag['mean'].to_numpy()
            e2 = by_label_lag['sem'].to_numpy()
            ax_lab.plot(x2, m2, color='C1', label='Label lag')
            ax_lab.fill_between(x2, m2 - e2, m2 + e2, color='C1', alpha=0.2)
            ax_lab.set_xlabel('Label lag')
            ax_lab.set_ylabel('Distance')
            ax_lab.set_title('Distance vs label lag')
            ax_lab.legend(loc='best')
            figs['label'] = (fig_lab, ax_lab)

    return {
        'D': D,
        'pairs_df': pairs_df,
        'by_index_lag': by_index,
        'by_label_lag': by_label_lag,
        'figs': figs,
    }


def w1_cdf_distance_matrix(prob_mat, bin_edges=None, normalize=False):
    """
    Compute the 1D Wasserstein-1 (Earth Mover's) distance matrix between rows
    of a probability matrix using the CDF trick.

    Parameters
    ----------
    prob_mat : (n_time, n_feat) array_like
        Each row is a discrete distribution over ordered bins (histogram).
    bin_edges : (n_feat+1,) array_like or None
        Bin edges along the ordered axis. If None, bins are assumed equally spaced
        with width=1. For non-uniform bins, supply edges so widths = np.diff(bin_edges).
    normalize : bool
        If True, renormalize each row to sum to 1 (safe if inputs are counts).

    Returns
    -------
    D : (n_time, n_time) ndarray
        Pairwise W1 distances.
    C : (n_time, n_feat) ndarray
        Row-wise CDFs used for the computation.
    """
    P = np.asarray(prob_mat, dtype=float)
    if normalize:
        row_sums = P.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0.0] = 1.0
        P = np.clip(P, 0.0, None) / row_sums

    # Row-wise CDFs at bin ends (right-closed bins)
    C = np.cumsum(P, axis=1)

    # Bin widths (weights for integrating |F_p - F_q| over x)
    if bin_edges is None:
        w = np.ones(P.shape[1], dtype=float)  # equal-width bins (Δx = 1)
    else:
        edges = np.asarray(bin_edges, dtype=float)
        if edges.ndim != 1 or edges.size != P.shape[1] + 1:
            raise ValueError("bin_edges must have shape (n_feat+1,)")
        w = np.diff(edges)  # positive widths

    # Weighted L1 distance between CDF rows:
    # W1(p,q) = sum_i w_i * |C_p[i] - C_q[i]|
    # Implemented by scaling features then using cityblock.
    Cw = C * w[None, :]
    D = cdist(Cw, Cw, metric='cityblock')
    return D, C


def _upper_triangle_pairs(D, labels):
    """Return upper-tri pairs after dropping NaN labels."""
    D = np.asarray(D, dtype=float)
    labels = np.asarray(labels, dtype=float)
    assert D.ndim == 2 and D.shape[0] == D.shape[1], "D must be square"
    assert labels.shape[0] == D.shape[0], "labels length must match D"

    keep = np.isfinite(labels)
    idx = np.where(keep)[0]
    Dv = D[np.ix_(idx, idx)]
    lv = labels[idx]
    iu, ju = np.triu_indices(len(idx), 1)
    x = np.abs(lv[ju] - lv[iu])         # label distance
    y = Dv[iu, ju]                       # distances
    m = np.isfinite(y)                   # (Dv should be finite; filter anyway)
    iu, ju, x, y = iu[m], ju[m], x[m], y[m]
    # map back to original indices for the dataframe
    i_orig, j_orig = idx[iu], idx[ju]
    return Dv, lv, iu, ju, x, y, i_orig, j_orig, idx

def _bin_stats(x, y, *, bin_edges=None, nbins=50, binning='uniform', z=1.96):
    """
    Bin x and compute mean/std/CI of y per bin. Empty bins -> NaN.
    """
    x = np.asarray(x); y = np.asarray(y)
    if bin_edges is None:
        if binning == 'uniform':
            lo, hi = np.nanmin(x), np.nanmax(x)
            if not np.isfinite(lo) or not np.isfinite(hi) or lo == hi:
                bin_edges = np.array([lo, hi])  # degenerate; handle below
            else:
                bin_edges = np.linspace(lo, hi, nbins + 1)
        elif binning == 'quantile':
            qs = np.linspace(0, 1, nbins + 1)
            bin_edges = np.quantile(x, qs)
            # ensure strictly increasing to avoid zero-width bins
            bin_edges = np.unique(bin_edges)
            if bin_edges.size < 2:
                bin_edges = np.array([x.min(), x.max()])
        else:
            raise ValueError("binning must be 'uniform' or 'quantile'")

    # digitize
    bins = np.digitize(x, bin_edges, right=False) - 1  # 0..nbins-1
    nb = len(bin_edges) - 1
    means = np.full(nb, np.nan)
    stds  = np.full(nb, np.nan)
    ns    = np.zeros(nb, dtype=int)

    for b in range(nb):
        sel = (bins == b)
        if np.any(sel):
            ys = y[sel]
            means[b] = np.mean(ys)
            stds[b]  = np.std(ys, ddof=1) if ys.size > 1 else 0.0
            ns[b]    = ys.size

    sem = np.where(ns > 1, stds / np.sqrt(ns), np.nan)
    ci_lo = means - z * sem
    ci_hi = means + z * sem

    centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    out = pd.DataFrame({
        "bin_left": bin_edges[:-1],
        "bin_right": bin_edges[1:],
        "bin_center": centers,
        "n": ns,
        "mean": means,
        "std": stds,
        "ci_low": ci_lo,
        "ci_high": ci_hi,
    })
    return out, bin_edges

def _linregress_np(x, y):
    """Simple OLS (y = a + b x) with R^2, fully NumPy."""
    x = np.asarray(x, float); y = np.asarray(y, float)
    xm, ym = x.mean(), y.mean()
    vx = np.sum((x - xm)**2)
    if vx == 0:
        return dict(intercept=np.nan, slope=np.nan, r=np.nan, r2=np.nan)
    slope = np.sum((x - xm)*(y - ym)) / vx
    intercept = ym - slope * xm
    # Pearson r (same as correlation)
    r = np.corrcoef(x, y)[0,1]
    return dict(intercept=intercept, slope=slope, r=r, r2=r**2)

def _residualize_on_time(y, t):
    """
    Residualize y by removing a linear effect of t: y_resid = y - (a + b*t).
    Returns (y_resid, model_dict(intercept=a, slope=b)). If t has zero variance,
    uses slope=0 and intercept=mean(y).
    """
    y = np.asarray(y, float)
    t = np.asarray(t, float)
    if y.size == 0:
        return y, dict(intercept=np.nan, slope=np.nan)
    tm = np.mean(t)
    ym = np.mean(y)
    vt = np.sum((t - tm)**2)
    if vt == 0 or not np.isfinite(vt):
        a = ym
        b = 0.0
    else:
        b = np.sum((t - tm) * (y - ym)) / vt
        a = ym - b * tm
    resid = y - (a + b * t)
    return resid, dict(intercept=a, slope=b)




def distance_vs_label_regression(
    D, labels, *, bin_edges=None, nbins=50, binning='uniform', z=1.96, return_pairs_df=True,
    timestamps=None, label_distance_threshold=None
):
    """
    Build the upper-triangle dataset (distance vs |Δlabel|), run OLS, and compute
    binned mean/std/CI vs label distance. Optionally include time as an additional
    regressor using pairwise |Δtime|.

    Parameters
    ----------
    D : (n,n) square array
    labels : (n,) array_like
    bin_edges, nbins, binning, z : see _bin_stats
    return_pairs_df : if True, return the pairs dataframe
    timestamps : (n,) array_like or None
        If provided, include pairwise |Δtime| as an additional regressor.
    label_distance_threshold : float or None
        If provided, replace continuous |Δlabel| with a binary variable:
        1 if |Δlabel| > threshold else 0. If threshold==0, this enforces strict
        sameness as the 0 category.

    Returns
    -------
    dict with keys: pairs_df, summary, binned, edges, kept_idx
      - summary: dict(intercept, slope, r, r2), where slope is the coefficient
        for the label regressor; r is NaN when multiple regressors are used.
    """
    Dv, lv, iu, ju, x_cont, y, i_orig, j_orig, kept_idx = _upper_triangle_pairs(D, labels)

    # Optional time differences
    if timestamps is not None:
        tv = np.asarray(timestamps, float)[kept_idx]
        t_pairs = np.abs(tv[ju] - tv[iu])
    else:
        t_pairs = None

    # Choose regressor: continuous |Δlabel| or binary category
    if label_distance_threshold is not None:
        thr = float(label_distance_threshold)
        x = (x_cont > thr).astype(float)
        edges_eff = np.array([-0.5, 0.5, 1.5])
    else:
        x = x_cont
        edges_eff = bin_edges

    # Build regression design (const, label, [time]) with NaN-safe mask
    cols = {"label": x}
    if t_pairs is not None:
        cols["time"] = t_pairs
    X = np.column_stack([cols[c] for c in cols])
    X = sm.add_constant(X, has_constant='add')

    # Create mask of finite rows
    mask = np.isfinite(y)
    for arr in cols.values():
        mask &= np.isfinite(arr)

    y_use = y[mask]
    X_use = X[mask]

    # Fit OLS
    model = sm.OLS(y_use, X_use)
    result = model.fit()

    # Extract coefficients
    params = result.params
    intercept = params[0]
    slope_label = params[1] if "label" in cols else np.nan
    r2 = float(result.rsquared)
    # r is undefined for multi-regressor; keep for 1D case
    if t_pairs is None:
        r = np.sign(slope_label) * np.sqrt(r2)
    else:
        r = np.nan

    summary = dict(intercept=intercept, slope=slope_label, r=r, r2=r2)

    # Binned stats using the masked data
    x_for_bins = x[mask]
    binned, edges_used = _bin_stats(x_for_bins, y_use, bin_edges=edges_eff, nbins=nbins, binning=binning, z=z)

    pairs_df = None
    if return_pairs_df:
        i_use = i_orig[mask]; j_use = j_orig[mask]
        pairs_data = {
            "i": i_use, "j": j_use,
            "label_i": labels[i_use], "label_j": labels[j_use],
            "label_dist": x_cont[mask],
            "dist": y_use,
        }
        if t_pairs is not None:
            pairs_data["time_dist"] = t_pairs[mask]
        if label_distance_threshold is not None:
            pairs_data["label_dist_bin"] = x_for_bins
        pairs_df = pd.DataFrame(pairs_data)

    res = dict(pairs_df=pairs_df, summary=summary, binned=binned, edges=edges_used, kept_idx=kept_idx)

    return res

def shuffle_test_distance_vs_label(
    D, labels, *, n_shuffles=1000, rng=None, bin_edges=None, nbins=50, binning='uniform',
    timestamps=None, label_distance_threshold=None
):
    """
    Shuffle test: randomly permute rows/cols of D (labels stay put),
    then recompute regression and binned means each time.

    If timestamps are provided, include pairwise |Δtime| as an additional regressor.

    If label_distance_threshold is provided, regress on the binary category
    1[|Δlabel| <= threshold]; binning is fixed to two bins.

    Returns:
      results dict with:
        slope_obs, intercept_obs, r2_obs
        slopes_shuf (n_shuffles,), intercepts_shuf (n_shuffles,), r2_shuf (n_shuffles,)
        p_slope_two_sided (empirical)
        binned_obs : DataFrame
        binned_mean_shuf : (n_bins,) mean across shuffles
        binned_lo_shuf, binned_hi_shuf : (n_bins,) 2.5% and 97.5% across shuffles
        bin_edges : ndarray used
    """
    rng = np.random.default_rng(rng)
    # Observed using the same API to get edges and mask semantics
    obs = distance_vs_label_regression(
        D, labels, bin_edges=bin_edges, nbins=nbins, binning=binning, return_pairs_df=False,
        timestamps=timestamps, label_distance_threshold=label_distance_threshold
    )
    summary_obs = obs['summary']
    binned_obs = obs['binned']
    edges = obs['edges']
    kept_idx = obs['kept_idx']

    # Build design on the full (kept) upper-triangle pairs (mask later like observed)
    Dv, lv, iu, ju, x_cont, y, *_ = _upper_triangle_pairs(D, labels)

    # Optional time differences
    if timestamps is not None:
        tv = np.asarray(timestamps, float)[kept_idx]
        t_pairs = np.abs(tv[ju] - tv[iu])
    else:
        t_pairs = None

    # Label regressor
    if label_distance_threshold is not None:
        thr = float(label_distance_threshold)
        x_reg = (x_cont <= thr).astype(float)
    else:
        x_reg = x_cont

    # Build mask to emulate observed filtering: finite y, finite x, finite time if used
    mask = np.isfinite(y) & np.isfinite(x_reg)
    if t_pairs is not None:
        mask &= np.isfinite(t_pairs)

    # Pre-compute design matrix for masked rows
    cols = {"label": x_reg[mask]}
    if t_pairs is not None:
        cols["time"] = t_pairs[mask]
    X = np.column_stack([cols[c] for c in cols])
    X = sm.add_constant(X, has_constant='add')

    nb = len(edges) - 1
    slopes = np.empty(n_shuffles); intercepts = np.empty(n_shuffles); r2s = np.empty(n_shuffles)
    binned_means = np.full((n_shuffles, nb), np.nan)

    n = Dv.shape[0]
    for s in range(n_shuffles):
        perm = rng.permutation(n)
        # Sample shuffled upper-triangle distances without materializing Dp
        y_all = Dv[perm[iu], perm[ju]]
        y_use = y_all[mask]
        # Fit OLS against fixed design
        model = sm.OLS(y_use, X)
        result = model.fit()
        params = result.params
        intercepts[s] = params[0]
        slopes[s] = params[1] if X.shape[1] >= 2 else np.nan
        r2s[s] = float(result.rsquared)
        # Binned means on fixed edges
        binned_s, _ = _bin_stats(cols["label"], y_use, bin_edges=edges)
        binned_means[s, :] = binned_s['mean'].to_numpy()

    # empirical two-sided p for slope
    slope_obs = summary_obs['slope']
    p_two = (1 + np.sum(np.abs(slopes) >= np.abs(slope_obs))) / (n_shuffles + 1)

    # summarize shuffles for plotting/CI
    lo = np.nanpercentile(binned_means, 2.5, axis=0)
    hi = np.nanpercentile(binned_means, 97.5, axis=0)
    mean_shuf = np.nanmean(binned_means, axis=0)

    return dict(
        slope_obs=slope_obs,
        intercept_obs=summary_obs['intercept'],
        r2_obs=summary_obs['r2'],
        slopes_shuf=slopes,
        intercepts_shuf=intercepts,
        r2_shuf=r2s,
        p_slope_two_sided=p_two,
        binned_obs=binned_obs,
        binned_mean_shuf=mean_shuf,
        binned_lo_shuf=lo,
        binned_hi_shuf=hi,
        bin_edges=edges
    )




import numpy as np

def interpolate_stacks(mats, *, n_point=10, ddof=0):
    """
    Interpolate a list of time×feature matrices onto a common grid in [0, 1].

    Parameters
    ----------
    mats : list of np.ndarray
        Each array has shape (n_time_i, n_feature). The time axis is assumed
        to be sampled at np.linspace(0, 1, n_time_i, endpoint=True).
        All matrices must have the same n_feature.
    n_point : int
        Number of target points on the common grid in [0, 1].
    ddof : int, optional (default=0)
        Delta degrees of freedom for std across the list (0 = population std,
        1 = sample std).

    Returns
    -------
    out : dict
        {
          "grid": np.ndarray of shape (n_point,),          # the common grid
          "stack": np.ndarray of shape (n_list, n_point, n_feature),
          "mean": np.ndarray of shape (n_point, n_feature),
          "std":  np.ndarray of shape (n_point, n_feature),
        }
    """
    if not mats:
        raise ValueError("`mats` must be a non-empty list of 2D arrays.")

    # Basic validation and feature dimension check
    first = np.asarray(mats[0])
    if first.ndim != 2:
        raise ValueError("Each item must be a 2D array (n_time × n_feature).")
    n_feature = first.shape[1]

    for i, M in enumerate(mats):
        M = np.asarray(M)
        if M.ndim != 2:
            raise ValueError(f"Item {i} is not 2D.")
        if M.shape[1] != n_feature:
            raise ValueError(
                f"Item {i} has n_feature={M.shape[1]} != {n_feature} (from item 0)."
            )

    n_list = len(mats)
    x_new = np.linspace(0.0, 1.0, n_point, endpoint=True)

    stack = np.empty((n_list, n_point, n_feature), dtype=np.float64)

    # Helper: robust 1D interpolation that tolerates NaNs by ignoring them
    def _interp_nan_safe(x_old, y_old, x_new):
        valid = np.isfinite(y_old)
        if not np.any(valid):
            return np.full_like(x_new, np.nan, dtype=float)
        return np.interp(x_new, x_old[valid], y_old[valid])

    # Interpolate each matrix to the common grid
    for i, M in enumerate(mats):
        M = np.asarray(M, dtype=float)
        n_time_i = M.shape[0]
        x_old = np.linspace(0.0, 1.0, n_time_i, endpoint=True)

        # Column-wise interpolation (one feature at a time)
        for j in range(n_feature):
            stack[i, :, j] = _interp_nan_safe(x_old, M[:, j], x_new)

    mean = np.nanmean(stack, axis=0)
    std = np.nanstd(stack, axis=0, ddof=ddof)

    return {"grid": x_new, "stack": stack, "mean": mean, "std": std}


def interpolate_compute_dist_mat(mats, *, n_point=10, metric='euclidean', ddof=0):
    """
    Interpolate each matrix onto a common time grid, compute the pairwise
    distance matrix (over time points) for each interpolated matrix using
    ``compute_distance_lag``, and aggregate the list into mean and std.

    Parameters
    ----------
    mats : list of array_like
        Each element has shape (n_time_i, n_feature). All must share the same
        number of features. Time axis is assumed to span [0, 1].
    n_point : int, default=10
        Number of points on the common interpolation grid in [0, 1].
    metric : str or callable, default='euclidean'
        Distance metric passed through to ``compute_distance_lag``/``pdist``.
    ddof : int, default=0
        Delta degrees of freedom used for the standard deviation across the list
        of distance matrices.

    Returns
    -------
    dict
        {
          "D_list": list of (n_point, n_point) arrays,
          "D_mean": (n_point, n_point) array,
          "D_std":  (n_point, n_point) array,
        }
    """
    interp = interpolate_stacks(mats, n_point=n_point, ddof=ddof)
    stack = interp["stack"]  # (n_list, n_point, n_feature)

    D_list = []
    for i in range(stack.shape[0]):
        # For each interpolated matrix (time × feature), compute distance over time
        res = compute_distance_lag(stack[i], metric=metric, do_plot=False)
        D_list.append(res["D"])  # (n_point, n_point)

    if len(D_list) == 0:
        # Should not happen due to upstream validation, but handle defensively
        D_mean = np.full((n_point, n_point), np.nan)
        D_std = np.full((n_point, n_point), np.nan)
    else:
        D_stack = np.stack(D_list, axis=0)  # (n_list, n_point, n_point)
        D_mean = np.nanmean(D_stack, axis=0)
        D_std = np.nanstd(D_stack, axis=0, ddof=ddof)

    return {"D_list": D_list, "D_mean": D_mean, "D_std": D_std}

import numpy as np
from typing import Sequence, Iterable, Tuple, Dict, List, Optional

def labels_to_transition_matrix(
    labels: Sequence,
    mode: str = "frame",                 # "frame" or "segment"
    exclude: Optional[Iterable] = None,  # e.g., {-1} to drop noise
    smoothing: float = 0.0,              # Laplace add-α smoothing
    state_order: str = "sorted"          # "sorted" or "appearance"
) -> Tuple[np.ndarray, List]:
    """
    Compute empirical transition probabilities P[i,j] = Pr(s_{t+1}=j | s_t=i)
    from a sequence of cluster labels.

    Parameters
    ----------
    labels : sequence of hashables (ints/str)
    mode : "frame" or "segment"
        - "frame": count transitions for every adjacent pair
        - "segment": collapse consecutive identical labels first
    exclude : iterable of labels to drop entirely (e.g., {-1})
    smoothing : float
        Laplace add-α smoothing to counts before row-normalization.
        Set to 0.0 to avoid any smoothing.
    state_order : "sorted" or "appearance"
        Controls the row/column order of states in the returned matrix.

    Returns
    -------
    P : (K, K) ndarray
        Row-stochastic transition matrix.
    states : list
        The state labels corresponding to rows/cols of P.
    """
    arr = np.asarray(labels)

    # Drop excluded labels (e.g., noise = -1)
    if exclude is not None:
        mask = ~np.isin(arr, list(exclude))
        arr = arr[mask]

    # Not enough data?
    if arr.size == 0:
        return np.zeros((0, 0), dtype=float), []
    if mode not in {"frame", "segment"}:
        raise ValueError("mode must be 'frame' or 'segment'")

    # Segment mode: run-length encode to unique blocks
    if mode == "segment":
        keep = np.r_[True, arr[1:] != arr[:-1]]
        arr = arr[keep]

    # States and index mapping
    if state_order == "appearance":
        # preserve first-appearance order
        seen, states = set(), []
        for s in arr:
            if s not in seen:
                seen.add(s); states.append(s)
    else:  # "sorted"
        states = sorted(set(arr.tolist()))
    idx = {s: i for i, s in enumerate(states)}
    K = len(states)

    # Count transitions
    counts = np.zeros((K, K), dtype=float)
    for a, b in zip(arr[:-1], arr[1:]):
        i, j = idx[a], idx[b]
        counts[i, j] += 1.0

    # Laplace smoothing (optional)
    if smoothing > 0.0:
        counts = counts + smoothing

    # Row-normalize to probabilities
    row_sums = counts.sum(axis=1, keepdims=True)
    # avoid divide-by-zero (rows with no outgoing transitions)
    P = np.divide(counts, row_sums, out=np.zeros_like(counts), where=row_sums > 0)

    return P, states


def get_transmat_and_shuffle(labels_hd,n_shuffle=100,quantile=0.99):
    trans_mat,_ = labels_to_transition_matrix(labels_hd)
    trans_mat_sh_l = []
    for i in range(n_shuffle):
        reind=np.random.choice(np.arange(len(labels_hd)),size=len(labels_hd),replace=False)
        trans_mat_sh,_=labels_to_transition_matrix(labels_hd[reind])
        trans_mat_sh_l.append(trans_mat_sh)
    trans_mat_sh_l = np.array(trans_mat_sh_l) #n_shuffle x n_cluster x n_cluster
    trans_mat_sh_l_up = np.quantile(trans_mat_sh_l,quantile,axis=0)
    is_sig = trans_mat > trans_mat_sh_l_up
    res = {'trans_mat':trans_mat,'trans_mat_sh_l':trans_mat_sh_l,'trans_mat_sh_l_up':trans_mat_sh_l_up,'is_sig':is_sig}
    return res
    