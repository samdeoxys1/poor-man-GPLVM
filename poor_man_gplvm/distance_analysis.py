'''
analysis of distance
'''

import numpy as np
import pandas as pd

import numpy as np
from scipy.spatial.distance import cdist

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

def distance_vs_label_regression(
    D, labels, *, bin_edges=None, nbins=50, binning='uniform', z=1.96, return_pairs_df=True
):
    """
    Build the upper-triangle dataset (distance vs |Δlabel|), run OLS,
    and compute binned mean/std/CI vs label distance.

    Returns:
      pairs_df : DataFrame (i,j, label_i, label_j, label_dist, dist) [optional]
      summary  : dict(intercept, slope, r, r2)
      binned   : DataFrame (bin_left, bin_right, bin_center, n, mean, std, ci_low, ci_high)
      bin_edges: ndarray used (reusable for shuffles)
      kept_idx : indices kept after dropping NaN labels (map to D submatrix)
    """
    Dv, lv, iu, ju, x, y, i_orig, j_orig, kept_idx = _upper_triangle_pairs(D, labels)

    # Regression
    summary = _linregress_np(x, y)

    # Binned stats
    binned, edges = _bin_stats(x, y, bin_edges=bin_edges, nbins=nbins, binning=binning, z=z)

    # Pairwise DF (upper triangle only)
    pairs_df = None
    if return_pairs_df:
        pairs_df = pd.DataFrame({
            "i": i_orig, "j": j_orig,
            "label_i": labels[i_orig], "label_j": labels[j_orig],
            "label_dist": x, "dist": y
        })
    
    res = dict(pairs_df=pairs_df, summary=summary, binned=binned, edges=edges, kept_idx=kept_idx)

    return res

def shuffle_test_distance_vs_label(
    D, labels, *, n_shuffles=1000, rng=None, bin_edges=None, nbins=50, binning='uniform'
):
    """
    Shuffle test: randomly permute rows/cols of D (labels stay put),
    then recompute regression and binned means each time.

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
    # Build once (obs)
    pairs_df, summary_obs, binned_obs, edges, kept_idx = distance_vs_label_regression(
        D, labels, bin_edges=bin_edges, nbins=nbins, binning=binning, return_pairs_df=False
    )
    # Reuse kept submatrix and upper-tri indices
    Dv, lv, iu, ju, x, y, *_ = _upper_triangle_pairs(D, labels)  # recompute to get iu,ju,x
    nb = len(edges) - 1
    slopes = np.empty(n_shuffles); intercepts = np.empty(n_shuffles); r2s = np.empty(n_shuffles)
    binned_means = np.full((n_shuffles, nb), np.nan)

    n = Dv.shape[0]
    for s in range(n_shuffles):
        perm = rng.permutation(n)
        Dp = Dv[np.ix_(perm, perm)]
        y_shuf = Dp[iu, ju]
        # regression
        reg = _linregress_np(x, y_shuf)
        slopes[s] = reg['slope']; intercepts[s] = reg['intercept']; r2s[s] = reg['r2']
        # binned means on fixed edges
        binned_s, _ = _bin_stats(x, y_shuf, bin_edges=edges)
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
