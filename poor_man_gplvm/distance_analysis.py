'''
analysis of distance
'''

import numpy as np
import pandas as pd

import numpy as np
from scipy.spatial.distance import cdist
import statsmodels.api as sm

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

