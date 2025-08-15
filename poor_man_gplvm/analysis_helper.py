'''
helper functions for analysis after fitting the model
'''
import numpy as np
import pynapple as nap
import pandas as pd
import scipy.stats
import tqdm
import scipy.ndimage

def get_state_interval(p_l,p_thresh=0.8, merge_thresh=1,duration_thresh=2,):
    '''
    get the state interval from the posterior, by thresholding, get intervals, merge, filter for duration

    e.g. can use this to get the chunks of continuous fragmented states

    p_l: nap.Tsd, probability of some state (usually a dynamics like continuous / jump)
    p_thresh: threshold for considering dynamics to be continuous
    merge_thresh: threshold for merging adjacent intervals
    duration_thresh: threshold for filtering out short intervals

    return: 
    intv_merge: nap.IntervalSet, the interval of the chunks
    '''
    intv=p_l.threshold(p_thresh).time_support
    intv_merge = intv.merge_close_intervals(merge_thresh) # threshold for merging adjacent intervals
    ma=intv_merge[:,1]-intv_merge[:,0] > duration_thresh
    intv_merge = intv_merge[ma]
    return intv_merge

def get_peri_event_with_shuffle(feature_tsd,event_ts,n_shuffle=100,minmax=4,do_zscore=True):
    '''
    get peri event signal
    get peri event average over many shuffles -- circularly shuffle the event times
    feature_tsd: nap.Tsd, the feature to get peri event of
    event_ts: nap.Ts, the event times
    n_shuffle: int, the number of shuffles; if 0 then no shuffling
    minmax: int, the window for looking at peri event
    do_zscore: bool, if True, zscore across the time within each event

    return:
    peri_event: pd.DataFrame, n_event x n_time
    peri_event_sh: pd.DataFrame,n_shuffle x n_time or empty list
    '''
    event_ts = event_ts[(event_ts.t>minmax) & (event_ts.t<feature_tsd.t[-1]-minmax)]

    peri_event = nap.compute_perievent_continuous(timeseries=feature_tsd,tref=event_ts,minmax=minmax) # n_time x n_event
    peri_event = peri_event.as_dataframe().T # n_event x n_time
    if do_zscore:
        peri_event = scipy.stats.zscore(peri_event,axis=1)

    peri_event_sh_l=[]
    if n_shuffle > 0:
        for i in tqdm.trange(n_shuffle):
            event_ts_sh=nap.shift_timestamps(event_ts)
            try:
                event_ts_sh = event_ts_sh[(event_ts_sh.t>minmax) & (event_ts_sh.t<event_ts_sh.t[-1]-minmax)]
            except:
                import pdb; pdb.set_trace()

            peri_event_sh=nap.compute_perievent_continuous(timeseries=feature_tsd,tref=event_ts_sh,minmax=minmax) # n_time x n_event
            peri_event_sh = peri_event_sh.as_dataframe().T # n_event x n_time
            if do_zscore:
                peri_event_sh=scipy.stats.zscore(peri_event_sh,axis=1).mean(axis=0) # n_time
            else:
                peri_event_sh=peri_event_sh.mean(axis=0) # n_time
            peri_event_sh_l.append(peri_event_sh)
        peri_event_sh_l=pd.DataFrame(peri_event_sh_l) # n_shuffle x n_time
        
    
    return peri_event,peri_event_sh_l
    

    




def get_consecutive_pv_distance(X, smooth_window=None,metric="cosine"):
    """
    Compute consecutive population vector distances.

    Parameters
    ----------
    X : array-like, shape (T, N)
        Population activity matrix (time points × neurons).
    smooth_window : int or in seconds (if X is pynapple TsdFrame), optional
        Window size for smoothing the population activity.
    metric : {'cosine', 'correlation', 'euclidean'}
        Distance metric to use.

    Returns
    -------
    distances : ndarray, shape (T-1,)
        Distance between consecutive time points.
    """
    if isinstance(X,nap.TsdFrame):
        is_pyanppe=True
        if smooth_window is not None:
            X = X.smooth(smooth_window)
        X_ = X.d.astype(float)
        

    else:
        is_pyanppe=False
        if smooth_window is not None:
            X = scipy.ndimage.gaussian_filter1d(X.astype(float),smooth_window)
        X_ = X
    x1, x2 = X_[:-1], X_[1:]

    if metric == "euclidean":
        dist = np.linalg.norm(x2 - x1, axis=1)

    elif metric == "cosine":
        numerator = np.sum(x1 * x2, axis=1)
        norm1 = np.linalg.norm(x1, axis=1)
        norm2 = np.linalg.norm(x2, axis=1)
        denom = norm1 * norm2
        with np.errstate(invalid='ignore', divide='ignore'):
            sim = np.divide(numerator, denom, out=np.zeros_like(numerator), where=denom > 0)
        dist = 1 - sim
        is_zero1 = norm1 <= 1e-12
        is_zero2 = norm2 <= 1e-12
        both_zero = is_zero1 & is_zero2
        one_zero = is_zero1 ^ is_zero2
        dist[both_zero] = 0.0
        dist[one_zero] = 2.0

    elif metric == "correlation":
        x1_centered = x1 - x1.mean(axis=1, keepdims=True)
        x2_centered = x2 - x2.mean(axis=1, keepdims=True)
        numerator = np.sum(x1_centered * x2_centered, axis=1)
        norm1 = np.linalg.norm(x1_centered, axis=1)
        norm2 = np.linalg.norm(x2_centered, axis=1)
        denom = norm1 * norm2
        with np.errstate(invalid='ignore', divide='ignore'):
            sim = np.divide(numerator, denom, out=np.zeros_like(numerator), where=denom > 0)
        dist = 1 - sim
        is_zero1 = norm1 <= 1e-12
        is_zero2 = norm2 <= 1e-12
        both_zero = is_zero1 & is_zero2
        one_zero = is_zero1 ^ is_zero2
        dist[both_zero] = 0.0
        dist[one_zero] = 2.0

    else:
        raise ValueError(f"Unknown metric: {metric}")

    if is_pyanppe:
        dist = nap.Tsd(t=X.t[1:],d=dist)
    return dist




#### after doing peri event, regression analysis ####
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

def fit_time_prepost_interaction(
    df_wide: pd.DataFrame,
    time=None,
    repeat_name: str = "repeat",
    response_name: str = "y",
    cov: str = "cluster",   # "cluster" (by repeat) or "HC1"
):
    """
    Fit: response ~ time_within * C(is_post)
    - time_within: z-scored within pre (<0) and post (>=0) separately
    - is_post: 1 if time>=0 else 0

    Returns dict with:
      - summary_df: tidy table of slopes & intercepts
          index: ["slope_pre","slope_post","slope_diff",
                  "intercept_pre","intercept_post","intercept_diff"]
          columns: ["estimate","std_value","pvalue","ci_low","ci_high"]
        * 'std_value' is the t-statistic (estimate / SE)
      - params, bse, pvalues, conf_int, rsquared, rsquared_adj, f_pvalue, nobs, cov_type
      - coef_pre, coef_post, p_interaction, p_intercept_diff
      - model (statsmodels results), data_long (for convenience)
    """

    # ----- 1) Build long-format table -----
    wide = df_wide.copy()

    # Get time vector
    if time is None:
        try:
            t = pd.to_numeric(wide.columns, errors="raise").astype(float)
        except Exception:
            raise ValueError("Cannot parse df_wide.columns as numeric times; pass `time` explicitly.")
    else:
        t = np.asarray(time, dtype=float)
        if len(t) != wide.shape[1]:
            raise ValueError("`time` length must match number of columns in df_wide.")

    wide.columns = t
    long = (
        wide.rename_axis(index=repeat_name, columns="time")
            .stack()
            .reset_index(name=response_name)
    )

    if not ((long["time"] < 0).any() and (long["time"] >= 0).any()):
        raise ValueError("Time grid must include both pre (<0) and post (>=0) samples.")

    # ----- 2) Regressors -----
    long["is_post"] = (long["time"] >= 0).astype(int)

    def _z_by_side(x):
        s = x.std(ddof=0)
        return (x - x.mean()) / s if s > 0 else x * 0.0

    long["time_within"] = long.groupby("is_post")["time"].transform(_z_by_side)

    # ----- 3) Fit OLS with interaction -----
    formula = f"{response_name} ~ time_within * C(is_post)"
    if cov == "cluster":
        res = smf.ols(formula, data=long).fit(
            cov_type="cluster", cov_kwds={"groups": long[repeat_name]}
        )
    elif cov == "HC1":
        res = smf.ols(formula, data=long).fit(cov_type="HC1")
    else:
        raise ValueError("cov must be 'cluster' or 'HC1'.")

    # Name helpers
    p_time = "time_within"
    p_group = "C(is_post)[T.1]"
    p_interact = "time_within:C(is_post)[T.1]"

    # Convenience function for linear contrasts -> tidy stats
    def _lincon(expr):
        tt = res.t_test(expr)
        est = float(np.atleast_1d(tt.effect)[0])
        tval = float(np.atleast_1d(tt.tvalue)[0])
        pval = float(np.atleast_1d(tt.pvalue)[0])
        ci = tt.conf_int()
        low, high = float(ci[0,0]), float(ci[0,1])
        return {"estimate": est, "std_value": tval, "pvalue": pval, "ci_low": low, "ci_high": high}

    # Slopes
    stats_slope_pre  = _lincon(f"{p_time} = 0")                  # β1
    stats_slope_post = _lincon(f"{p_time} + {p_interact} = 0")   # β1 + β3
    stats_slope_diff = _lincon(f"{p_interact} = 0")              # β3

    # Intercepts
    stats_int_pre   = _lincon("Intercept = 0")                   # β0
    stats_int_post  = _lincon(f"Intercept + {p_group} = 0")      # β0 + β2
    stats_int_diff  = _lincon(f"{p_group} = 0")                  # β2

    summary_df = pd.DataFrame.from_dict(
        {
            "slope_pre":       stats_slope_pre,
            "slope_post":      stats_slope_post,
            "slope_diff":      stats_slope_diff,
            "intercept_pre":   stats_int_pre,
            "intercept_post":  stats_int_post,
            "intercept_diff":  stats_int_diff,
        },
        orient="index",
    )

    # Other useful outputs
    ci_full = res.conf_int()
    ci_full.columns = ["low", "high"]

    # Pre/Post slope point estimates for quick access (same as summary_df values)
    beta0 = res.params.get("Intercept", np.nan)
    beta1 = res.params.get(p_time, np.nan)
    beta2 = res.params.get(p_group, np.nan)
    beta3 = res.params.get(p_interact, np.nan)

    coef_pre = {"intercept": beta0, "slope": beta1, "p_slope": stats_slope_pre["pvalue"]}
    coef_post = {"intercept": beta0 + beta2, "slope": beta1 + beta3, "p_slope": stats_slope_post["pvalue"]}

    return {
        "summary_df": summary_df,            # << clean table you asked for
        "params": res.params,
        "bse": res.bse,
        "pvalues": res.pvalues,
        "conf_int": ci_full,
        "rsquared": res.rsquared,
        "rsquared_adj": res.rsquared_adj,
        "f_pvalue": res.f_pvalue,
        "nobs": int(res.nobs),
        "cov_type": res.cov_type,
        "coef_pre": coef_pre,
        "coef_post": coef_post,
        "p_interaction": stats_slope_diff["pvalue"],     # = test of β3
        "p_intercept_diff": stats_int_diff["pvalue"],    # = test of β2
        "model": res,
        "data_long": long,
    }
