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
            event_ts_sh = event_ts_sh[(event_ts_sh.t>minmax) & (event_ts_sh.t<feature_tsd.t[-1]-minmax)]
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




