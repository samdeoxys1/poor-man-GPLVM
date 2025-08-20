'''
For Yiyao Zhang's ACH dataset

need:
- transition events
    - detecting ACh bouts
        - further subdivide by sleep stage
    - continuous and fragmented chunks (deprecated)
- feature
    - (ach), pop fr, consec pv difference, p_jump/continuous
- peri transition feature average and shuffle
- tuning
    - jump probability vs sleep stage
- representational analysis 
    - (here the observation is the latent during continuous states tend to be similar within a NREM interval, but can be different across NREM intervals)
    - feature: mean_latent_posterior, pv
    - unit of comparison: ach bouts, ripples
    - method: regression, dist ~ time + C(same nrem interval)
        - null: shuffle the nrem interval label, so that if the two ach bouts used to be in the same nrem interval, they might not be after shuffle. But keep the time label correct?
            - can do one for time too
'''

import numpy as np
import scipy
import pynapple as nap
import dill
import poor_man_gplvm.analysis_helper as ah
import poor_man_gplvm.plot_helper as ph
import matplotlib.pyplot as plt
import seaborn as sns
import os,copy
from scipy.spatial.distance import squareform, pdist
import poor_man_gplvm.distance_analysis as da
import pandas as pd

# helper function to get list of processed decoding result from em_res_l (multiple em fit results)
def get_decode_res_l_from_em_res_l(em_res_l,t_l=None):
    decode_res_l = []
    for em_res in em_res_l:
        log_posterior_final=em_res['log_posterior_final']
        post_latent_marg = np.exp(scipy.special.logsumexp(log_posterior_final,axis=1))
        post_dynamics_marg = np.exp(scipy.special.logsumexp(log_posterior_final,axis=2))
        if t_l is None:
            t_l = np.arange(post_latent_marg.shape[0])
        decode_res_one = {'posterior_latent_marg':nap.TsdFrame(d=post_latent_marg,t=t_l),'posterior_dynamics_marg':nap.TsdFrame(d=post_dynamics_marg,t=t_l)}
        
        decode_res_l.append(decode_res_one)
    return decode_res_l

def load_data_and_fit_res(data_path,fit_res_path):
    data_load_res = dill.load(open(data_path,'rb'))
    fit_res_load_res = dill.load(open(fit_res_path,'rb'))
    em_res_l = fit_res_load_res['em_res_l']
    t_l = data_load_res['t_l']
    decode_res_l = get_decode_res_l_from_em_res_l(em_res_l,t_l)
    model_eval_result = fit_res_load_res['metric_eval_result']
    model_index=model_eval_result['metric_overall']['best_index']
    model_fit = fit_res_load_res['model_fit_l'][model_index]
    decode_res = decode_res_l[model_index]

    prep_res = {**data_load_res,**decode_res,'model_fit':model_fit}

    return prep_res

def find_ach_ramp_onset(ach_data,smooth_win=1,finite_diff_window_s=1,height=0.05,do_zscore=True,detrend_cutoff=None):
    '''
    current method: gaussian smooth, finite difference for slope value(t+finite_diff_window_s) - value(t))/finite_diff_window_s, smooth the slope, find peaks with some threshold, 
    detrend_cutoff: float, optional, usually 0.01

    '''
    if do_zscore:
        t_l = ach_data.t
        ach_data = scipy.stats.zscore(ach_data)
        ach_data = nap.Tsd(d=ach_data,t=t_l)
    if detrend_cutoff is not None:
        ach_data = ach_data - nap.apply_lowpass_filter(ach_data,detrend_cutoff).d
    if smooth_win is not None:
        ach_data_smth = ach_data.smooth(smooth_win) # not really used, only for examination
    else:
        ach_data_smth = ach_data
    
    
    finite_diff_window = int(finite_diff_window_s / np.median(np.diff(ach_data.t)))
    extended_data = np.concatenate([ach_data.d,np.ones(finite_diff_window) * ach_data.d[-1]])
    slope = (extended_data[finite_diff_window:] - extended_data[:-finite_diff_window]) / finite_diff_window_s
    slope = nap.Tsd(d=slope,t=ach_data.t)[:-finite_diff_window].smooth(smooth_win)
    

    peaks,metadata=scipy.signal.find_peaks(slope,height=height)
    peak_heights = metadata['peak_heights']
    peak_heights = nap.Tsd(d=peak_heights,t=slope.t[peaks])

    slope_peak_time = nap.Ts(slope.t[peaks])
    ach_ramp_onset_res = {'ach_ramp_onset':slope_peak_time,'slope':slope,'slope_peak_time':slope_peak_time,'peak_heights':peak_heights}
    return ach_ramp_onset_res


def find_ach_ramp_onset_old(ach_data,smooth_win=1,height=0.05,do_zscore=True,detrend_cutoff=None,shift=-1.):
    '''
    current method: gaussian smooth, finite difference for slope, find peaks with some threshold, shift

    detrend_cutoff: float, optional, usually 0.01
    shift: float, optional, shift the onset by this amount in second, to correct for the detection not using an acausal window
    '''
    if do_zscore:
        t_l = ach_data.t
        ach_data = scipy.stats.zscore(ach_data)
        ach_data = nap.Tsd(d=ach_data,t=t_l)
    if detrend_cutoff is not None:
        ach_data = ach_data - nap.apply_lowpass_filter(ach_data,detrend_cutoff).d
    if smooth_win is not None:
        ach_data_smth = ach_data.smooth(smooth_win)
    else:
        ach_data_smth = ach_data
    
    slope = ach_data_smth.derivative()
    peaks,metadata=scipy.signal.find_peaks(slope,height=height)
    peak_heights = metadata['peak_heights']
    peak_heights = nap.Tsd(d=peak_heights,t=slope.t[peaks])
    ach_ramp_onset = nap.Ts(slope.t[peaks]+shift)
    ach_ramp_onset_res = {'ach_ramp_onset':ach_ramp_onset,'slope':slope,'ach_data_smth':ach_data_smth,'ach_data':ach_data,'peak_heights':peak_heights}
    return ach_ramp_onset_res

def event_triggered_analysis(feature,event_ts,n_shuffle=10,minmax=4,do_zscore=False,test_win=1,do_plot=False,fig=None,ax=None,ylabel=None,title=None,ylim=None):
    '''
    do event triggered analysis on a feature, as well as circularly shuffle the events to get null
    for test statistics, get the pre post difference, correlation with time within pre/post, regression analysis with time_within and C(is_post)
    optionally plot
    feature: Tsd, the feature to be analyzed
    event_ts: Ts, the event timestamps
    n_shuffle: int, number of shuffles
    do_zscore: bool, whether to zscore the feature within each peri-event window
    minmax: [-minmax,minmax] will be the peri event window, in second
    test_win: in second, a smaller window to do the pre post difference test
    '''
    peri_event_res=ah.get_peri_event_with_shuffle(feature,event_ts,n_shuffle=n_shuffle,minmax=minmax,do_zscore=do_zscore)
    ach_peri,ach_peri_shuffle = peri_event_res
    analysis_res={'feature':ach_peri,'shuffle':ach_peri_shuffle}

    # wilcoxon test
    toplot = toplot_z = ach_peri
    pre=toplot.loc[:,(toplot.columns<0)&(toplot.columns>-test_win)].mean(axis=1)
    post=toplot.loc[:,(toplot.columns>0)&(toplot.columns<test_win)].mean(axis=1)

    diff = post-pre
    diff_median=diff.median()
    print(f'n={len(diff)}')
    print(f'diff median: {diff_median}')
    analysis_res['diff_median'] = diff_median

    effect_size=diff.mean() / diff.std()
    print(f'effect size {effect_size}')
    analysis_res['effect_size'] = effect_size

    wc_res=scipy.stats.wilcoxon(diff)
    analysis_res['wc_res'] = wc_res
    print(wc_res)
    
    # correlation
#     toplot_z=scipy.stats.zscore(toplot,axis=1)
    corr_res={}
    to_corr=toplot_z.loc[:,toplot_z.columns<0].melt()
    corr_res['pre']=scipy.stats.pearsonr(to_corr['variable'],to_corr['value'])
    to_corr=toplot_z.loc[:,toplot_z.columns>0].melt()
    corr_res['post']=scipy.stats.pearsonr(to_corr['variable'],to_corr['value'])
    analysis_res['corr_res']=corr_res
    print(corr_res)
    
    
    # regression
    reg_res=ah.fit_time_prepost_interaction(ach_peri)
    reg_res_shuffle=ah.fit_time_prepost_interaction(ach_peri_shuffle)
    analysis_res['reg_res'] = reg_res
    analysis_res['reg_res_shuffle'] = reg_res_shuffle
    print(reg_res['summary_df'])
    
    
    # plot
    if do_plot:
        if ax is None:
            fig,ax=plt.subplots(figsize=(1,1.5))
        fig,ax=ph.plot_mean_error_plot(toplot_z,mean_axis=0,ax=ax,fig=fig)
        fig,ax=ph.plot_mean_error_plot(ach_peri_shuffle,mean_axis=0,fig=fig,ax=ax,color='grey')
        ax.set_xlabel('Time (s)')
        if ylabel is not None:
            ax.set_ylabel(ylabel)
        if title is not None:
            ax.set_title(title)
        if ylim is not None:
            ax.set_ylim(ylim)
        ax=ph.set_two_ticks(ax,apply_to='y')
        ax=ph.set_symmetric_ticks(ax,apply_to='x')
        sns.despine(ax=ax)

        return analysis_res,fig,ax

    return analysis_res

##### main analysis: event triggered analysis #####
def event_triggered_analysis_multiple_feature_event(feature_d,event_ts_d,n_shuffle=10,minmax=4,do_zscore=False,test_win=1,do_plot=False,fig=None,ax=None,ylabel_d={},title_d={},ylim_d={}):
    '''
    wrapper of event_triggered_analysis for multiple features and events
    '''
    analysis_res_d = {}
    ylabel_d_ = {k:k for k in feature_d.keys()} # by default use the feature name as ylabel, but can be overridden by ylabel_d
    title_d_ = {k:k for k in event_ts_d.keys()} # by default use the event name as title, but can be overridden by title_d
    ylim_d_ = {k:None for k in feature_d.keys()} # by default use None as ylim, but can be overridden by ylim_d

    ylabel_d_.update(ylabel_d)
    title_d_.update(title_d)
    ylim_d_.update(ylim_d)
    if do_plot:
        fig_d = {}
        ax_d = {}
        
    for feat_name,feat in feature_d.items():
        print(f'====Feature: {feat_name}====')
        for event_name,event_ts in event_ts_d.items():
            print(f'====Event: {event_name}====')
            if do_plot:
                analysis_res,fig_,ax_=event_triggered_analysis(feat,event_ts,n_shuffle=n_shuffle,minmax=minmax,do_zscore=do_zscore,test_win=test_win,do_plot=do_plot,fig=fig,ax=ax,ylabel=ylabel_d_[feat_name],title=title_d_[event_name],ylim=ylim_d_[feat_name])
            else:
                analysis_res = event_triggered_analysis(feat,event_ts,n_shuffle=n_shuffle,minmax=minmax,do_zscore=do_zscore,test_win=test_win)
            analysis_res_d[feat_name,event_name] = analysis_res
            if do_plot:
                fig_d[feat_name,event_name] = fig_
                ax_d[feat_name,event_name] = ax_
    if do_plot:
        return analysis_res_d,fig_d,ax_d
    return analysis_res_d

from sklearn.cluster import KMeans
def cluster_peri_event(peri_event,n_cluster=2,do_plot=False,fig=None,ax=None,do_zscore=False):
    '''
    peri_event: n_sample x n_time
    '''
    # for removing change in baseline and overall variance, if i only want to cluster based on direction of change
    # when analyzing and showing, go back to original scale
    if do_zscore: 
        mean_peri_event = peri_event.mean(axis=0)
        std_peri_event = peri_event.std(axis=0)
        peri_event_z = (peri_event - mean_peri_event) / std_peri_event
    else:
        peri_event_z = peri_event
    kmeans = KMeans(n_clusters=n_cluster,random_state=0).fit(peri_event_z)
    peri_event_cluster_mean_d ={}
    peri_event_per_cluster_d={}
    for i in range(n_cluster):
        peri_event_per_cluster_d[i] = peri_event[kmeans.labels_==i]
        peri_event_cluster_mean_d[i] = peri_event_per_cluster_d[i].mean(axis=0)
    to_return = {'peri_event_cluster_mean_d':peri_event_cluster_mean_d,'peri_event_per_cluster_d':peri_event_per_cluster_d,'kmeans':kmeans}
    if do_plot:
        for i in range(n_cluster):
            fig,ax = ph.plot_mean_error_plot(peri_event_per_cluster_d[i],mean_axis=0,fig=fig,ax=ax,color=f'C{i}')
            ax.set_title(f'Cluster {i}')
            ax.set_xlabel('Time (s)')
        return to_return,fig,ax
    return to_return

def manual_cluster_peri_event(peri_event,time_window=(-2,0),bin=None,n_cluster=2,do_plot=False,fig=None,ax=None,do_zscore=False):
    '''
    manual cluster based on quantile of average within some time window
    '''
    if do_zscore:
        mean_peri_event = peri_event.mean(axis=0)
        std_peri_event = peri_event.std(axis=0)
        peri_event_z = (peri_event - mean_peri_event) / std_peri_event
    else:
        peri_event_z = peri_event
    peri_event_sub = peri_event_z.loc[:,(peri_event_z.columns >=time_window[0]) & (peri_event_z.columns <=time_window[1])]
    temporal_mean = peri_event_sub.mean(axis=1)
    if bin is None:
        temporal_mean_quantile = pd.qcut(temporal_mean,n_cluster,labels=False)
    else:
        temporal_mean_quantile = pd.cut(temporal_mean,bin,labels=False)
        n_cluster = len(bin)-1
        
    peri_event_per_cluster_d = {}
    peri_event_per_cluster_mean_d = {}
    for i in range(n_cluster):
        peri_event_per_cluster_d[i] = peri_event.loc[temporal_mean_quantile==i]
        peri_event_per_cluster_mean_d[i] = peri_event_per_cluster_d[i].mean(axis=0)
        
    to_return = {'temporal_mean_quantile':temporal_mean_quantile,'temporal_mean':temporal_mean,'peri_event_per_cluster_d':peri_event_per_cluster_d,'peri_event_per_cluster_mean_d':peri_event_per_cluster_mean_d}
    if do_plot:
        for i in range(n_cluster):
            fig,ax = ph.plot_mean_error_plot(peri_event_per_cluster_d[i],fig=fig,ax=ax,color=f'C{i}')
            ax.set_title(f'Cluster {i}')
            ax.set_xlabel('Time (s)')
        return to_return,fig,ax
    return to_return


def prep_feature_d(prep_res,consec_pv_dist_metric='correlation',continuous_dynamics_ind=0,jump_dynamics_ind=1,feature_to_include=['ach','pop_fr','consec_pv_dist','p_continuous','p_jump'],mask_d={}):
    '''
    prepare the features used for peri event analysis
    This function is updated when we care about different features
    '''
    
    
    spike_mat_sub = prep_res['spike_mat_sub']
    pop_fr = spike_mat_sub.mean(axis=1) / np.median(np.diff(spike_mat_sub.t))
    consec_pv_dist=ah.get_consecutive_pv_distance(spike_mat_sub,metric=consec_pv_dist_metric)

    p_continuous = prep_res['posterior_dynamics_marg'][:,continuous_dynamics_ind]
    p_jump = prep_res['posterior_dynamics_marg'][:,jump_dynamics_ind]
    
    feature_d = {}
    if 'ach' in feature_to_include:
        ach = prep_res['fluo_data']['ACh']
        feature_d['ach'] = ach
    if 'pop_fr' in feature_to_include:
        feature_d['pop_fr'] = pop_fr
    if 'pv' in feature_to_include:
        feature_d['pv'] = spike_mat_sub
    if 'p_latent' in feature_to_include:
        
        ma = mask_d.get('p_latent',None)
        if ma is None:
            feature_d['p_latent'] = prep_res['posterior_latent_marg']
        else:
            feature_d['p_latent'] = prep_res['posterior_latent_marg'][ma]
    if 'consec_pv_dist' in feature_to_include:
        feature_d['consec_pv_dist'] = consec_pv_dist
    if 'p_continuous' in feature_to_include:
        feature_d['p_continuous'] = p_continuous
    if 'p_jump' in feature_to_include:
        feature_d['p_jump'] = p_jump
    return feature_d

def turn_sleep_state_tsd_to_interval(sleep_state_index,sleep_state_label_d={'Wake':0,'NREM':2,'REM':4}):
    '''
    turn numerically coded sleep state time series to interval for each state
    sleep_state_index: Tsd, n_time, the sleep state index
    '''
    sleep_state_intv_d = {}
    for label,label_num in sleep_state_label_d.items():
        intv=(sleep_state_index==label_num).threshold(0.5).time_support
        sleep_state_intv_d[label] = intv
    return sleep_state_intv_d

def segregate_event_ts_by_sleep_state(event_ts_d,sleep_state_label_d):
    '''
    segregate the event timestamps by sleep state
    event_ts_d: dict of Ts, key is the event name, value is the event timestamps; input dictionary to make keys more informative
    sleep_state_label_d: dict, key is the sleep state label, value is the interval
    return: dict, key is the sleep state label, value is the event timestamps
    '''
    event_ts_d_ = {}
    for event_name,event_ts in event_ts_d.items():
        for label,intv in sleep_state_label_d.items():
            event_ts_d_[event_name+'_'+label] = event_ts.restrict(intv)
    return event_ts_d_


def get_post_pre_diff(df,center=0,test_win=None):
    '''
    assume columne is peri event time, 
    center: the time of the event, default is 0
    '''
    if test_win is None:
        test_win = np.minimum(center - df.columns.min(),df.columns.max() - center)
    pre = df.loc[:,(df.columns<center)&(df.columns>=center-test_win)].mean(axis=1)
    post = df.loc[:,(df.columns>center)&(df.columns<=center+test_win)].mean(axis=1)
    diff = post-pre
    diff_median=diff.median()
    effect_size=diff.mean() / diff.std()
    dres = {'pre':pre,'post':post,'diff':diff,'diff_median':diff_median,'effect_size':effect_size}
    return dres

def test_pre_post_against_shuffle(df,df_shuffle,center=0,test_win=None):
    '''
    test the pre post difference against shuffle
    df: pd.DataFrame, the feature to be analyzed, n_sample x n_time; the n_sample here is mean/median over, so doesn't matter in significance
    df_shuffle: pd.DataFrame, the shuffle of the feature, n_shuffle x n_time
    center: the time of the event, default is 0
    test_win: in second, a smaller window to do the pre post difference test
    '''
    dres=get_post_pre_diff(df,center=center,test_win=test_win)
    dres_shuffle=get_post_pre_diff(df_shuffle,center=center,test_win=test_win)
    diff = dres['diff_median']
    diff_shuffle= dres_shuffle['diff']
    p = np.mean(diff >= diff_shuffle)
    test_res = {'diff':diff,'diff_shuffle':diff_shuffle,'p':p,'effect_size':dres['effect_size']}
    return test_res



# ===== main analysis: distance vs label distance ===== #
# need a dict of intervals (eg ACh bouts, ripples), a dict of features (e.g. posterior, pv), a dict of labels (e.g. indices of NREM intervals)
# for each interval, get the mean feature within
# get distance between each pair of intervals
# get label distance
# do regression and shuffle test
# plot tentative: dist matrix marked by label transition; mean feature in interval indices, marked by label transition; regression weight with shuffle

def get_mean_feature_in_interval(feature_d,interval_d):
    '''
    get the mean feature within each interval
    feature_d: dict, key is the feature name, value is the feature nap.Tsd
    interval_d: dict, key is the interval name, value is the nap.IntervalSet, or nap.Tsd which is a mask
    
    '''
    mean_feature_d = {}
    
    for feat_name,feat in feature_d.items():
        t_l = []
        for interval_name,interval in interval_d.items():
            if isinstance(interval,nap.IntervalSet):
                
                mean_feat = []
                for intv in interval:
                    
                    feat_sub_intv = feat.restrict(intv)
                    if feat_sub_intv.shape[0]>0:
                        mean_feat.append(feat_sub_intv.mean(axis=0))
                        t_l.append(feat_sub_intv.t[0])
                    
                mean_feature_d[feat_name,interval_name] = nap.TsdFrame(d=mean_feat,t=t_l)
            else:            
               mean_feature_d[feat_name,interval_name] = feat[interval.d]
        
    return mean_feature_d

def get_distance_matrix(mean_feature_d,metric_d={'pv':'correlation'}):
    '''
    get distance between each pair of intervals for the mean feature within each interval
    default: posterior, using wasserstein-1 distance
    '''
    dist_d = {}
    for k,val in mean_feature_d.items():
        if 'pv' in k:
            dist_d[k] = squareform(pdist(val.d,metric=metric_d['pv']))
        else:
            # dist_d[k],C = da.w1_cdf_distance_matrix(val.d)
            dist_d[k] = squareform(pdist(val.d, metric="jensenshannon"))
    return dist_d


def feature_distance_vs_label_distance_analysis(prep_res,label_intv,ach_intv=None,ach_onset=None,ach_extend_win=1,feature_key_l=['p_latent','pv'],interval_key_l=['ACh_onset','ripple'],n_shuffles=200,label_distance_threshold=None,mask_d={}):
    '''
    ach_onset: Ts, the ACh onset timestamps
    ach_extend_win: int, the window to extend the ACh onset into interval, in second
    label_intv: the nap.IntervalSet whose index is the label, e.g. NREM intervals
    feature_key_l: list, the features to include
    interval_key_l: list, the intervals to include
    '''
    feature_d = prep_feature_d(prep_res,feature_to_include=feature_key_l,mask_d=mask_d)
    interval_d = {}
    if 'ACh_onset' in interval_key_l:
        if ach_intv is None:
            assert ach_onset is not None
            ach_onset_sub = ach_onset.restrict(label_intv)
            interval_d['ACh_onset'] = nap.IntervalSet(ach_onset_sub.t,ach_onset_sub.t+ach_extend_win)
        else:
            interval_d['ACh_onset'] = ach_intv
    if 'ripple' in interval_key_l:
        if 'is_ripple' in prep_res:
            interval_d['ripple'] = prep_res['is_ripple'].restrict(label_intv)
        else:
            print('ripple interval not found, skipping')
    mean_feature_d = get_mean_feature_in_interval(feature_d,interval_d)
    
    dist_d = get_distance_matrix(mean_feature_d)
    
    analysis_res_d = {}
    which_interval_index_d = {}
    when_label_change_d = {}
    for key,feat in mean_feature_d.items():
        which_interval_index = label_intv.in_interval(feat)
        which_interval_index_d[key] = which_interval_index
        when_label_change = np.diff(which_interval_index)>0
        when_label_change = np.concatenate([[0],when_label_change])
        when_label_change_d[key] = when_label_change
        shuffle_res=da.shuffle_test_distance_vs_label(dist_d[key], which_interval_index, n_shuffles=n_shuffles, rng=None,label_distance_threshold=label_distance_threshold,timestamps=feat.t)
        analysis_res_d[key] = shuffle_res

    feature_dist_vs_label_dist_res = {'dist_d':dist_d,'analysis_res_d':analysis_res_d,'mean_feature_d':mean_feature_d,'interval_d':interval_d,'which_interval_index_d':which_interval_index_d,'when_label_change_d':when_label_change_d}
    
    return feature_dist_vs_label_dist_res

def within_nrem_interval_ach_induced_latent_ramp_analysis():
    '''
    within a nrem interval, test whether ach induced latent ramp up / down
    '''
    pass

# ====all together====#
def main(data_path=None,fit_res_path=None,prep_res=None,
    ach_ramp_kwargs = {'height':0.05,'detrend_cutoff':None,'smooth_win':1,'finite_diff_window_s':1},
    event_triggered_analysis_kwargs = {'n_shuffle':100,'minmax':4,'do_zscore':False,'test_win':2,'do_plot':True},
    res_data_save_path = None,
    res_fig_save_path = None,
):
    '''
    save_path: convention: {session folder}/py_data/post_fit_quantification/ or {session folder}/py_figure/post_fit_quantification/
    '''
    
    # load data and fit res
    if prep_res is None:
        assert data_path is not None and fit_res_path is not None
        prep_res = load_data_and_fit_res(data_path,fit_res_path)
    sleep_state_index = prep_res['sleep_state_index']
    if 'fluo_data' in prep_res:
        has_ach=True
        has_stim=False
        ach = prep_res['fluo_data']['ACh']
        ach_onset_res=find_ach_ramp_onset(ach,**ach_ramp_kwargs)
    else:
        has_ach=False
        has_stim=True
        is_stim = prep_res['is_stim']

    # prepare features
    if has_ach:
        feature_to_include=['p_continuous','ach','pop_fr','consec_pv_dist']
    else:
        feature_to_include=['p_continuous','pop_fr','consec_pv_dist']
    feature_d = prep_feature_d(prep_res,feature_to_include=feature_to_include)
    print(feature_d.keys())

    # prepare event timestamps
    sleep_state_intv=turn_sleep_state_tsd_to_interval(sleep_state_index,)
    event_ts = ach_onset_res['ach_ramp_onset']
    event_ts_d = {'ACh_onset':event_ts}
    event_ts_by_sleep=segregate_event_ts_by_sleep_state(event_ts_d,sleep_state_intv)

    # do event triggered analysis
    analysis_res_d,fig_d,ax_d = event_triggered_analysis_multiple_feature_event(feature_d,event_ts_by_sleep,**event_triggered_analysis_kwargs)
    
    
    if res_data_save_path is not None:
        os.makedirs(os.path.dirname(res_data_save_path),exist_ok=True)
        # np.savez(res_data_save_path,**analysis_res_d)
        dill.dump(analysis_res_d,open(res_data_save_path,'wb'))
    if res_fig_save_path is not None and event_triggered_analysis_kwargs['do_plot']:
        os.makedirs(os.path.dirname(res_fig_save_path),exist_ok=True)
        for feat_name,event_name in fig_d.keys():
            fig_fn = f'{feat_name};{event_name}_peri_event'
            ph.save_fig(fig_d[feat_name,event_name],fig_fn,res_fig_save_path)
            plt.close(fig_d[feat_name,event_name])
    

    return analysis_res_d


def gather_feature_shuffle_across_sessions(analysis_res_d_allsess,prep_fig_save_dir='./'):
    '''
    analysis_res_d_allsess: dict, key is the feature_key, value is the analysis_res_d from event_triggered_analysis_multiple_feature_event of all sessions
    '''
    all_feature_allsess = {} # (feature_key,event_key): df, n_sess x n_time
    all_shuffle_allsess = {} # mean across session, (feature_key,event_key): n_shuffle x n_time
    # both are already shifted

    to_shift_d={} # (feature_key,event_key): n_sess
    for kk in analysis_res_d_allsess[0].keys():
        all_feature_allsess[kk] = []
        all_shuffle_allsess[kk] = []
        for analysis_res_d in analysis_res_d_allsess:
            feature_mean = analysis_res_d[kk]['feature'].mean(axis=0)
            shuffle_mean = analysis_res_d[kk]['shuffle']
            all_feature_allsess[kk].append(feature_mean)
            all_shuffle_allsess[kk].append(shuffle_mean)
        
        all_shuffle_allsess[kk] = np.array(all_shuffle_allsess[kk]) # n_session x n_shuffle x n_time
        to_shift = all_shuffle_allsess[kk].mean(axis=(1,2)) - all_shuffle_allsess[kk].mean()
        to_shift_d[kk] = to_shift # n_sess
        all_shuffle_allsess[kk] = all_shuffle_allsess[kk] - to_shift[:,None,None]
        all_shuffle_allsess[kk] = all_shuffle_allsess[kk].mean(axis=0) # n_shuffle x n_time 
        all_shuffle_allsess[kk] = pd.DataFrame(all_shuffle_allsess[kk],columns=shuffle_mean.columns)
        all_feature_allsess[kk] = pd.DataFrame(all_feature_allsess[kk]) - to_shift[:,None]
        
    test_res_d={}
    ylim_d={'ach':[-0.06,0.06],'pop_fr':[0,1.],'consec_pv_dist':[0,1],'p_continuous':[0,1]}
    ylabel_d = {'ach':'ACh (dF/F)','pop_fr':'Rate (Hz)','consec_pv_dist':'Consec. PV Dist.','p_continuous':'P(Continuous)'}
    for (feature_key,event_key),one_feat_allsess in all_feature_allsess.items():
        one_shuffle_allsess = all_shuffle_allsess[(feature_key,event_key)]
        
        test_res=test_pre_post_against_shuffle(one_feat_allsess,one_shuffle_allsess,center=0,test_win=None)
        test_res_d[(feature_key,event_key)] = test_res
        fig,ax=plt.subplots(figsize=(1.5,2))
        feat_toplot = one_feat_allsess.T 
        shuffle_toplot=one_shuffle_allsess
        feat_toplot_mean = feat_toplot.mean(axis=1)
        ax.plot(feat_toplot,alpha=0.3)
        ax.plot(feat_toplot_mean,c='k',linewidth=3)
    #     ax.plot(one_shuffle_allsess.T- to_shift.values[None,:],alpha=0.3,linestyle=':')
        ph.plot_mean_error_plot(shuffle_toplot,color='grey',fig=fig,ax=ax,error_type='std')
        

        ylim = ylim_d.get(feature_key,None)
        
        ax=ph.set_two_ticks(ax,apply_to='y',do_int=False,ylim=ylim)
        ax=ph.set_symmetric_ticks(ax,apply_to='x')
        
        
        ax.set_xlabel('Time (s)')
        ylabel = ylabel_d.get(feature_key,feature_key)
        ax.set_ylabel(ylabel)
        ax.set_title(event_key.replace('_',' '))
        
        sns.despine()
        
        figfn = f'{feature_key};{event_key}_peri_event_session_agg'
        ph.save_fig(fig,figfn,prep_fig_save_dir)
    test_res_d=pd.DataFrame(test_res_d).T
    test_res_d.to_csv(os.path.join(prep_fig_save_dir,'peri_event_session_agg_test_res.csv'))

    res = {'all_feature_allsess':all_feature_allsess,'all_shuffle_allsess':all_shuffle_allsess,'to_shift_d':to_shift_d,'test_res_d':test_res_d}