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

def find_ach_ramp_onset(ach_data,smooth_win=1,height=0.3,do_zscore=True,detrend_cutoff=None):
    '''
    detrend_cutoff: float, optional, usually 0.01
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
    ach_ramp_onset = nap.Ts(slope.t[peaks])
    ach_ramp_onset_res = {'ach_ramp_onset':ach_ramp_onset,'slope':slope,'ach_data_smth':ach_data_smth,'ach_data':ach_data,'peak_heights':peak_heights}
    return ach_ramp_onset_res

def event_triggered_analysis(feature,event_ts,n_shuffle=10,minmax=4,do_zscore=False,test_win=1,do_plot=False,fig=None,ax=None):
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
        fig,ax=ph.plot_mean_error_plot(toplot_z,mean_axis=0,ax=ax,fig=fig)
        fig,ax=ph.plot_mean_error_plot(ach_peri_shuffle,mean_axis=0,fig=fig,ax=ax,color='grey')
        ax.set_xlabel('Time (s)')
        return analysis_res,fig,ax

    return analysis_res

def event_triggered_analysis_multiple_feature_event(feature_d,event_ts_d,n_shuffle=10,minmax=4,do_zscore=False,test_win=1,do_plot=False,fig=None,ax=None):
    '''
    wrapper of event_triggered_analysis for multiple features and events
    '''
    analysis_res_d = {}
    if do_plot:
        fig_d = {}
        ax_d = {}
    for feat_name,feat in feature_d.items():
        print(f'====Feature: {feat_name}====')
        for event_name,event_ts in event_ts_d.items():
            print(f'====Event: {event_name}====')
            if do_plot:
                analysis_res,fig_,ax_=event_triggered_analysis(feat,event_ts,n_shuffle=n_shuffle,minmax=minmax,do_zscore=do_zscore,test_win=test_win,do_plot=do_plot,fig=fig,ax=ax)
            else:
                analysis_res = event_triggered_analysis(feat,event_ts,n_shuffle=n_shuffle,minmax=minmax,do_zscore=do_zscore,test_win=test_win)
            analysis_res_d[feat_name,event_name] = analysis_res
            if do_plot:
                fig_d[feat_name,event_name] = fig_
                ax_d[feat_name,event_name] = ax_
    if do_plot:
        return analysis_res_d,fig_d,ax_d
    return analysis_res_d

def prep_feature_d(prep_res,consec_pv_dist_metric='correlation',continuous_dynamics_ind=0,jump_dynamics_ind=1):
    '''
    prepare the features used for peri event analysis
    This function is updated when we care about different features
    '''
    ach = prep_res['fluo_data']['ACh']
    spike_mat_sub = prep_res['spike_mat_sub']
    pop_fr = spike_mat_sub.mean(axis=1) / np.median(np.diff(spike_mat_sub.t))
    consec_pv_dist=ah.get_consecutive_pv_distance(spike_mat_sub,metric=consec_pv_dist_metric)

    p_continuous = prep_res['posterior_dynamics_marg'][:,continuous_dynamics_ind]
    p_jump = prep_res['posterior_dynamics_marg'][:,jump_dynamics_ind]
    feature_d = {'ach':ach,'pop_fr':pop_fr,'consec_pv_dist':consec_pv_dist,'p_continuous':p_continuous,'p_jump':p_jump}
    return feature_d

def turn_sleep_state_tsd_to_interval(sleep_state_index,sleep_state_label_d={'Wake':0,'NREM':2,'REM':4}):
    '''
    turn numerically coded sleep state time series to interval for each state
    sleep_state_index: Tsd, n_time, the sleep state index
    '''
    for label,label_num in sleep_state_label_d.items():
        intv=(sleep_state_index==label_num).threshold(0.5).time_support
        sleep_state_label_d[label] = intv
    return sleep_state_label_d

def segregate_event_ts_by_sleep_state(event_ts,sleep_state_label_d):
    '''
    segregate the event timestamps by sleep state
    event_ts: Ts, the event timestamps
    sleep_state_label_d: dict, key is the sleep state label, value is the interval
    return: dict, key is the sleep state label, value is the event timestamps
    '''
    event_ts_d = {}
    for label,intv in sleep_state_label_d.items():
        event_ts_d[label] = event_ts.restrict(intv)
    return event_ts_d

# def prep_event_ts_d(prep_res):
