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