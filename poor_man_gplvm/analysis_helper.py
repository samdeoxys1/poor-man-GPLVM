'''
helper functions for analysis after fitting the model
'''

def get_state_interval(p_l,p_thresh=0.8, merge_thresh=1,duration_thresh=2,):
    '''
    get the state interval from the posterior, by thresholding, get intervals, merge, filter for duration

    e.g. can use this to get the chunks of continuous fragmented states

    p_l: nap.Tsd, probability of some state (usually a dynamics like continuous / jump)
    p_thresh: threshold for considering dynamics to be continuous
    merge_thresh: threshold for merging adjacent intervals
    duration_thresh: threshold for filtering out short intervals

    return: 
    continuous_intv_merge: nap.IntervalSet, the interval of the chunks
    '''
    p_continuous_thresh = p_thresh # threshold for considering dynamics to be continuous
    continuous_intv=p_l.threshold(p_continuous_thresh).time_support
    continuous_intv_merge = continuous_intv.merge_close_intervals(merge_thresh) # threshold for merging adjacent intervals
    ma=continuous_intv_merge[:,1]-continuous_intv_merge[:,0] > duration_thresh
    continuous_intv_merge = continuous_intv_merge[ma]
    return continuous_intv_merge



