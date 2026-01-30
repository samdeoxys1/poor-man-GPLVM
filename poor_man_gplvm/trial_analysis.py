'''
helper function for analyzing trial-by-trial data
'''
import numpy as np
import pandas as pd
import seaborn as sns

import pynapple as nap

def get_event_phase_in_trial(trials,event_key_l=[],span_key=['start','end']):
    '''
    get the phase of the event in the trial
    span_key define the start and end of the epoch we look at, usually 'start' and 'end' of the trial
    '''
    event_phase_per_trial_per_event = {}
    for event_key in event_key_l:
        event_phase_per_trial_per_event[event_key] = (trials[event_key] - trials[span_key[0]]) / (trials[span_key[1]] - trials[span_key[0]])
    event_phase_per_trial_per_event = pd.DataFrame(event_phase_per_trial_per_event)
    return event_phase_per_trial_per_event

def plot_event_range(event_phase_df,xs=None,event_key_l=None,ax=None,quantile_range=[0.25,0.75],alpha=0.5,palette='Set1'):
    '''
    event_phase_df: n_trials x n_events
    plot shaded region for the event range, with the quantile range
    '''
    if event_key_l is None:
        event_key_l = event_phase_df.columns.tolist()
    if xs is None: # if not provided, assume plot has been normalized to [0,1]
        xs=(0,1)
    palette = sns.color_palette(palette, len(event_key_l))
    for i,event_key in enumerate(event_key_l):
        left_,right_ = event_phase_df[event_key].quantile(quantile_range[0]),event_phase_df[event_key].quantile(quantile_range[1])
        left=(xs[-1]-xs[0]) * left_ + xs[0]
        right=(xs[-1]-xs[0]) * right_ + xs[0]
        ax.axvspan(left,right,color=palette[i],alpha=alpha)
    return ax

# turn continuous-time data into trial / event-based data
def bin_spike_train_to_trial_based(spike_train,trial_intervals,binsize=0.02):
    '''
    spike_train: nap.TsGroup
    trial_intervals: nap.IntervalSet; can be any event, e.g. population burst events

    return:
    spike_mat: concatenated binned counts (nap.TsdFrame-like), from spike_train.count
    spike_mat_padded: n_trial x n_bin x n_neuron; padded with 0 
    mask: n_trial x n_bin x 1; True for valid bins
    event_index_per_bin: n_bin; index of the event for each bin
    n_bin_per_trial: n_trial; number of bins per trial
    time_l: (n_bin,) timestamps for concatenated binned data (same order as event_index_per_bin)
    time_per_trial: list[np.ndarray], len n_trial; timestamps per trial (inhomogeneous)
    '''

    # concatenated spike matrix
    spike_mat = spike_train.count(binsize,ep=trial_intervals)
    time_l = np.asarray(spike_mat.t)

    # event index for each bin
    event_index_per_bin = np.asarray(trial_intervals.in_interval(nap.Ts(time_l)))
    n_trial = int(trial_intervals.shape[0])
    time_per_trial = [time_l[event_index_per_bin == i] for i in range(n_trial)]

    # padded tensor
    spike_mat_padded = nap.build_tensor(spike_mat,ep=trial_intervals,bin_size=binsize) # n_neuron x n_trial x n_bin
    spike_mat_padded = spike_mat_padded.swapaxes(0,1).swapaxes(1,2) # n_trial x n_bin x n_neuron

    # mask
    mask_full = np.logical_not(np.isnan(spike_mat_padded))
    mask = mask_full[:,:,[0]] # n_trial x n_bin x 1

    # n_bin of each trial
    n_bin_per_trial = np.squeeze(mask_full.sum(axis=1))

    # repad with 0 to avoid nan
    spike_mat_padded[np.logical_not(mask_full)] = 0

    bin_spk_res = {
        'spike_mat': spike_mat,
        'spike_mat_padded': spike_mat_padded,
        'mask': mask,
        'event_index_per_bin': event_index_per_bin,
        'n_bin_per_trial': n_bin_per_trial,
        'time_l': time_l,
        'time_per_trial': time_per_trial,
    }
    return bin_spk_res
