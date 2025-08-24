'''
helper function for analyzing trial-by-trial data
'''
import numpy as np
import pandas as pd
import seaborn as sns

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