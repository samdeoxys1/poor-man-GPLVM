'''
helper function for analyzing trial-by-trial data
'''
import numpy as np
import pandas as pd

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

