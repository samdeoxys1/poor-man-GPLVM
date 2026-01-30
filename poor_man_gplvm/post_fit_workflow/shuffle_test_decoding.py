'''
Shuffle test for whether the decoding of some event is "significant"
Currently using neuron-id shuffle and circular shuffle
Support both the supervised and unsupervised decoding (both use tuning, only difference is where the tuning is from), 
Currently do the test on naive bayes marginal likelihood, since the state-space has a "built-in" test for dynamics
'''

def create_neuron_id_shuffle_one_event():
    '''
    given spk_mat (n_time, n_neuron), create neuron-id shuffled spk_mat (n_shuffle, n_time, n_neuron); each spike count is assigned a random neuron-id
    '''
    pass

def create_circular_shuffle_one_event():
    '''
    given spk_mat (n_time, n_neuron), create circular shuffled spk_mat (n_shuffle, n_time, n_neuron): each neuron's spike count is circularly shuffled independently
    '''
    pass

def create_neuron_id_shuffle_all_events_concat():
    '''
    given spk_mat (n_time,n_neuron), and event_index_per_bin (n_time,) (from trial_analysis.bin_spike_train_to_trial_based), 
    create neuron-id shuffled spk_mat (n_shuffle, n_time, n_neuron); each spike count is assigned a random neuron-id
    '''

    pass

def create_circular_shuffle_all_events_concat():
    '''
    given spk_mat (n_time,n_neuron), and event_index_per_bin (n_time,) (from trial_analysis.bin_spike_train_to_trial_based), 
    create circular shuffled spk_mat (n_shuffle, n_time, n_neuron): each neuron's spike count is circularly shuffled independently
    '''
    pass

def shuffle_test_naive_bayes_marginal_l():
    '''
    create neuron-id and circular shuffle; loop over to decode and get the log marginal likelihood per event; get quantiles
    also get the quantiles (0-1, 0.025 binsize) of the time-averaged log likelihood, log posterior, posterior, log marginal likelihood; 
    
    '''
    pass

