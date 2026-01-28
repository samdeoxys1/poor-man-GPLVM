'''
helper function for getting special event windows
- e.g. population burst events PBE

'''
import numpy as np
import pynapple as nap

def detect_population_synchrony(spike_times, mask=None, ep=None, bin_size=0.001, smooth_std=0.0075, 
                                 z_thresh=3.0, min_duration=0.05, max_duration=0.5,
                                 ripple_intervals=None):
    '''
    Detect population synchrony events based on z-scored population firing rate.
    
    Parameters:
    -----------
    spike_times : nap.TsGroup
        Spike times for all units
    mask : array-like, optional
        Boolean mask or indices to select subset of units
    ep : nap.IntervalSet, optional
        Time epoch to restrict analysis
    bin_size : float
        Bin size for counting spikes (default 0.001s = 1ms)
    smooth_std : float
        Standard deviation for Gaussian smoothing (default 0.0075s = 7.5ms)
    z_thresh : float
        Z-score threshold for putative events (default 3.0)
    min_duration : float
        Minimum duration for valid events (default 0.05s)
    max_duration : float
        Maximum duration for valid events (default 0.5s)
    ripple_intervals : nap.IntervalSet, optional
        Ripple intervals to count within each event window
        
    Returns:
    --------
    dict with:
        - event_windows: nap.IntervalSet with all detected events (with n_ripples in metadata)
        - event_windows_with_ripple: nap.IntervalSet with events containing at least one ripple
        - population_rate: nap.Tsd of smoothed population rate (restricted to event_windows)
        - population_rate_z: nap.Tsd of z-scored population rate (restricted to event_windows)
    '''
    
    
    # Apply mask if provided
    if mask is not None:
        mask = np.asarray(mask)
        if mask.dtype == bool:
            indices = np.array(list(spike_times.keys()))[mask]
        else:
            indices = mask
        spike_times = spike_times[indices]
    
    # Count spikes in bins
    spike_counts = spike_times.count(bin_size=bin_size, ep=ep)
    
    # Sum across all units to get population rate
    population_rate = spike_counts.sum(axis=1)
    
    # Smooth the population rate
    population_rate_smooth = population_rate.smooth(std=smooth_std)
    
    # Z-score the population rate
    rate_values = population_rate_smooth.values
    mean_rate = np.mean(rate_values)
    std_rate = np.std(rate_values)
    z_values = (rate_values - mean_rate) / std_rate
    population_rate_z = nap.Tsd(t=population_rate_smooth.times(), d=z_values, 
                                 time_support=population_rate_smooth.time_support)
    
    # Find putative event windows above z_thresh
    putative_above_thresh = population_rate_z.threshold(z_thresh, method='above')
    putative_intervals = putative_above_thresh.time_support
    
    # Find all windows above mean (z > 0)
    above_mean = population_rate_z.threshold(0, method='above')
    above_mean_intervals = above_mean.time_support
    
    # For each above-mean interval, check if it contains putative events
    # This extends putative events to when rate drops to mean
    extended_starts = []
    extended_ends = []
    
    if len(putative_intervals) > 0 and len(above_mean_intervals) > 0:
        putative_starts = np.asarray(putative_intervals.start)
        putative_ends = np.asarray(putative_intervals.end)
        above_mean_starts = np.asarray(above_mean_intervals.start)
        above_mean_ends = np.asarray(above_mean_intervals.end)
        
        # Vectorized overlap check using broadcasting
        # Shape: (n_above_mean, n_putative)
        putative_start_in = (putative_starts[None, :] >= above_mean_starts[:, None]) & (putative_starts[None, :] <= above_mean_ends[:, None])
        putative_end_in = (putative_ends[None, :] >= above_mean_starts[:, None]) & (putative_ends[None, :] <= above_mean_ends[:, None])
        putative_contains = (putative_starts[None, :] <= above_mean_starts[:, None]) & (putative_ends[None, :] >= above_mean_ends[:, None])
        
        # Each above_mean interval that overlaps with any putative
        contains_putative = np.any(putative_start_in | putative_end_in | putative_contains, axis=1)
        
        # Filter by duration
        durations = above_mean_ends - above_mean_starts
        valid_duration = (durations >= min_duration) & (durations <= max_duration)
        
        # Combine conditions
        valid_mask = contains_putative & valid_duration
        extended_starts = above_mean_starts[valid_mask].tolist()
        extended_ends = above_mean_ends[valid_mask].tolist()
    
    if len(extended_starts) > 0:
        event_windows = nap.IntervalSet(start=extended_starts, end=extended_ends)
    else:
        event_windows = nap.IntervalSet(start=[], end=[])
    
    # Count ripples in each event window
    n_events = len(event_windows)
    n_ripples_per_event = np.zeros(n_events, dtype=int)
    
    if ripple_intervals is not None and n_events > 0:
        ripple_starts = np.asarray(ripple_intervals.start)
        event_starts = np.asarray(event_windows.start)
        event_ends = np.asarray(event_windows.end)
        
        for i in range(n_events):
            # Count ripples whose start falls within this event window
            n_ripples = np.sum((ripple_starts >= event_starts[i]) & (ripple_starts <= event_ends[i]))
            n_ripples_per_event[i] = n_ripples
    
    # Set metadata for event_windows
    if n_events > 0:
        event_windows.set_info({'n_ripples': n_ripples_per_event})
    
    # Filter to events with at least one ripple
    event_windows_with_ripple = None
    if ripple_intervals is not None:
        has_ripple = n_ripples_per_event > 0
        if np.any(has_ripple):
            # Get indices of events with ripples
            ripple_event_indices = np.where(has_ripple)[0]
            ripple_starts = [event_windows.start[i] for i in ripple_event_indices]
            ripple_ends = [event_windows.end[i] for i in ripple_event_indices]
            event_windows_with_ripple = nap.IntervalSet(start=ripple_starts, end=ripple_ends)
            event_windows_with_ripple.set_info({'n_ripples': n_ripples_per_event[has_ripple]})
        else:
            event_windows_with_ripple = nap.IntervalSet(start=[], end=[])
    
    print(f"Detected {n_events} population synchrony events")
    if ripple_intervals is not None:
        n_with_ripple = np.sum(n_ripples_per_event > 0)
        print(f"  {n_with_ripple} events contain at least one ripple")
    
    # Restrict population rates to event windows
    if n_events > 0:
        population_rate_restricted = population_rate_smooth.restrict(event_windows)
        population_rate_z_restricted = population_rate_z.restrict(event_windows)
    else:
        population_rate_restricted = nap.Tsd(t=[], d=[], time_support=population_rate_smooth.time_support)
        population_rate_z_restricted = nap.Tsd(t=[], d=[], time_support=population_rate_z.time_support)
    
    return {
        'event_windows': event_windows,
        'event_windows_with_ripple': event_windows_with_ripple,
        'population_rate': population_rate_restricted,
        'population_rate_z': population_rate_z_restricted,
        'mean_rate': mean_rate,
        'std_rate': std_rate,
        'n_ripples_per_event': n_ripples_per_event,
    }