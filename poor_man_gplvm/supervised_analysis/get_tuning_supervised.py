'''
function for obtaining tuning curves from labels 

input:
    label_l: nap.TsdFrame, n_time x n_dimension or {maze_key : nap.TsdFrame n_time x n_dimension} for multiple mazes
    spk_mat: nap.TsdFrame, n_time x n_neuron (for multiple maze can use the interval sets from label_l to restrict the spike times)
    ep = None: nap.IntervalSet or {maze_key : nap.IntervalSet} for multiple mazes, to restrict the label and spike times (e.g. for speed threshold)
    custom_smooth_func: function or {maze_key : function} for multiple mazes, evaluate on grid centers to return a smoothing matrix
    label_bin_size: float (same for all label dimensions), array (different for each dimension), or {maze_key : float or array} for multiple mazes, the bin size for the label grid
    smooth_std: float (same for all label dimensions), array (different for each dimension), or {maze_key : float or array} for multiple mazes, the standard deviation for the Gaussian kernel for smoothing; if any ==0/None, then no smoothing; in label unit
    occupancy_threshold: threshold in seconds (need to convert to bin units) for a label bin to be considered occupied; unoccupied bins will be masked with nan; if None, then all bins are considered occupied
    label_min: float, array (per dim), or {maze_key : float or array}; lower edge of first bin; values below go to first bin; None => infer from data
    label_max: float, array (per dim), or {maze_key : float or array}; upper limit for binning; values above go to last bin; None => infer from data
        useful for coarse binning, e.g. speed bins [0,5), [5,10), [10,15] with label_min=0, label_max=10, label_bin_size=5
    categorical_labels: None (auto-detect integer-valued columns like 0.0,1.0,2.0), list of column names, [] (none), or True (all)
        categorical columns use nearest-neighbor alignment instead of interpolation


Returns tuning_res dict with:
    tuning: xr.DataArray (*grid_shape, n_neuron) or {maze_key: xr.DataArray} for multi-maze
        firing rate; unoccupied bins are nan; computed as smoothed spk_count / smoothed occupancy
    tuning_flat: (n_valid_bin, n_neuron); for multi-maze, concatenated with coord_to_flat_idx mapping
    coord_to_flat_idx: pd.Series, multi-index (label coords + maze_key for multi) -> index in tuning_flat
    occupancy, occupancy_smth: xr.DataArray or {maze_key: xr.DataArray}
    occupancy_flat, spk_count_flat, etc.: flat arrays (valid bins only); for multi-maze, concatenated
    bin_edges, bin_centers, grid_shape, label_dim_names, neuron_names, dt, etc.
'''

import numpy as np
import pynapple as nap
import xarray as xr
import pandas as pd

# ============================================================================
# Helper: normalize inputs for single vs multi-maze
# ============================================================================

def _to_dict(val, maze_keys, name=''):
    '''convert single value to dict keyed by maze_keys if not already a dict'''
    if isinstance(val, dict):
        return val
    return {k: val for k in maze_keys}


def _is_integer_valued(arr):
    '''check if array values are all integers (zero decimal part)'''
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return False
    return np.allclose(finite, np.floor(finite))


def _nearest_align(source_tsd, target_times):
    '''
    Align source_tsd to target_times using nearest-neighbor (not interpolation).
    Returns values at nearest source timestamp for each target time.
    '''
    source_t = source_tsd.t
    source_d = source_tsd.d
    
    # find nearest source index for each target time
    idx = np.searchsorted(source_t, target_times)
    # compare with idx and idx-1 to find truly nearest
    idx = np.clip(idx, 1, len(source_t) - 1)
    left_dist = np.abs(target_times - source_t[idx - 1])
    right_dist = np.abs(target_times - source_t[idx])
    nearest_idx = np.where(left_dist <= right_dist, idx - 1, idx)
    
    if source_d.ndim == 1:
        return source_d[nearest_idx]
    else:
        return source_d[nearest_idx, :]


def _normalize_inputs(label_l, spk_mat, ep=None, custom_smooth_func=None, 
                      label_bin_size=1., smooth_std=None, label_min=None, label_max=None,
                      categorical_labels=None):
    '''
    Normalize single vs multi-maze inputs.
    Returns dict-form for all inputs keyed by maze_key.
    Also aligns labels to spk_mat timestamps.
    
    categorical_labels: None (auto-detect integer-valued columns), list of column names,
                        [] (none categorical), or True (all categorical).
                        Categorical columns use nearest-neighbor alignment instead of interpolation.
    '''
    # determine if multi-maze
    if isinstance(label_l, dict):
        maze_keys = list(label_l.keys())
        is_multi = True
    else:
        maze_keys = ['_single']
        label_l = {'_single': label_l}
        is_multi = False

    ep_d = _to_dict(ep, maze_keys, 'ep')
    custom_smooth_func_d = _to_dict(custom_smooth_func, maze_keys, 'custom_smooth_func')
    label_bin_size_d = _to_dict(label_bin_size, maze_keys, 'label_bin_size')
    smooth_std_d = _to_dict(smooth_std, maze_keys, 'smooth_std')
    label_min_d = _to_dict(label_min, maze_keys, 'label_min')
    label_max_d = _to_dict(label_max, maze_keys, 'label_max')

    # align and restrict
    label_aligned_d = {}
    spk_aligned_d = {}
    for k in maze_keys:
        label_k = label_l[k]
        ep_k = ep_d.get(k, None)
        
        # get column names
        if hasattr(label_k, 'columns'):
            col_names = list(label_k.columns)
        else:
            col_names = list(range(label_k.d.shape[1]))
        
        # determine categorical columns
        if categorical_labels is True:
            cat_cols = col_names
        elif categorical_labels is None:
            # auto-detect: columns where all values are integer-valued
            cat_cols = []
            for c in col_names:
                col_data = label_k[c].d if hasattr(label_k[c], 'd') else label_k.d[:, col_names.index(c)]
                if _is_integer_valued(col_data):
                    cat_cols.append(c)
        else:
            # explicit list (can be empty)
            cat_cols = list(categorical_labels)
        
        cont_cols = [c for c in col_names if c not in cat_cols]
        
        # restrict label to epoch first (before alignment)
        if ep_k is not None:
            label_k_restricted = label_k.restrict(ep_k)
            spk_sub = spk_mat.restrict(ep_k)
        else:
            label_k_restricted = label_k
            spk_sub = spk_mat
        
        # get common time support
        common_ts = label_k_restricted.time_support.intersect(spk_sub.time_support)
        label_k_restricted = label_k_restricted.restrict(common_ts)
        spk_sub = spk_sub.restrict(common_ts)
        
        # target times for alignment
        target_times = spk_sub.t
        
        # align columns
        aligned_data = np.zeros((len(target_times), len(col_names)))
        for i, c in enumerate(col_names):
            col_tsd = label_k_restricted[c] if hasattr(label_k_restricted, '__getitem__') else None
            if col_tsd is None:
                col_data = label_k_restricted.d[:, i]
                col_tsd = nap.Tsd(t=label_k_restricted.t, d=col_data)
            
            if c in cat_cols:
                # nearest-neighbor alignment
                aligned_data[:, i] = _nearest_align(col_tsd, target_times)
            else:
                # interpolation
                aligned_col = col_tsd.interpolate(spk_sub)
                aligned_data[:, i] = aligned_col.d
        
        # build aligned TsdFrame
        label_aligned = nap.TsdFrame(t=target_times, d=aligned_data, columns=col_names)
        
        # spk is already at target times
        spk_aligned = spk_sub

        label_aligned_d[k] = label_aligned
        spk_aligned_d[k] = spk_aligned

    return {
        'label_d': label_aligned_d,
        'spk_d': spk_aligned_d,
        'ep_d': ep_d,
        'custom_smooth_func_d': custom_smooth_func_d,
        'label_bin_size_d': label_bin_size_d,
        'smooth_std_d': smooth_std_d,
        'label_min_d': label_min_d,
        'label_max_d': label_max_d,
        'maze_keys': maze_keys,
        'is_multi': is_multi,
    }


# ============================================================================
# label_to_grid: build bin edges and centers
# ============================================================================

def label_to_grid(label, label_bin_size, label_min=None, label_max=None):
    '''
    Build bin edges and centers from label data.
    
    label: nap.TsdFrame (n_time x n_dim)
    label_bin_size: float or array (per dim)
    label_min: float or array (per dim), or None => infer from data - 0.5*bin_size
        lower edge of first bin; values below go to first bin
    label_max: float or array (per dim), or None => infer from data + 0.5*bin_size
        upper limit for binning; values above go to last bin
        (the 0.5*bin_size extension makes bin centers align to natural values like 0,10,20...)
    
    Returns:
        bin_edges_l: list of arrays, bin edges for each dim
        bin_centers_l: list of arrays, bin centers for each dim
        grid_shape: tuple, shape of the grid (n_bins_dim0, n_bins_dim1, ...)
        label_dim_names: list of str, column names from label
    '''
    label_arr = label.d  # n_time x n_dim
    n_dim = label_arr.shape[1]
    
    # get dim names
    if hasattr(label, 'columns'):
        label_dim_names = list(label.columns)
    else:
        label_dim_names = [f'dim{i}' for i in range(n_dim)]
    
    # broadcast bin_size, label_min, label_max
    # convert to list to preserve None values per dimension
    label_bin_size = np.atleast_1d(label_bin_size).astype(float)
    if label_bin_size.size == 1:
        label_bin_size = np.repeat(label_bin_size, n_dim)
    
    # handle label_min/max as lists to preserve per-dim None
    if label_min is not None:
        label_min = list(np.atleast_1d(label_min))
        if len(label_min) == 1:
            label_min = label_min * n_dim
    else:
        label_min = [None] * n_dim
    
    if label_max is not None:
        label_max = list(np.atleast_1d(label_max))
        if len(label_max) == 1:
            label_max = label_max * n_dim
    else:
        label_max = [None] * n_dim
    
    bin_edges_l = []
    bin_centers_l = []
    for d in range(n_dim):
        col = label_arr[:, d]
        finite_ma = np.isfinite(col)
        col_finite = col[finite_ma]
        bs = label_bin_size[d]
        
        # determine lo (first bin edge) - check per-dim None
        # extend by 0.5*bs when inferred from data so bin centers align to natural values
        lmin_d = label_min[d]
        if lmin_d is not None and np.isfinite(float(lmin_d)):
            lo = float(lmin_d)
        elif col_finite.size == 0:
            lo = -0.5 * bs
        else:
            lo = col_finite.min() - 0.5 * bs
        
        # determine hi - check per-dim None
        # extend by 0.5*bs when inferred from data
        lmax_d = label_max[d]
        if lmax_d is not None and np.isfinite(float(lmax_d)):
            hi = float(lmax_d)
        elif col_finite.size == 0:
            hi = 0.5 * bs
        else:
            hi = col_finite.max() + 0.5 * bs
        
        # build edges from lo to hi (values below lo go to first bin, above hi to last bin)
        edges = np.arange(lo, hi + 1e-9, bs)
        centers = 0.5 * (edges[:-1] + edges[1:])
        
        bin_edges_l.append(edges)
        bin_centers_l.append(centers)
    
    grid_shape = tuple(len(c) for c in bin_centers_l)
    
    return bin_edges_l, bin_centers_l, grid_shape, label_dim_names


def _digitize_labels(label_arr, bin_edges_l):
    '''
    Convert label array to per-dim bin indices.
    Returns bin_inds (n_time, n_dim), valid_mask (n_time,).
    Bin indices are 0-based, clipped to valid range.
    
    Values below min edge go to first bin (index 0).
    Values >= max edge go to last bin (since last edge is inf).
    '''
    n_time, n_dim = label_arr.shape
    bin_inds = np.zeros((n_time, n_dim), dtype=int)
    valid_mask = np.ones(n_time, dtype=bool)
    
    for d in range(n_dim):
        col = label_arr[:, d]
        edges = bin_edges_l[d]
        n_bins = len(edges) - 1
        
        # digitize: returns 1..n_bins for values in range, 0 for below first edge
        # since last edge is inf, values >= second-to-last edge go to last bin
        inds = np.digitize(col, edges) - 1  # now 0-based
        
        # mark invalid: only NaN
        invalid = ~np.isfinite(col)
        valid_mask &= ~invalid
        
        # clip: values below min go to 0, values above (shouldn't happen with inf edge) clip to n_bins-1
        bin_inds[:, d] = np.clip(inds, 0, n_bins - 1)
    
    return bin_inds, valid_mask


def _flat_index(bin_inds, grid_shape):
    '''
    Convert multi-dim bin indices to flat index (C-order).
    bin_inds: (n_time, n_dim)
    grid_shape: tuple
    Returns flat_inds: (n_time,)
    '''
    return np.ravel_multi_index(bin_inds.T, grid_shape, order='C')


# ============================================================================
# get_count_vs_grid: accumulate spike counts
# ============================================================================

def get_count_vs_grid(spk_arr, bin_inds, valid_mask, grid_shape):
    '''
    Accumulate spike counts per label bin.
    
    spk_arr: (n_time, n_neuron)
    bin_inds: (n_time, n_dim) from _digitize_labels
    valid_mask: (n_time,) bool
    grid_shape: tuple
    
    Returns:
        spk_count_flat: (n_flat_bins, n_neuron)
        spk_count_grid: (*grid_shape, n_neuron)
    '''
    n_flat = int(np.prod(grid_shape))
    n_neuron = spk_arr.shape[1]
    
    spk_count_flat = np.zeros((n_flat, n_neuron), dtype=float)
    
    # only valid timepoints
    flat_inds = _flat_index(bin_inds[valid_mask], grid_shape)
    spk_valid = spk_arr[valid_mask]
    
    # accumulate
    np.add.at(spk_count_flat, flat_inds, spk_valid)
    
    spk_count_grid = spk_count_flat.reshape((*grid_shape, n_neuron))
    
    return spk_count_flat, spk_count_grid


# ============================================================================
# get_occupancy_vs_grid: compute occupancy in seconds
# ============================================================================

def get_occupancy_vs_grid(spk_mat, bin_inds, valid_mask, grid_shape, dt=None):
    '''
    Compute occupancy per label bin in seconds.
    
    spk_mat: nap.TsdFrame (for timestamps) or array
    bin_inds: (n_time, n_dim)
    valid_mask: (n_time,)
    grid_shape: tuple
    dt: time bin size in seconds; if None, infer from spk_mat.t
    
    Returns:
        occupancy_flat: (n_flat_bins,) in seconds
        occupancy_grid: (*grid_shape,) in seconds
    '''
    # infer dt
    if dt is None:
        if hasattr(spk_mat, 't'):
            dt = np.median(np.diff(spk_mat.t))
        else:
            dt = 1.0  # fallback
    
    n_flat = int(np.prod(grid_shape))
    flat_inds = _flat_index(bin_inds[valid_mask], grid_shape)
    
    # occupancy = count * dt
    occupancy_flat = np.bincount(flat_inds, minlength=n_flat).astype(float) * dt
    occupancy_grid = occupancy_flat.reshape(grid_shape)
    
    return occupancy_flat, occupancy_grid, dt


# ============================================================================
# get_smoothing_matrix: Gaussian or custom
# ============================================================================

def _gaussian_kernel_1d(centers, std):
    '''
    Build row-normalized Gaussian smoothing matrix for 1D bin centers.
    centers: (n_bins,)
    std: float, in same unit as centers
    Returns: (n_bins, n_bins) row-normalized
    '''
    c = np.asarray(centers)
    diff = c[:, None] - c[None, :]  # (n, n)
    K = np.exp(-0.5 * (diff / std) ** 2)
    # row normalize
    row_sums = K.sum(axis=1, keepdims=True)
    row_sums = np.where(row_sums > 0, row_sums, 1.)
    K = K / row_sums
    return K


def get_smoothing_matrix(bin_centers_l, grid_shape, smooth_std=None, custom_smooth_func=None,
                         label_grid_centers=None):
    '''
    Build smoothing matrix S of shape (n_flat_bins, n_flat_bins).
    
    If custom_smooth_func is provided: S = custom_smooth_func(label_grid_centers)
    Else: separable Gaussian kernels combined via Kronecker product.
    
    bin_centers_l: list of 1D arrays, bin centers per dim
    grid_shape: tuple
    smooth_std: float, array (per dim), or None (no smoothing)
    custom_smooth_func: callable(label_grid_centers) -> S
    label_grid_centers: meshgrid of centers (for custom_smooth_func)
    
    Returns: S (n_flat, n_flat)
    '''
    n_flat = int(np.prod(grid_shape))
    n_dim = len(bin_centers_l)
    
    if custom_smooth_func is not None:
        # build label_grid_centers if not provided
        if label_grid_centers is None:
            label_grid_centers = np.meshgrid(*bin_centers_l, indexing='ij')
        S = custom_smooth_func(label_grid_centers)
        if S.shape != (n_flat, n_flat):
            raise ValueError(f"custom_smooth_func returned shape {S.shape}, expected ({n_flat}, {n_flat})")
        return S
    
    # no smoothing
    if smooth_std is None:
        return np.eye(n_flat)
    
    smooth_std = np.atleast_1d(smooth_std)
    if smooth_std.size == 1:
        smooth_std = np.repeat(smooth_std, n_dim)
    
    # check if all zero => identity
    if np.all(smooth_std == 0) or np.all(smooth_std == None):
        return np.eye(n_flat)
    
    # build per-dim kernels and Kronecker product
    # Kronecker order must match C-order flatten: dim0 varies slowest
    K_list = []
    for d in range(n_dim):
        std_d = smooth_std[d]
        if std_d is None or std_d == 0:
            K_d = np.eye(len(bin_centers_l[d]))
        else:
            K_d = _gaussian_kernel_1d(bin_centers_l[d], std_d)
        K_list.append(K_d)
    
    # Kronecker product: K_0 ⊗ K_1 ⊗ ... (order matches C-order ravel)
    S = K_list[0]
    for K_d in K_list[1:]:
        S = np.kron(S, K_d)
    
    return S


# ============================================================================
# get_tuning: main function
# ============================================================================

def get_tuning(label_l, spk_mat, ep=None, custom_smooth_func=None,
               label_bin_size=1., smooth_std=None, occupancy_threshold=None,
               label_min=None, label_max=None, categorical_labels=None):
    '''
    Compute tuning curves from labels and spike matrix.
    
    Parameters
    ----------
    label_l : nap.TsdFrame or dict of nap.TsdFrame
        Labels, n_time x n_dimension. Dict keys are maze_key for multi-maze.
    spk_mat : nap.TsdFrame
        Spike counts, n_time x n_neuron.
    ep : nap.IntervalSet or dict, optional
        Epoch to restrict data.
    custom_smooth_func : callable or dict, optional
        Custom smoothing function(label_grid_centers) -> S matrix.
    label_bin_size : float, array, or dict
        Bin size for label grid.
    smooth_std : float, array, or dict, optional
        Gaussian smoothing std in label units. 0/None => no smoothing.
    occupancy_threshold : float, optional
        Threshold in seconds; bins with raw occupancy below this are NaN.
    label_min : float, array, or dict, optional
        Lower edge of first bin; values below go to first bin. None => infer from data.
    label_max : float, array, or dict, optional
        Upper limit for binning; values above go to last bin. None => infer from data.
        Useful for coarse binning, e.g. speed [0,5), [5,10), [10,15] with min=0, max=10, bin_size=5.
    categorical_labels : None, list, or True, optional
        None: auto-detect integer-valued columns (0.0, 1.0, 2.0...) → nearest-neighbor alignment.
        []: explicitly no categorical columns (use interpolation for all).
        ['col1', 'col2']: specific columns are categorical.
        True: all columns are categorical.
    
    Returns
    -------
    tuning_res : dict
        tuning: xr.DataArray or {maze_key: xr.DataArray}
        tuning_flat: (n_valid_bin, n_neuron), concatenated for multi-maze
        coord_to_flat_idx: pd.Series mapping (label coords [+ maze_key]) -> flat index
        occupancy, occupancy_smth, spk_count, spk_count_smth, bin_edges, bin_centers, etc.
        For multi-maze: xr outputs are dicts keyed by maze_key; flat arrays are concatenated.
    '''
    # normalize inputs
    norm = _normalize_inputs(label_l, spk_mat, ep=ep, 
                             custom_smooth_func=custom_smooth_func,
                             label_bin_size=label_bin_size, 
                             smooth_std=smooth_std,
                             label_min=label_min, label_max=label_max,
                             categorical_labels=categorical_labels)
    
    label_d = norm['label_d']
    spk_d = norm['spk_d']
    custom_smooth_func_d = norm['custom_smooth_func_d']
    label_bin_size_d = norm['label_bin_size_d']
    smooth_std_d = norm['smooth_std_d']
    label_min_d = norm['label_min_d']
    label_max_d = norm['label_max_d']
    maze_keys = norm['maze_keys']
    is_multi = norm['is_multi']
    
    # collect per-maze results
    per_maze = {}
    
    for k in maze_keys:
        label_k = label_d[k]
        spk_k = spk_d[k]
        csf_k = custom_smooth_func_d.get(k, None)
        lbs_k = label_bin_size_d.get(k, 1.)
        ss_k = smooth_std_d.get(k, None)
        lmin_k = label_min_d.get(k, None)
        lmax_k = label_max_d.get(k, None)
        
        # build grid
        bin_edges_l, bin_centers_l, grid_shape, label_dim_names = label_to_grid(
            label_k, lbs_k, label_min=lmin_k, label_max=lmax_k)
        n_flat = int(np.prod(grid_shape))
        
        # get neuron names
        if hasattr(spk_k, 'columns'):
            neuron_names = list(spk_k.columns)
        else:
            neuron_names = list(range(spk_k.d.shape[1]))
        n_neuron = len(neuron_names)
        
        # digitize labels
        label_arr = label_k.d
        bin_inds, valid_mask = _digitize_labels(label_arr, bin_edges_l)
        
        # occupancy
        occupancy_flat, occupancy_grid, dt = get_occupancy_vs_grid(
            spk_k, bin_inds, valid_mask, grid_shape, dt=None)
        
        # spike count
        spk_arr = spk_k.d
        spk_count_flat, spk_count_grid = get_count_vs_grid(
            spk_arr, bin_inds, valid_mask, grid_shape)
        
        # smoothing matrix
        label_grid_centers = np.meshgrid(*bin_centers_l, indexing='ij')
        S = get_smoothing_matrix(bin_centers_l, grid_shape, smooth_std=ss_k,
                                 custom_smooth_func=csf_k,
                                 label_grid_centers=label_grid_centers)
        
        # smooth
        occupancy_smth_flat = S @ occupancy_flat
        spk_count_smth_flat = S @ spk_count_flat
        
        # tuning = smoothed spk_count / smoothed occupancy
        with np.errstate(divide='ignore', invalid='ignore'):
            tuning_flat_full = spk_count_smth_flat / occupancy_smth_flat[:, None]
        tuning_flat_full = np.where(np.isfinite(tuning_flat_full), tuning_flat_full, np.nan)
        
        # occupancy threshold mask (on raw occupancy)
        if occupancy_threshold is not None:
            occupied_mask = occupancy_flat >= occupancy_threshold
        else:
            occupied_mask = np.ones(n_flat, dtype=bool)
        
        # apply mask
        tuning_flat_full[~occupied_mask, :] = np.nan
        
        # reshape to grid
        occupancy_smth_grid = occupancy_smth_flat.reshape(grid_shape)
        spk_count_smth_grid = spk_count_smth_flat.reshape((*grid_shape, n_neuron))
        tuning_grid = tuning_flat_full.reshape((*grid_shape, n_neuron))
        
        # build xr.DataArray for tuning
        dims = label_dim_names + ['neuron']
        coords = {dn: bin_centers_l[i] for i, dn in enumerate(label_dim_names)}
        coords['neuron'] = neuron_names
        tuning_xr = xr.DataArray(tuning_grid, dims=dims, coords=coords)
        
        # also build xr for occupancy (no neuron dim)
        occupancy_xr = xr.DataArray(occupancy_grid, dims=label_dim_names,
                                    coords={dn: bin_centers_l[i] for i, dn in enumerate(label_dim_names)})
        occupancy_smth_xr = xr.DataArray(occupancy_smth_grid, dims=label_dim_names,
                                         coords={dn: bin_centers_l[i] for i, dn in enumerate(label_dim_names)})
        
        # keep only valid (non-NaN) bins in flat arrays
        valid_flat_mask = occupied_mask & np.all(np.isfinite(tuning_flat_full), axis=1)
        valid_flat_inds = np.where(valid_flat_mask)[0]
        tuning_flat = tuning_flat_full[valid_flat_mask]
        occupancy_flat_valid = occupancy_flat[valid_flat_mask]
        occupancy_smth_flat_valid = occupancy_smth_flat[valid_flat_mask]
        spk_count_flat_valid = spk_count_flat[valid_flat_mask]
        spk_count_smth_flat_valid = spk_count_smth_flat[valid_flat_mask]
        
        # build coord_to_flat_idx: multi-index from label bin centers -> local flat index
        # unravel valid_flat_inds to get grid coords
        grid_coords = np.unravel_index(valid_flat_inds, grid_shape)
        coord_tuples = []
        for i in range(len(valid_flat_inds)):
            coord = tuple(bin_centers_l[d][grid_coords[d][i]] for d in range(len(label_dim_names)))
            coord_tuples.append(coord)
        
        per_maze[k] = {
            'tuning': tuning_xr,
            'occupancy': occupancy_xr,
            'occupancy_smth': occupancy_smth_xr,
            'spk_count': spk_count_grid,
            'spk_count_smth': spk_count_smth_grid,
            'bin_edges': bin_edges_l,
            'bin_centers': bin_centers_l,
            'label_grid_centers': label_grid_centers,
            'grid_shape': grid_shape,
            'label_dim_names': label_dim_names,
            'neuron_names': neuron_names,
            'occupied_mask': occupied_mask,
            'tuning_flat': tuning_flat,
            'occupancy_flat': occupancy_flat_valid,
            'occupancy_smth_flat': occupancy_smth_flat_valid,
            'spk_count_flat': spk_count_flat_valid,
            'spk_count_smth_flat': spk_count_smth_flat_valid,
            'valid_flat_mask': valid_flat_mask,
            'coord_tuples': coord_tuples,
            'dt': dt,
            'smoothing_matrix': S,
            'n_valid_timepoints': valid_mask.sum(),
        }
        
        print(f"[get_tuning] maze={k}: grid_shape={grid_shape}, n_neurons={n_neuron}, "
              f"dt={dt:.4f}s, n_valid_time={valid_mask.sum()}, n_occupied_bins={occupied_mask.sum()}")
    
    # build output
    if not is_multi:
        # single maze: flatten structure, build coord_to_flat_idx
        res = per_maze['_single']
        label_dim_names = res['label_dim_names']
        coord_tuples = res['coord_tuples']
        if coord_tuples:
            midx = pd.MultiIndex.from_tuples(coord_tuples, names=label_dim_names)
            coord_to_flat_idx = pd.Series(np.arange(len(coord_tuples)), index=midx)
        else:
            coord_to_flat_idx = pd.Series(dtype=int)
        
        tuning_res = {
            'tuning': res['tuning'],
            'tuning_flat': res['tuning_flat'],
            'coord_to_flat_idx': coord_to_flat_idx,
            'occupancy': res['occupancy'],
            'occupancy_smth': res['occupancy_smth'],
            'spk_count': res['spk_count'],
            'spk_count_smth': res['spk_count_smth'],
            'occupancy_flat': res['occupancy_flat'],
            'occupancy_smth_flat': res['occupancy_smth_flat'],
            'spk_count_flat': res['spk_count_flat'],
            'spk_count_smth_flat': res['spk_count_smth_flat'],
            'bin_edges': res['bin_edges'],
            'bin_centers': res['bin_centers'],
            'label_grid_centers': res['label_grid_centers'],
            'grid_shape': res['grid_shape'],
            'label_dim_names': res['label_dim_names'],
            'neuron_names': res['neuron_names'],
            'occupied_mask': res['occupied_mask'],
            'valid_flat_mask': res['valid_flat_mask'],
            'dt': res['dt'],
            'smoothing_matrix': res['smoothing_matrix'],
            'n_valid_timepoints': res['n_valid_timepoints'],
        }
        return tuning_res
    
    # multi-maze: maze_key at lowest level, concatenate flat arrays
    # xr outputs as dicts keyed by maze_key
    tuning_res = {
        'tuning': {k: per_maze[k]['tuning'] for k in maze_keys},
        'occupancy': {k: per_maze[k]['occupancy'] for k in maze_keys},
        'occupancy_smth': {k: per_maze[k]['occupancy_smth'] for k in maze_keys},
        'spk_count': {k: per_maze[k]['spk_count'] for k in maze_keys},
        'spk_count_smth': {k: per_maze[k]['spk_count_smth'] for k in maze_keys},
        'bin_edges': {k: per_maze[k]['bin_edges'] for k in maze_keys},
        'bin_centers': {k: per_maze[k]['bin_centers'] for k in maze_keys},
        'label_grid_centers': {k: per_maze[k]['label_grid_centers'] for k in maze_keys},
        'grid_shape': {k: per_maze[k]['grid_shape'] for k in maze_keys},
        'label_dim_names': {k: per_maze[k]['label_dim_names'] for k in maze_keys},
        'neuron_names': per_maze[maze_keys[0]]['neuron_names'],  # same across mazes
        'occupied_mask': {k: per_maze[k]['occupied_mask'] for k in maze_keys},
        'valid_flat_mask': {k: per_maze[k]['valid_flat_mask'] for k in maze_keys},
        'dt': {k: per_maze[k]['dt'] for k in maze_keys},
        'smoothing_matrix': {k: per_maze[k]['smoothing_matrix'] for k in maze_keys},
        'n_valid_timepoints': {k: per_maze[k]['n_valid_timepoints'] for k in maze_keys},
    }
    
    # concatenate flat arrays across mazes and build coord_to_flat_idx with maze_key
    tuning_flat_l = []
    occupancy_flat_l = []
    occupancy_smth_flat_l = []
    spk_count_flat_l = []
    spk_count_smth_flat_l = []
    coord_tuples_all = []
    
    for k in maze_keys:
        res_k = per_maze[k]
        tuning_flat_l.append(res_k['tuning_flat'])
        occupancy_flat_l.append(res_k['occupancy_flat'])
        occupancy_smth_flat_l.append(res_k['occupancy_smth_flat'])
        spk_count_flat_l.append(res_k['spk_count_flat'])
        spk_count_smth_flat_l.append(res_k['spk_count_smth_flat'])
        # add maze_key to coord tuples
        for ct in res_k['coord_tuples']:
            coord_tuples_all.append(ct + (k,))
    
    tuning_res['tuning_flat'] = np.concatenate(tuning_flat_l, axis=0) if tuning_flat_l else np.array([])
    tuning_res['occupancy_flat'] = np.concatenate(occupancy_flat_l, axis=0) if occupancy_flat_l else np.array([])
    tuning_res['occupancy_smth_flat'] = np.concatenate(occupancy_smth_flat_l, axis=0) if occupancy_smth_flat_l else np.array([])
    tuning_res['spk_count_flat'] = np.concatenate(spk_count_flat_l, axis=0) if spk_count_flat_l else np.array([])
    tuning_res['spk_count_smth_flat'] = np.concatenate(spk_count_smth_flat_l, axis=0) if spk_count_smth_flat_l else np.array([])
    
    # build coord_to_flat_idx with maze_key as additional level
    if coord_tuples_all:
        # get label_dim_names from first maze (assume same across mazes)
        label_dim_names = per_maze[maze_keys[0]]['label_dim_names']
        midx = pd.MultiIndex.from_tuples(coord_tuples_all, names=label_dim_names + ['maze'])
        coord_to_flat_idx = pd.Series(np.arange(len(coord_tuples_all)), index=midx)
    else:
        coord_to_flat_idx = pd.Series(dtype=int)
    
    tuning_res['coord_to_flat_idx'] = coord_to_flat_idx
    
    return tuning_res
