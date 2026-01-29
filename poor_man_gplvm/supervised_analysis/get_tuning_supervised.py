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


tuning_flat: n_label_bin x n_neuron
tuning: xr.DataArray or {maze_key : xr.DataArray} for multiple mazes
    firing rate in Cartesian product of the multiple label dimensions
    the unoccupied bins are filled with nan; when flattening it the nan is dropped

    tuning is computed by smoothed spk count / smoothed occupancy
    smoothing is done assuming independent Gaussian kernel for each dimension
occupancy:
occupancy_smth:
spk_count:
spk_count_smth:
label_grid: (n_label_bin_1+1) x (n_label_bin_2+1) x...
label_grid_centers: n_label_bin_1 x n_label_bin_2 x...
'''

import numpy as np
import pynapple as nap
import xarray as xr

# ============================================================================
# Helper: normalize inputs for single vs multi-maze
# ============================================================================

def _to_dict(val, maze_keys, name=''):
    '''convert single value to dict keyed by maze_keys if not already a dict'''
    if isinstance(val, dict):
        return val
    return {k: val for k in maze_keys}


def _normalize_inputs(label_l, spk_mat, ep=None, custom_smooth_func=None, 
                      label_bin_size=1., smooth_std=None):
    '''
    Normalize single vs multi-maze inputs.
    Returns dict-form for all inputs keyed by maze_key.
    Also aligns labels to spk_mat timestamps via interpolation.
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

    # align and restrict
    label_aligned_d = {}
    spk_aligned_d = {}
    for k in maze_keys:
        label_k = label_l[k]
        ep_k = ep_d.get(k, None)

        # align label to spk_mat timestamps
        label_aligned = label_k.interpolate(spk_mat)

        # restrict to epoch if provided
        if ep_k is not None:
            label_aligned = label_aligned.restrict(ep_k)
            spk_sub = spk_mat.restrict(ep_k)
        else:
            spk_sub = spk_mat

        # further restrict to common time support
        common_ts = label_aligned.time_support.intersect(spk_sub.time_support)
        label_aligned = label_aligned.restrict(common_ts)
        spk_sub = spk_sub.restrict(common_ts)

        # ensure same timestamps via interpolation
        spk_aligned = spk_sub.interpolate(label_aligned)

        label_aligned_d[k] = label_aligned
        spk_aligned_d[k] = spk_aligned

    return {
        'label_d': label_aligned_d,
        'spk_d': spk_aligned_d,
        'ep_d': ep_d,
        'custom_smooth_func_d': custom_smooth_func_d,
        'label_bin_size_d': label_bin_size_d,
        'smooth_std_d': smooth_std_d,
        'maze_keys': maze_keys,
        'is_multi': is_multi,
    }


# ============================================================================
# label_to_grid: build bin edges and centers
# ============================================================================

def label_to_grid(label, label_bin_size):
    '''
    Build bin edges and centers from label data.
    
    label: nap.TsdFrame (n_time x n_dim)
    label_bin_size: float or array (per dim)
    
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
    
    # broadcast bin_size
    label_bin_size = np.atleast_1d(label_bin_size)
    if label_bin_size.size == 1:
        label_bin_size = np.repeat(label_bin_size, n_dim)
    
    bin_edges_l = []
    bin_centers_l = []
    for d in range(n_dim):
        col = label_arr[:, d]
        finite_ma = np.isfinite(col)
        col_finite = col[finite_ma]
        if col_finite.size == 0:
            # fallback
            lo, hi = 0., 1.
        else:
            lo, hi = col_finite.min(), col_finite.max()
        bs = label_bin_size[d]
        # build edges from lo to hi+bs (so hi is included in last bin)
        edges = np.arange(lo, hi + bs + 1e-9, bs)
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
    '''
    n_time, n_dim = label_arr.shape
    bin_inds = np.zeros((n_time, n_dim), dtype=int)
    valid_mask = np.ones(n_time, dtype=bool)
    
    for d in range(n_dim):
        col = label_arr[:, d]
        edges = bin_edges_l[d]
        n_bins = len(edges) - 1
        
        # digitize: returns 1..n_bins for values in range, 0 or n_bins+1 for out
        inds = np.digitize(col, edges) - 1  # now 0-based
        
        # mark invalid: NaN or out of range
        invalid = ~np.isfinite(col) | (inds < 0) | (inds >= n_bins)
        valid_mask &= ~invalid
        
        # clip for safe indexing (invalids will be masked out)
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
               label_bin_size=1., smooth_std=None, occupancy_threshold=None):
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
    
    Returns
    -------
    If single maze:
        tuning : xr.DataArray, (*grid_shape, n_neuron)
        res : dict with occupancy, occupancy_smth, spk_count, spk_count_smth,
              bin_edges, bin_centers, occupied_mask, tuning_flat, dt, etc.
    If multi-maze:
        tuning_d : dict of xr.DataArray keyed by maze_key
        res_d : dict of res dicts keyed by maze_key
    '''
    # normalize inputs
    norm = _normalize_inputs(label_l, spk_mat, ep=ep, 
                             custom_smooth_func=custom_smooth_func,
                             label_bin_size=label_bin_size, 
                             smooth_std=smooth_std)
    
    label_d = norm['label_d']
    spk_d = norm['spk_d']
    custom_smooth_func_d = norm['custom_smooth_func_d']
    label_bin_size_d = norm['label_bin_size_d']
    smooth_std_d = norm['smooth_std_d']
    maze_keys = norm['maze_keys']
    is_multi = norm['is_multi']
    
    tuning_d = {}
    res_d = {}
    
    for k in maze_keys:
        label_k = label_d[k]
        spk_k = spk_d[k]
        csf_k = custom_smooth_func_d.get(k, None)
        lbs_k = label_bin_size_d.get(k, 1.)
        ss_k = smooth_std_d.get(k, None)
        
        # build grid
        bin_edges_l, bin_centers_l, grid_shape, label_dim_names = label_to_grid(label_k, lbs_k)
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
            tuning_flat = spk_count_smth_flat / occupancy_smth_flat[:, None]
        tuning_flat = np.where(np.isfinite(tuning_flat), tuning_flat, np.nan)
        
        # occupancy threshold mask (on raw occupancy)
        if occupancy_threshold is not None:
            occupied_mask = occupancy_flat >= occupancy_threshold
        else:
            occupied_mask = np.ones(n_flat, dtype=bool)
        
        # apply mask
        tuning_flat[~occupied_mask, :] = np.nan
        
        # reshape to grid
        occupancy_smth_grid = occupancy_smth_flat.reshape(grid_shape)
        spk_count_smth_grid = spk_count_smth_flat.reshape((*grid_shape, n_neuron))
        tuning_grid = tuning_flat.reshape((*grid_shape, n_neuron))
        
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
        
        # flatten valid for convenience
        valid_flat_mask = np.all(np.isfinite(tuning_flat), axis=1)
        tuning_flat_valid = tuning_flat[valid_flat_mask]
        flat_inds_valid = np.where(valid_flat_mask)[0]
        
        res = {
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
            'tuning_flat_valid': tuning_flat_valid,
            'flat_inds_valid': flat_inds_valid,
            'dt': dt,
            'smoothing_matrix': S,
            'n_valid_timepoints': valid_mask.sum(),
        }
        
        tuning_d[k] = tuning_xr
        res_d[k] = res
        
        print(f"[get_tuning] maze={k}: grid_shape={grid_shape}, n_neurons={n_neuron}, "
              f"dt={dt:.4f}s, n_valid_time={valid_mask.sum()}, n_occupied_bins={occupied_mask.sum()}")
    
    if not is_multi:
        return tuning_d['_single'], res_d['_single']
    return tuning_d, res_d
