"""Utility functions for the Poor Man's GPLVM."""

import numpy as np
import pynapple as nap

def rbf_kernel(X, Y=None, length_scale=1.0):
    """Radial Basis Function kernel.
    
    Parameters
    ----------
    X : array-like of shape (n_samples_X, n_features)
        Left argument of the kernel
    Y : array-like of shape (n_samples_Y, n_features), default=None
        Right argument of the kernel. If None, Y=X.
    length_scale : float, default=1.0
        Length scale parameter
        
    Returns
    -------
    K : ndarray of shape (n_samples_X, n_samples_Y)
        Kernel matrix
    """
    X = np.asarray(X)
    if Y is None:
        Y = X
    else:
        Y = np.asarray(Y)
        
    XX = np.sum(X**2, axis=1)[:, np.newaxis]
    YY = np.sum(Y**2, axis=1)[np.newaxis, :]
    XY = np.dot(X, Y.T)
    
    # Compute squared euclidean distance
    sq_dists = XX + YY - 2 * XY
    
    # Apply RBF kernel
    K = np.exp(-0.5 * sq_dists / (length_scale**2))
    
    return K


# post fit neuron sorting + normalization; used in post fit visualization
def post_fit_sort_neuron(fit_res,spk=None,do_norm='max',method='tuning_peak',t_l=None):
    '''
    fit_res: dict, usually from fit_em, but if that is not available, usuallly can do {'tuning':model_fit.tuning}; this way to ensure other types of sorting is compatible in the future

    fit result: can include tuning or posterior, depending on the sorting method
    spk: binned spike: n_time x n_neuron, if not provided, then only return the argsort; if provided, return the sorted and optionally normalized spk
    # if t_l (timestamps) is provided, return a TsdFrame
    '''
    if method == 'tuning_peak':
        assert 'tuning' in fit_res, "Tuning is not in the fit result"
        tuning = fit_res['tuning']
        argsort=np.argsort(np.argmax(tuning[:,],axis=0))

    else:
        raise ValueError(f"Invalid method: {method}")
    to_return = {}
    if spk is not None:
        if do_norm == 'max':
            spk_to_plot = spk / spk.max(axis=0,keepdims=True)
        elif do_norm == 'zscore':
            spk_to_plot = (spk - spk.mean(axis=0,keepdims=True)) / spk.std(axis=0,keepdims=True)
        elif do_norm==None:
            spk_to_plot = spk
        else:
            raise ValueError(f"Invalid normalization method: {do_norm}")
        spk_no_sort = spk_to_plot # no sort but with normalization
        spk_to_plot = spk_to_plot[:,argsort]
        if t_l is not None: 
            spk_to_plot = nap.TsdFrame(d=spk_to_plot,t=t_l)
            spk_no_sort = nap.TsdFrame(d=spk_no_sort,t=t_l)
        to_return['spk_to_plot'] = spk_to_plot
        to_return['spk_no_sort'] = spk_no_sort

    
    to_return['argsort'] = argsort
    return to_return



# tested so far not good; not used
def pca_init(Y, latent_dim):
    """Initialize latent points using PCA.
    
    Parameters
    ----------
    Y : array-like of shape (n_samples, n_features)
        Observed data
    latent_dim : int
        Dimensionality of the latent space
        
    Returns
    -------
    X : ndarray of shape (n_samples, latent_dim)
        Initial latent points
    """
    Y = np.asarray(Y)
    n_samples = Y.shape[0]
    
    # Center the data
    Y_centered = Y - np.mean(Y, axis=0)
    
    # Compute SVD
    U, S, Vh = np.linalg.svd(Y_centered, full_matrices=False)
    
    # Return the first latent_dim principal components
    X = U[:, :latent_dim] * S[:latent_dim]
    
    return X 

def restrict_xr(xr_data,intv,time_var='time'):
    '''
    restrict xarray dataarray to a given interval, similar to pynapple .restrict
    xr_data: xarray dataarray
    intv: interval, can be a tuple of (start, end) or a pynapple IntervalSet
    '''
    if not isinstance(intv,nap.IntervalSet):
        intv = nap.IntervalSet(intv)
    xr_data_restricted= []
    for win in intv:
        xr_data_ = xr_data.sel({time_var: slice(win[0,0], win[0,1])})
        xr_data_restricted.append(xr_data_)
    xr_data_restricted = xr.concat(xr_data_restricted,dim=time_var)
    return xr_data_restricted

