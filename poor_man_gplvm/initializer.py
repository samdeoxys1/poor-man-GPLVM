import sklearn
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler
import jax.numpy as jnp
import jax.random as jr
from jax.scipy.special import logsumexp
import pynapple as nap
import numpy as np
import pandas as pd

def init_with_pca(y,n_latent_bin,n_pca_components=None,noise_scale=0,key=jr.PRNGKey(0),**kwargs):
    '''
    initialize the latent with PCA
    First make sure n_latent_bin < n_neuron, later can think of extension 
    do PCA to preserve the time to time correlation, then convert to latent probability

    y: n_time x n_neuron, data
    n_latent_bin: number of latent bins
    n_pca_components: number of PCA components, default is n_latent_bin
    noise_scale: scale of noise to add to the PCA latent, default is 0
    key: random key

    '''
    assert n_latent_bin < y.shape[1], 'n_latent_bin should be less than n_neuron'

    if n_pca_components is None:
        n_pca_components = n_latent_bin
    pca = PCA(n_components=n_pca_components)
    pca.fit(y)
    pca_latent = pca.transform(y)
    
    if noise_scale > 0:
        pca_latent = pca_latent + jr.normal(key,shape=pca_latent.shape) * noise_scale
    pca_latent_norm =pca_latent / jnp.linalg.norm(pca_latent,axis=1,keepdims=True) # make sure elements are not too extreme; divide by norm at each time
    log_p_latent = pca_latent_norm - logsumexp(pca_latent_norm,axis=1,keepdims=True)
    return log_p_latent


# initialize with supervised label
def init_with_label_1D(label_tsd,n_latent_bin=100,t_l=None,key=jr.PRNGKey(0),noise_scale=1e-3):
    '''
    given label, tsd, bin the label, assign the corresponding latent probability to be 1, the rest to be 0, then add some noise
    label_tsd: pynapple Tsd, value of the label
    n_latent_bin: number of latent bins
    t_l: time stamps for the binned spikes (and thus the latents); if None then assume label_tsd is already aligned to the binned spikes
    key: random key
    noise_scale: scale of noise to add to the latent initialization
    '''
    if t_l is not None:
        if isinstance(t_l,np.ndarray):
            t_l = nap.Ts(t_l)
        label_tsd=t_l.value_from(label_tsd)
    label_binned = pd.cut(label_tsd,bins=n_latent_bin,retbins=True,labels=False)
    log_p_latent = np.zeros((label_tsd.shape[0],n_latent_bin))
    log_p_latent[np.arange(label_tsd.shape[0]),label_binned[0]] = 1
    log_p_latent += jr.uniform(key,shape=log_p_latent.shape) * noise_scale
    log_p_latent = log_p_latent - logsumexp(log_p_latent,axis=1,keepdims=True)
    return log_p_latent
    