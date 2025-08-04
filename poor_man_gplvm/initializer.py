from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler
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

# [TODO] :initialize with supervised label
def init_with_label():
    pass