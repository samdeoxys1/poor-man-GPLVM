'''
tests for validating the model, latents, etc.
tentative and subject to change
'''
import numpy as np
import pynapple as nap
import jax.numpy as jnp
import tqdm

def circular_shuffle_data(spk_tsdf,n_shuffle=100,ep=None):
    '''
    circular shuffle the data
    circular shuffle each neuron independently
    spk_tsdf: binned spike, pyanpple TsdFrame or np.ndarray
    '''
    if ep is not None:
        assert isinstance(spk_tsdf,nap.TsdFrame), "input data must be a pynapple TsdFrame"
        spk_tsdf = spk_tsdf.restrict(ep)
    n_time,n_neuron = spk_tsdf.d.shape
    for i in range(n_shuffle):
        spk_tsdf_shuffled = spk_tsdf.copy()
        for j in range(n_neuron):
            spk_tsdf_shuffled[:,j] = np.roll(spk_tsdf[:,j],np.random.randint(0,n_time))
        yield jnp.array(spk_tsdf_shuffled)


def shuffle_and_decode(model,spk_tsdf,n_time_per_chunk=10000,dt_l=1,n_shuffle=100,ep=None,decoder_type='naive_bayes'):
    '''
    shuffle the data and decode the latent
    decoder_type: 'naive_bayes' or 'dynamics'; dynamics is the one used for EM, the bayesian smoother, with different dynamics on latent; 
    '''
    y_shuffled_l = circular_shuffle_data(spk_tsdf,n_shuffle=n_shuffle,ep=ep)
    decoding_res_l = []
    for y_shuffled in tqdm.tqdm(y_shuffled_l,total=n_shuffle):
        if decoder_type == 'naive_bayes':
            decoding_res = model.decode_latent_naive_bayes(y_shuffled,n_time_per_chunk=n_time_per_chunk,dt_l=dt_l)
        elif decoder_type == 'dynamics':
            decoding_res = model.decode_latent(y_shuffled,n_time_per_chunk=n_time_per_chunk)
        else:
            raise ValueError(f"decoder_type {decoder_type} not supported")
        decoding_res_l.append(decoding_res)
    
    # reshape the decoding_res_l to each key having n_shuffle elements
    decoding_res_l = {k:np.array([d[k] for d in decoding_res_l]) for k in decoding_res_l[0].keys()}
    return decoding_res_l

def compute_entropy(logp_l,axis=(-1,-2)):
    '''
    logp_l: n_time x n_latent or n_time x ... , by default vmap over n_time, do entropy over the rest dimensions
    axis: axis to collapse, default is (-1,-2)

    return entropy_l: axis to keep or scalar (if axis is None)
    '''
    
    entropy_l = -np.sum(np.exp(logp_l) * logp_l,axis=axis)
    return entropy_l
