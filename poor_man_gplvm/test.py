'''
tests for validating the model, latents, etc.
tentative and subject to change
'''
import numpy as np
import pynapple as nap
import jax.numpy as jnp

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

