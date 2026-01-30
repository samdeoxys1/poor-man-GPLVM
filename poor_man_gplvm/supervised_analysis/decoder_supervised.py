'''
wrappers of the decoder.py, specifically for supervised analysis
'''

from ast import Pass
import jax.numpy as jnp
import jax
import jax.random as jr
import jax.scipy as jscipy
from jax.scipy.special import logsumexp
from jax import jit, vmap
from functools import partial
from jax.lax import scan
import tqdm

from poor_man_gplvm.decoder import get_naive_bayes_ma_chunk

def decode_naive_bayes(spk,tuning,):
    '''
    spk: n_time x n_neuron, or padded tensor: n_trial x n_time x n_neuron
    '''
    pass

def _decode_naive_bayes_matrix():
    pass

def _decode_naive_bayes_tensor():
    pass
