import numpy as np
import scipy
import sys,pdb,copy
import jax
from jax import lax
# jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import jax.random as jxr
import jax.random as jr
import jax.scipy as jscipy
from jax.scipy.special import logsumexp
import matplotlib.pyplot as plt
import tqdm
from tqdm import trange

from jax import vmap,jit,grad,hessian

