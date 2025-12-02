# Poor Man's GPLVM

An efficient Jax-based implementation of Gaussian Process Latent Variable Models (GPLVM) that also allows the inference of latent dynamics (continuous vs jump). Latent variables are low dimensional structure that governs neural covariability. This model learns: 
  1. smooth tuning curves of neurons with respect to the latent variables
  2. posterior probability of the (discretized) latent variables and 
  3. posterior probability of the dynamics of the latent evolution (i.e. smoothly varying or jumping)

## Installation

### GPU Installation

#### Non-editable installation

```bash
# Create a new conda environment with all dependencies
conda create -n pmgplvm -c conda-forge -c nvidia cuda-nvcc jaxlib=0.4.26=cuda120py312h4008524_201 jax=0.4.26 python=3.12.5 jaxopt=0.8.2 optax=0.2.2 numpy scipy tqdm=4.66.5 xarray=2024.3.0 matplotlib=3.9.2 plotly=5.24.1 seaborn=0.13.1 scikit-learn=1.5.2 statsmodels=0.14.3

# Activate the environment
conda activate pmgplvm

# Clone the repository
git clone https://github.com/samdeoxys1/poor-man-GPLVM.git
cd poor-man-GPLVM

# Install
pip install .
```

#### Editable installation (for development)

```bash
# Create a new conda environment with all dependencies
conda create -n pmgplvm -c conda-forge -c nvidia cuda-nvcc jaxlib=0.4.26=cuda120py312h4008524_201 jax=0.4.26 python=3.12.5 jaxopt=0.8.2 optax=0.2.2 numpy scipy tqdm=4.66.5 xarray=2024.3.0 matplotlib=3.9.2 plotly=5.24.1 seaborn=0.13.1 scikit-learn=1.5.2 statsmodels=0.14.3

# Activate the environment
conda activate pmgplvm

# Clone the repository
git clone https://github.com/samdeoxys1/poor-man-GPLVM.git
cd poor-man-GPLVM

# Install in editable mode
pip install -e .
```

### CPU-Only Installation

#### Non-editable installation

```bash
# Create a new conda environment with all dependencies
conda create -n pmgplvm -c conda-forge python=3.12.5 jax=0.4.26 jaxlib=0.4.26 jaxopt=0.8.2 optax=0.2.2 numpy scipy tqdm=4.66.5 xarray=2024.3.0 matplotlib=3.9.2 plotly=5.24.1 seaborn=0.13.1 scikit-learn=1.5.2 statsmodels=0.14.3

# Activate the environment
conda activate pmgplvm

# Clone the repository
git clone https://github.com/samdeoxys1/poor-man-GPLVM.git
cd poor-man-GPLVM

# Install
pip install .
```

#### Editable installation (for development)

```bash
# Create a new conda environment with all dependencies
conda create -n pmgplvm -c conda-forge python=3.12.5 jax=0.4.26 jaxlib=0.4.26 jaxopt=0.8.2 optax=0.2.2 numpy scipy tqdm=4.66.5 xarray=2024.3.0 matplotlib=3.9.2 plotly=5.24.1 seaborn=0.13.1 scikit-learn=1.5.2 statsmodels=0.14.3

# Activate the environment
conda activate pmgplvm

# Clone the repository
git clone https://github.com/samdeoxys1/poor-man-GPLVM.git
cd poor-man-GPLVM

# Install in editable mode
pip install -e .
```

## Usage

Here's a quick example of how to use the package:

```python
import numpy as np
import jax.numpy as jnp
import jax.random as jr
import poor_man_gplvm as pmg
import matplotlib.pyplot as plt

# initialize model
n_neuron = 30
model=pmg.PoissonGPLVMJump1D(n_neuron,n_latent_bin=100,movement_variance=1,tuning_lengthscale=10.) # can modify the hyperparameters

#### additional hyperparameters and their default values
# param_prior_std=1 (L2 penalty of the weights)
# p_move_to_jump=0.01 (Transition probability from continuous to fragmented dynamics)
# p_jump_to_move=0.01 (Transition probability from fragmented to continuous dynamics)
# w_init_variance=1, w_init_mean (mean and variance of the Gaussian initialization of the basis weights)

# simulate some data

T = 1000
state_l, spk = model.sample(T)
y = spk # data for fitting is n_time x n_neuron; binned spike counts

# fit the model 
em_res=model.fit_em(y,key=jr.PRNGKey(3),
                    n_iter=20,
                      posterior_init=None,ma_neuron=None,ma_latent=None,n_time_per_chunk=10000
                   )

# monitor training
plt.plot(em_res['log_marginal_l'])

# to decode (potentially) new data
decode_res = model.decode_latent(y)

### useful variables:

# tuning curves
model.tuning # n_neuron x n_latent_bin

# latent posterior
decode_res['posterior_latent_marg'] # n_time x n_latent_bin   
# dynamics posterior
decode_res['posterior_dynamics_marg'] # n_time x 2 

# jump probability
decode_res['posterior_dynamics_marg'][:,1]
# continuous probability = 1 - jump probability
decode_res['posterior_dynamics_marg'][:,0]

# posterior transition probability, p(x_k+1|x_k,O_1:T), x: latent or dynamics, O: observations (spikes)
decode_res['p_transition_latent'] # n_latent_bin x n_latent_bin
decode_res['p_transition_dynamics'] # n_dynamics x n_dynamics


# after fitting one can also decode without temporal dynamics, using Naive Bayes
decode_res_nb = model.decode_latent_naive_bayes(y)

# NB latent posterior
decode_res_nb['posterior_latent']

# Order the neurons based on latent tuning
import poor_man_gplvm.utils as ut
sort_to_return = ut.post_fit_sort_neurons(em_res) # em_res is from model.fit_em, or constructed from: em_res = {'tuning':model_fit.tuning}
argsort = sort_to_return[argsort] # this can be used to order the neurons for rasterplots





```
