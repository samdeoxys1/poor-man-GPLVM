# Poor Man's GPLVM

A simplified implementation of Gaussian Process Latent Variable Models (GPLVM) that also allows the inference of latent dynamics (continuous vs jump). 

## Installation

### GPU Installation

#### Non-editable installation

```bash
# Create a new conda environment with all dependencies
conda create -n pmgplvm -c conda-forge -c nvidia cuda-nvcc jaxlib=0.4.26=cuda120py312h4008524_201 jax=0.4.26 python=3.12.5 jaxopt=0.8.2 optax=0.2.2 numpy scipy tqdm=4.66.5 xarray=2024.3.0 matplotlib=3.9.2 plotly=5.24.1 seaborn=0.13.1 scikit-learn=1.5.2 pynapple=0.9.2 statsmodels=0.14.3

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
conda create -n pmgplvm -c conda-forge -c nvidia cuda-nvcc jaxlib=0.4.26=cuda120py312h4008524_201 jax=0.4.26 python=3.12.5 jaxopt=0.8.2 optax=0.2.2 numpy scipy tqdm=4.66.5 xarray=2024.3.0 matplotlib=3.9.2 plotly=5.24.1 seaborn=0.13.1 scikit-learn=1.5.2 pynapple=0.9.2 statsmodels=0.14.3

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
conda create -n pmgplvm -c conda-forge python=3.12.5 jax=0.4.26 jaxlib=0.4.26 jaxopt=0.8.2 optax=0.2.2 numpy scipy tqdm=4.66.5 xarray=2024.3.0 matplotlib=3.9.2 plotly=5.24.1 seaborn=0.13.1 scikit-learn=1.5.2 pynapple=0.9.2 statsmodels=0.14.3

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
conda create -n pmgplvm -c conda-forge python=3.12.5 jax=0.4.26 jaxlib=0.4.26 jaxopt=0.8.2 optax=0.2.2 numpy scipy tqdm=4.66.5 xarray=2024.3.0 matplotlib=3.9.2 plotly=5.24.1 seaborn=0.13.1 scikit-learn=1.5.2 pynapple=0.9.2 statsmodels=0.14.3

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
n_neuron = 100
model=pmg.PoissonGPLVMJump1D(n_neuron,movement_variance=1,tuning_lengthscale=10.)

# simulate some data

T = 10000
state_l, spk = model.sample(T)
y = spk # data for fitting is n_time x n_neuron 
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


# after fitting one can also decode without temporal dynamics, using Naive Bayes
decode_res_nb = model.decode_latent_naive_bayes(y)

# NB latent posterior
decode_res_nb['posterior_latent']
```
