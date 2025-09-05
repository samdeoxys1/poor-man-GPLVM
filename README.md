# Poor Man's GPLVM

A simplified implementation of Gaussian Process Latent Variable Models (GPLVM). This package provides an easy-to-use interface for dimensionality reduction and visualization using GPLVMs.

## Installation

### Recommended: Conda Installation

The recommended way to install this package is through conda, which correctly handles the JAX, CUDA, and other dependencies:

```bash
# Create a new conda environment with all required dependencies
conda create -n pmgplvm -c conda-forge -c nvidia cuda-nvcc jaxlib=0.4.26=cuda120py312h4008524_201 jax=0.4.26 python=3.12.5 jaxopt=0.8.2 optax=0.2.2

# Activate the environment
conda activate pmgplvm

# Install poor-man-gplvm
pip install poor-man-gplvm  # (once published)
```

### For CPU-Only Installation

If you don't have a compatible GPU:

```bash
conda create -n pmgplvm -c conda-forge python=3.12.5 jax=0.4.26 jaxlib=0.4.26 jaxopt=0.8.2 optax=0.2.2
conda activate pmgplvm
pip install poor-man-gplvm  # (once published)
```

### From Source

To install the latest development version:

```bash
# First create and set up the conda environment as above
# Then:
git clone https://github.com/samdeoxys1/poor-man-GPLVM.git
cd poor-man-GPLVM
pip install -e .
```

For more detailed installation instructions and troubleshooting, see [INSTALL.md](INSTALL.md).

## Usage

Here's a quick example of how to use the package:

```python
import numpy as np
import jax.numpy as jnp
import jax.random as jr
import poor_man_gplvm as pmg

# initialize model
n_neuron = 100
model=pmg.PoissonGPLVMJump1D(n_neuron,movement_variance=1,tuning_lengthscale=10.)

# simulate some data

T = 10000
state_l, spk = model.sample(T)
y = spk # data for fitting is n_time x n_neuron 
em_res=model.fit_em(y,key=jax.random.PRNGKey(3),
                    n_iter=20,
                      posterior_init=None,ma_neuron=None,ma_latent=None,n_time_per_chunk=10000,dt=1.,likelihood_scale=1.,
                      save_every=None,m_step_tol=1e-10,
                    posterior_init_kwargs={'random_scale':1}
                   )

# to decode (potentially) new data
decode_res = model.decode_latent(y)

# useful properties:

# tuning curves
model.tuning_fit

# latent posterior
decode_res['posterior_latent_marginal']
# dynamics posterior
decode_res['posterior_dynamics_marginal']



```

## Features


## Development

### Setting up the development environment

```bash
# Create and activate the conda environment first
conda create -n pmgplvm -c conda-forge -c nvidia cuda-nvcc jaxlib=0.4.26=cuda120py312h4008524_201 jax=0.4.26 python=3.12.5 jaxopt=0.8.2 optax=0.2.2
conda activate pmgplvm

# Then install the package in development mode
git clone https://github.com/samdeoxys1/poor-man-GPLVM.git
cd poor-man-GPLVM
pip install -e ".[dev]"
```

### Running tests

```bash
pytest
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Citation

If you use this software in your research, please cite:

```
@software{poor_man_gplvm,
  author = {Zheyang Sam Zheng},
  title = {Poor Man's GPLVM: A simplified implementation of Gaussian Process Latent Variable Models},
  year = {2024},
  url = {https://github.com/samdeoxys1/poor-man-GPLVM}
}
```
