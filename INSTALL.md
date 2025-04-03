# Installation Guide

This document provides detailed installation instructions for the Poor Man's GPLVM package.

## Recommended Installation Method: Using Conda

The recommended way to install this package is through conda, which handles complex dependencies like JAX, CUDA, and cuDNN properly.

### One-Step Installation with Conda

```bash
# Create a new conda environment with all required dependencies
conda create -n pmgplvm -c conda-forge -c nvidia cuda-nvcc jaxlib=0.4.26=cuda120py312h4008524_201 jax=0.4.26 python=3.12.5 jaxopt=0.8.2 optax=0.2.2

# Activate the environment
conda activate pmgplvm

# Install poor-man-gplvm
pip install poor-man-gplvm
# OR from source:
# git clone https://github.com/samdeoxys1/poor-man-GPLVM.git
# cd poor-man-GPLVM
# pip install -e .
```

This single conda command handles all the complex dependencies including:
- Python 3.12.5
- JAX 0.4.26 with CUDA 12.0 support
- JAXopt 0.8.2 for optimization
- Optax 0.2.2 for optimization algorithms
- All necessary CUDA components

### For CPU-Only Installation

If you don't have a compatible GPU, you can install a CPU-only version:

```bash
conda create -n pmgplvm -c conda-forge python=3.12.5 jax=0.4.26 jaxlib=0.4.26 jaxopt=0.8.2 optax=0.2.2
conda activate pmgplvm
pip install poor-man-gplvm
```

## Alternative: Manual Installation (Advanced Users)

If you prefer not to use conda or need a custom configuration, you can install the components separately.

### 1. Install poor-man-gplvm

```bash
pip install poor-man-gplvm
```

### 2. Install JAX Manually

Follow the [official JAX installation guide](https://github.com/google/jax#installation) to install JAX with your specific CUDA configuration.

For example:
```bash
# For CPU
pip install jax jaxlib

# For GPU with CUDA 12 support
pip install jax jaxlib==0.4.26+cuda12.cudnn89
```

### 3. Install Additional Dependencies

```bash
pip install jaxopt optax
```

## Verifying Installation

To verify the installation is working correctly:

```bash
python -c "import poor_man_gplvm; print(poor_man_gplvm.__version__)"
```

For JAX installations:

```bash
# Check JAX and GPU availability
python -c "import jax; print('JAX version:', jax.__version__); print('Available devices:', jax.devices())"
```

If JAX detects your GPU, the devices output should include `gpu:0`.

## Troubleshooting

### CUDA Not Found

If JAX cannot find your CUDA installation, ensure:
1. Your CUDA version matches the jaxlib+cuda version you've installed
2. CUDA is in your PATH
3. You've installed the correct drivers for your GPU

### Memory Issues with GPU

If you encounter GPU memory errors, you might need to limit JAX's GPU memory usage:

```python
import os
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'
os.environ['XLA_PYTHON_CLIENT_ALLOCATOR'] = 'platform'
```

### Version Conflicts

If you're experiencing version conflicts between packages, the conda installation method is strongly recommended as it resolves these conflicts automatically. 