"""Setup script for poor-man-GPLVM."""

from setuptools import setup, find_packages

# Read the content of your README file
with open("README.md", encoding="utf-8") as f:
    long_description = f.read()

# Read requirements - exclude JAX-related packages
with open("requirements.txt", encoding="utf-8") as f:
    requirements = [
        line.strip() for line in f 
        if line.strip() and not line.startswith("#") 
        and "jax" not in line.lower()
        and "optax" not in line.lower()
    ]

# Development requirements
try:
    with open("requirements-dev.txt", encoding="utf-8") as f:
        dev_requirements = [line.strip() for line in f if line.strip() and not line.startswith("#")]
except FileNotFoundError:
    dev_requirements = [
        "pytest>=6.0.0",
        "pytest-cov>=2.10.0",
        "black>=22.0.0",
        "flake8>=3.8.0",
    ]

setup(
    name="poor-man-gplvm",
    version="0.1.0",
    description="A simplified implementation of Gaussian Process Latent Variable Models",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Sam Zheng",
    author_email="samzhenguchi@gmail.com",
    url="https://github.com/samdeoxys1/poor-man-GPLVM",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Computational Neuroscience",
    ],
    python_requires=">=3.7",
    install_requires=requirements,
    extras_require={
        "dev": dev_requirements,
    },
) 