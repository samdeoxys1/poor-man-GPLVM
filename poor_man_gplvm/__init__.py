"""Poor Man's Gaussian Process Latent Variable Model (GPLVM)."""

__version__ = "0.1.0" 

# Import specific classes for convenience
from poor_man_gplvm.core import (
    AbstractGPLVM1D, PoissonGPLVM1D, GaussianGPLVM1D,  # Latent-only models
    AbstractGPLVMJump1D, PoissonGPLVMJump1D, GaussianGPLVMJump1D  # Models with dynamics
)

# Import modules to enable pmg.core, pmg.utils access
from . import core,test,utils,model_selection_helper,experimental,plot_helper