"""Poor Man's Gaussian Process Latent Variable Model (GPLVM)."""

__version__ = "0.1.0" 

# Import specific classes for convenience
from poor_man_gplvm.core import (
    AbstractGPLVM1D, PoissonGPLVM1D, GaussianGPLVM1D,  # Latent-only models
    AbstractGPLVMJump1D, PoissonGPLVMJump1D, GaussianGPLVMJump1D  # Models with dynamics
)

from poor_man_gplvm.supervised_analysis.get_tuning_supervised import (
    get_tuning_supervised,
)

# Import modules to enable pmg.core, pmg.utils access
from . import core,test,utils,model_selection_helper,experimental,plot_helper,analysis_helper,distance_analysis