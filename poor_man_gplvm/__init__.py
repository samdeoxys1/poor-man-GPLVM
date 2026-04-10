"""Poor Man's Gaussian Process Latent Variable Model (GPLVM)."""

__version__ = "0.1.0" 

# Import specific classes for convenience
from poor_man_gplvm.core import (
    AbstractGPLVM1D, PoissonGPLVM1D, GaussianGPLVM1D,  # Latent-only models
    AbstractGPLVMJump1D, PoissonGPLVMJump1D, GaussianGPLVMJump1D  # Models with dynamics
)

from poor_man_gplvm.core_2d import (
    AbstractGPLVMJump2D, PoissonGPLVMJump2D, GaussianGPLVMJump2D  # 2D/ND models with dynamics
)

from poor_man_gplvm.supervised_analysis.get_tuning_supervised import (
    get_tuning_supervised,
)
from poor_man_gplvm.supervised_analysis.xarray_wrappers import (
    tuning_flat_to_xr,
    latent_time_series_to_xr,
    supervised_tuning_views,
)
import poor_man_gplvm.experimental.iterative_supervised_kde as iterative_supervised_kde

fit_decode_iterative_supervised_kde = iterative_supervised_kde.fit_decode_iterative_supervised_kde

# Import modules to enable pmg.core, pmg.utils access
from . import core,core_2d,test,utils,model_selection_helper,experimental,plot_helper,analysis_helper,distance_analysis