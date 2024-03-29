"""
Models for use in ProBO.
"""

from .gp_simple import SimpleGp
from .gp_simple_productkernel import SimpleProductKernelGp

# from .gp_distmat import SimpleDistmatGp

# Sklearn models
try:
    from .penn_sklearn import SklearnPenn
except:
    pass

try:
    from .nn_sklearn import SklearnHpnn
except:
    pass

# Stan models
try:
    from .gp_stan import StanGp
    from .gp_stan_product import StanProductGp
    from .gp_stan_distmat import StanDistmatGp
    from .gp_stan_transfer import StanTransferGp
except:
    pass

# GPyStan models
try:
    from .gp_gpystan import GpystanGp
except:
    pass

# GPyTorch models
try:
    from .gp_gpytorch import GpytorchGp
    from .gp_gpytorch_product import GpytorchProductGp
except:
    pass

# Pyro models
try:
    from .bnn_pyro import PyroBnn
except:
    pass
