"""
Models for use in tuun.
"""

from .gp_simple import SimpleGp

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
except:
    pass

# Gpytorch models
try:
    from .gp_gpytorch import GpytorchGp
except:
    pass

# Pyro models
try:
    from .bnn_pyro import PyroBnn
except:
    pass
