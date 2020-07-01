"""
Models for use in tuun.
"""

from .gp_simple import SimpleGp

# from .gp_distmat import SimpleDistmatGp

# Sklearn models
try:
    from .nn_sklearn import SklearnHpnn
    from .penn_sklearn import SklearnPenn
except:
    pass

# Stan models
try:
    from .gp_stan import StanGp
except:
    pass

# Pyro models
try:
    from .bnn_pyro import PyroBnn
except:
    pass
