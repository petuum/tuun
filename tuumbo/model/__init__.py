"""
Code for probabilistic programming language models for tuumbo.
"""

from .gp_simple import SimpleGp
#from .gp_distmat import SimpleDistmatGp

# Sklearn models
try:
    from .nn_sklearn import SklearnHpnn
    from .penn_sklearn import SklearnPenn
except:
    pass

# Stan models
try:
    from .gp_stan import StanGp
    from .gp_stan_distmat import StanDistmatGp
except:
    pass

# Pyro models
try:
    from .bnn_pyro import PyroBnn
except:
    pass


__all__ = ['SimpleGp']
#__all__ = ['SimpleGp',
           #'SimpleDistmatGp',
           #'PyroBnn',
           #'SklearnHpnn',
           #'SklearnPenn',
           #'StanGp',
           #'StanDistmatGp']
