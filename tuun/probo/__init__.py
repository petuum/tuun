"""
Code for ProBO.
"""

from .acq import (
    AcqFunction,
    AcqOptimizer,
    SpoAcqOptimizer,
    CobylaAcqOptimizer,
    NelderMeadAcqOptimizer,
    ProductAcqOptimizer,
)
from .sequence import SimpleBo
from .domain import RealDomain, ListDomain, IntegralDomain, ProductDomain
from .models import SimpleGp, SimpleProductKernelGp
from .design import AcqOptDesigner

# Stan models
try:
    from .models import StanGp, StanProductGp, StanDistmatGp
except:
    pass

# Scikit-learn models
try:
    from .models import SklearnPenn
except:
    pass

# Gpytorch models
try:
    from .models import GpytorchGp
    from .models import GpytorchProductGp
except:
    pass

# Pyro models
try:
    from .models import PyroBnn
except:
    pass


__all__ = [
    'AcqFunction',
    'AcqOptimizer',
    'CobylaAcqOptimizer',
    'NelderMeadAcqOptimizer',
    'SpoAcqOptimizer',
    'IntegralDomain',
    'ListDomain',
    'RealDomain',
    'ProductDomain',
    'SimpleGp',
    'SimpleBo',
    'AcqOptDesigner',
]
