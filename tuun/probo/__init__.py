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
from .domain import RealDomain, ListDomain, ProductDomain
from .models import SimpleGp
from .design import AcqOptDesigner

# Stan models
try:
    from .models import StanGp
    from .models import StanProductGp
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


__all__ = [
    'AcqFunction',
    'AcqOptimizer',
    'CobylaAcqOptimizer',
    'NelderMeadAcqOptimizer',
    'SpoAcqOptimizer',
    'ListDomain',
    'RealDomain',
    'ProductDomain',
    'SimpleGp',
    'SimpleBo',
    'AcqOptDesigner',
]
