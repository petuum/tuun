"""
Code for tuun.
"""

from .acq import (
    AcqFunction,
    AcqOptimizer,
    SpoAcqOptimizer,
    CobylaAcqOptimizer,
    NelderMeadAcqOptimizer,
)
from .bo import SimpleBo
from .domain import RealDomain, ListDomain
from .models import SimpleGp
from .tuun import Tuun


# Stan models
try:
    from .models import StanGp
except:
    pass


# Scikit-learn models
try:
    from .models import SklearnPenn
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
    'SimpleGp',
    'SimpleBo',
    'Tuun',
]
