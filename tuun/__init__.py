"""
Code for tuun.
"""

from .backend import ProboBackend, DragonflyBackend

from .main import Tuun

from .nni_tuner import TuunTuner

from .probo import (
    AcqFunction,
    AcqOptimizer,
    CobylaAcqOptimizer,
    NelderMeadAcqOptimizer,
    SpoAcqOptimizer,
    ProductAcqOptimizer,
    ListDomain,
    RealDomain,
    ProductDomain,
    SimpleGp,
    SimpleBo,
    AcqOptDesigner,
)

# ProBO Stan models
try:
    from .probo import StanGp, StanProductGp
except:
    pass

# Scikit-learn models
try:
    from .probo import SklearnPenn
except:
    pass

# Gpytorch models
try:
    from .probo import GpytorchGp, GpytorchProductGp
except:
    pass


__all__ = [
    'AcqFunction',
    'AcqOptimizer',
    'CobylaAcqOptimizer',
    'NelderMeadAcqOptimizer',
    'SpoAcqOptimizer',
    'ProductAcqOptimizer',
    'ListDomain',
    'RealDomain',
    'ProductDomain',
    'SimpleGp',
    'SimpleBo',
    'AcqOptDesigner',
]
