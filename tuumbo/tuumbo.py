"""
Classes for tuumbo.
"""

import time
from argparse import Namespace
import multiprocessing
import numpy as np
import copy

from .util.misc_util import dict_to_namespace


class Tuumbo:
    """
    Class to carry out tuumbo: tuning and uncertainty-model-based optimization.
    """

    def __init__(data, model=None, acqfunction=None, acqoptimizer=None,
                 params=None, verbose=True, seed=None):
        """
        Parameters
        ----------
        data : Namespace_or_dict
            Namespace or dict of initial data. Contains fields x and y (lists).
        model : Model
            Model instance. Default is None, which uses a random acquisition
            function. 
        acqfunction : AcqFunction_or_Namespace_or_dict
            AcqFunction instance or Namespace/dict of parameters. Default is
            None, which uses a random acquisition function.
        acqoptimizer : AcqOptimizer_or_Namespace_or_dict
            AcqOptimizer instance or Namespace/dict of parameters. Default is
            None, which uses random search.
        params : Namespace_or_dict
            Namespace or dict of additional parameters for tuumbo.
        verbose : bool
            If True, print description string.
        seed : int
            If not None, set numpy random seed to seed.

        Returns
        -------
        None
        """
        self.set_random_seed(seed)
        self.set_params(params)
        self.set_data(data)
        self.set_model(model)
        self.set_acqfunction(acqfunction)
        self.set_acqoptimizer(acqoptimizer)
        self.set_verbose(verbose)

    def set_random_seed(self, seed):
        """Set numpy random seed."""
        if seed is not None:
            self.seed = seed
            np.random.seed(self.seed)
        else:
            self.seed = -1

    def set_params(self, params):
        """Set self.params, the parameters for tuumbo."""
        params = dict_to_namespace(params)
        self.params = Namespace()
        self.params.n_iter = getattr(params, 'n_rep', 1)

    def set_data(self, data):
        """Set self.data, the dataset to be modeled."""
        data = dict_to_namespace(data)

        if data is None:
            data = Namespace()

        if not hasattr(data, 'x_init') and not hasattr(data, 'x'):
            raise Exception('Input data must contain x and/or x_init')

        if not hasattr(data, 'x_init'):
            data.x_init = []

        if hasattr(data, 'x') and not hasattr(data, 'y'):
            data.x_init.extend(copy.deepcopy(data.x))
            data.x = []

        if hasattr(data, 'y') and not hasattr(data, 'x'):
            raise Exception('Input data contains y but not x')

        if not hasattr(data, 'x'):
            data.x = [] 

        if not hasattr(data, 'y'):
            data.y = []

        self.data = data
        self.n_obs_init = self.data.y.shape[0]

    def set_model(self, model):
        """Set self.model, the model used by tuumbo."""
        self.model = model

    def set_acqfunction(self, acqfunction):
        """
        Set self.acqfunction, which represents the acquisition function.
        
        Parameters
        ----------
        acqfunction : AcqFunction_or_Namespace_or_dict
            AcqFunction instance or Namespace/dict of parameters. Default is
            None, which uses a random acquisition function.
        """
        acqfunction = dict_to_namespace(acqfunction)

        if acqfunction is None:
            self.acqfunction = get_default_acqfunction(verbose=True)
        elif isinstance(acqfunction, Namespace):
            self.acqfunction = get_acqfunction_from_config(acqfunction,
                                                           verbose=True)
        else:
            # If not Namespace or dict, treat as AcqFunction instance
            self.acqfunction = acqfunction

    def set_acqoptimizer(self, acqoptimizer):
        """
        Set self.acqoptimizer, which optimizes the acquisition function.
        
        Parameters
        ----------
        acqoptimizer : AcqOptimizer_or_Namespace_or_dict
            AcqOptimizer instance or Namespace/dict of parameters. Default is
            None, which uses random search.
        """
        acqoptimizer = dict_to_namespace(acqoptimizer)

        if acqoptimizer is None:
            self.acqoptimizer = get_default_acqoptimizer(self.data,
                                                         verbose=True)
        elif isinstance(acqoptimizer, Namespace):
            self.acqoptimizer = get_acqoptimizer_from_config(acqoptimizer,
                                                             verbose=True)
        else:
            # If not Namespace or dict, treat as AcqOptimizer instance
            self.acqoptimizer = acqoptimizer

    def set_verbose(self, verbose):
        """Set verbose options."""
        self.verbose = verbose
        if self.verbose:
            self.print_str()

    def print_str(self):
        """Print a description string."""
        print('*tuumbo with params={}'.format(self.params))
