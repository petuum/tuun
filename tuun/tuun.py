"""
Classes for tuun.
"""

import time
from argparse import Namespace
import multiprocessing
import numpy as np
import copy

from .util.misc_util import dict_to_namespace
from .acq.acqfun import get_acqfunction_from_config
from .acq.acqopt import get_acqoptimizer_from_config


np.set_printoptions(precision=3, suppress=True)


class Tuun:
    """
    Class to carry out tuning via uncertainty modeling.
    """

    def __init__(
        self,
        data=None,
        model=None,
        acqfunction=None,
        acqoptimizer=None,
        params=None,
        verbose=True,
        seed=None,
    ):
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
            Namespace or dict of additional parameters for tuun.
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
        """Set self.params, the parameters for tuun."""
        params = dict_to_namespace(params)
        self.params = Namespace()
        self.params.n_rep = getattr(params, 'n_rep', 1)

    def set_data(self, data):
        """Set self.data, the dataset to be modeled."""
        data = copy.deepcopy(data)
        data = dict_to_namespace(data)

        if data is None:
            data = Namespace()

        if not hasattr(data, 'x') and not hasattr(data, 'y'):
            data.x = []
            data.y = []

        if not hasattr(data, 'x') or not hasattr(data, 'y'):
            raise Exception('Input data must contain lists x and y')

        self.data = data

    def set_model(self, model):
        """Set self.model, the model used by tuun."""
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
            self.acqfunction = get_acqfunction_from_config(None, verbose=True)
        elif isinstance(acqfunction, Namespace):
            self.acqfunction = get_acqfunction_from_config(acqfunction, verbose=True)
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
            self.acqoptimizer = get_acqoptimizer_from_config(None, verbose=True)
        elif isinstance(acqoptimizer, Namespace):
            self.acqoptimizer = get_acqoptimizer_from_config(acqoptimizer, verbose=True)
        else:
            # If not Namespace or dict, treat as AcqOptimizer instance
            self.acqoptimizer = acqoptimizer

    def set_verbose(self, verbose):
        """Set verbose options."""
        self.verbose = verbose
        if self.verbose:
            self.print_str()

    def get(self):
        """Perform acquisition optimization and return optima."""

        # Setup acqoptimizer
        self.acqoptimizer.setup_optimize()

        # Multiprocessing
        mp_manager = multiprocessing.Manager()
        mp_return_dict = mp_manager.dict()

        subseed = np.random.randint(13337)
        proc = multiprocessing.Process(
            target=self.mp_acq_optimize,
            args=(
                self.data,
                self.model,
                self.acqfunction,
                self.acqoptimizer,
                mp_return_dict,
                subseed,
            ),
        )
        proc.start()
        proc.join()
        acq_optima = mp_return_dict['acq_optima']

        return acq_optima

    def mp_acq_optimize(
        self, data, model, acqfunction, acqoptimizer, return_dict, seed=None
    ):
        """Acquisition optimization for use in multiprocessing."""
        # Set random seed
        if seed is not None:
            np.random.seed(seed)

        # Setup acqfunction
        acqfunction.setup_function(data, model, acqoptimizer)

        # Optimize acqfunction
        acq_optima = acqoptimizer.optimize(acqfunction, data)

        # Add acq_optima to return_dict
        return_dict['acq_optima'] = acq_optima

    def print_str(self):
        """Print a description string."""
        print('*Tuun with params={}'.format(self.params))
