"""
Classes for Bayesian optimization (BO) routines.
"""

from argparse import Namespace
import copy
import numpy as np

from ..util.misc_util import dict_to_namespace
from ..design.model_based import AcqOptDesigner


np.set_printoptions(precision=3, suppress=True)


class SimpleBo:
    """
    Simple Bayesian optimization on a single machine.
    """

    def __init__(
        self,
        f,
        model=None,
        acqfunction=None,
        acqoptimizer=None,
        data=None,
        params=None,
        data_update_fun=None,
        verbose=True,
        seed=None,
    ):
        """
        Parameters
        ----------
        f : function
            Function to optimize.
        model : Model
            Model instance. Default is None, which uses a random acquisition function.
        acqfunction : AcqFunction_or_Namespace_or_dict
            AcqFunction instance or Namespace/dict of parameters. Default is EI.
        acqoptimizer : AcqOptimizer_or_Namespace_or_dict
            AcqOptimizer instance or Namespace/dict of parameters. Default is None,
            which uses random search.
        data : Namespace_or_dict
            Namespace or dict of initial data. Contains fields x and y (lists).
        data_update_fun : function
            Function that will update self.data given a tuple (x, y)
        params : Namespace_or_dict
            Namespace or dict of parameters.
        verbose : bool
            If True, print description string.
        seed : int
            If not None, set numpy random seed to seed.
        """
        self.set_random_seed(seed)
        self.set_params(params)
        self.set_designer(model, acqfunction, acqoptimizer, data)
        self.f = f
        self.set_data_update_fun(data_update_fun)
        self.set_data()
        self.set_verbose(verbose)

    def set_random_seed(self, seed):
        """Set numpy random seed."""
        if seed is not None and seed != -1:
            self.seed = seed
            np.random.seed(self.seed)
        else:
            self.seed = -1

    def set_params(self, params):
        """Set parameters of SimpleBo."""
        params = dict_to_namespace(params)

        # Set defaults
        self.params = Namespace()
        self.params.n_iter = getattr(params, 'n_iter', 10)
        self.params.reinit_designer = getattr(params, 'reinit_designer', False)
        self.params.seed = self.seed

    def set_designer(self, model, acqfunction, acqoptimizer, data):
        """Set self.designer."""
        self.designer = AcqOptDesigner(
            model, acqfunction, acqoptimizer, data, seed=self.params.seed
        )

    def set_data_update_fun(self, data_update_fun):
        """Set self.data_update_fun."""
        if data_update_fun is None:

            def _default_data_update_fun(x, y, data):
                data.x.append(x)
                data.y.append(y)

            data_update_fun = _default_data_update_fun

        self.data_update_fun = data_update_fun

    def set_data(self):
        """Set self.data using self.designer.data."""
        self.data_init = copy.deepcopy(self.designer.data)
        self.data = copy.deepcopy(self.data_init)

        if self.data_init is None:
            self.n_data_init = 0
        else:
            self.n_data_init = len(self.data_init.x)

    def set_verbose(self, verbose):
        """Set verbose options."""
        self.verbose = verbose
        if self.verbose:
            self.print_str()
            print('*Initialized with {} observations'.format(self.n_data_init))

    def run(self):
        """Run Bayesian optimization."""

        # The BO loop
        for i in range(self.params.n_iter):

            # Set designer
            designer = self.get_designer_for_iter()

            # Choose next x and query f(x)
            x = designer.get()
            y = self.f(x)
            y = self.format_function_output(y)

            # Update data and reset data in designer
            self.data_update_fun(x, y, self.data)
            designer.set_data(self.data)

            # Print iter info
            self.print_iter_info(i)

        self.print_final_info()
        results = self.get_final_results()
        return results

    def get_designer_for_iter(self):
        """Return designer for an iteration of BO."""
        if self.params.reinit_designer:
            subseed = np.random.randint(13337)
            designer = AcqOptDesigner(
                self.designer.model,
                self.designer.acqfunction,
                self.designer.acqoptimizer,
                self.data,
                seed=subseed,
                verbose=False
            )
        else:
            designer = self.designer

        return designer

    def format_function_output(self, y_out):
        """Format output of function query."""

        # For now, function output y_out should be a float
        y_out = float(y_out)
        return y_out

    def print_iter_info(self, iter_idx):
        """Print information for a given iteration of Bayesian optimization."""
        x_str_max_len = 14
        x_str = str(self.data.x[-1])
        x_str = x_str[:min(len(x_str), x_str_max_len)]
        y = self.data.y[-1]
        bsf = np.min(self.data.y)
        print('i: {},    x: {},\ty: {:.4f},\tBSF: {:.4f}'.format(
            iter_idx, x_str, y, bsf
        ))

    def print_final_info(self):
        """Print final information after Bayesian optimization is complete."""
        min_idx = np.argmin(self.data.y)
        min_x = self.data.x[min_idx]
        min_y = self.data.y[min_idx]
        print('Minimum y = {}'.format(min_y))
        print('Minimizer x = {}'.format(min_x))
        print('Found at i = {}'.format(min_idx - self.n_data_init))

    def get_final_results(self):
        """Return final results of a run of Bayesian optimization."""
        results = Namespace()
        results.min_idx = np.argmin(self.data.y)
        results.min_x = self.data.x[results.min_idx]
        results.min_y = self.data.y[results.min_idx]
        results.data = self.data
        return results

    def print_str(self):
        """Print a description string."""
        print('*SimpleBo with params={}'.format(self.params))
