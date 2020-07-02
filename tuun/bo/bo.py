"""
Classes for Bayesian optimization (BO) routines.
"""

from argparse import Namespace
import copy
import numpy as np

from ..util.misc_util import dict_to_namespace


np.set_printoptions(precision=3, suppress=True)


class SimpleBo:
    """
    Simple Bayesian optimization on a single machine.
    """

    def __init__(self, tuun, f, data_update_fun=None, params=None, verbose=True):
        """
        Parameters
        ----------
        tuun : Tuun
            Tuun instance.
        f : function
            Function to optimize.
        data_update_fun : function
            Function that will update self.data given a tuple (x, y)
        params : Namespace_or_dict
            Namespace or dict of parameters.
        verbose : bool
            If True, print description string.
        """
        self.set_params(params)
        self.tuun = tuun
        self.f = f
        self.set_data_update_fun(data_update_fun)
        self.set_data()
        self.set_verbose(verbose)

    def set_params(self, params):
        """Set parameters of SimpleBo."""
        params = dict_to_namespace(params)

        # Set defaults
        self.params = Namespace()
        self.params.n_iter = getattr(params, 'n_iter', 10)

    def set_data_update_fun(self, data_update_fun):
        """Set self.data_update_fun."""
        if data_update_fun is None:

            def default_data_update_fun(x, y, data):
                data.x.append(x)
                data.y.append(y)

            data_update_fun = default_data_update_fun

        self.data_update_fun = data_update_fun

    def set_data(self):
        """Set self.data using self.tuun.data."""
        self.data_init = copy.deepcopy(self.tuun.data)
        self.data = copy.deepcopy(self.data_init)

    def set_verbose(self, verbose):
        """Set verbose options."""
        self.verbose = verbose
        if self.verbose:
            self.print_str()

    def run(self):
        """Run Bayesian optimization."""

        # The BO loop
        for i in range(self.params.n_iter):

            # Choose next x and query f(x)
            x = self.tuun.get()
            y = self.f(x)

            # Update data and reset data in tu
            self.data_update_fun(x, y, self.data)
            self.tuun.set_data(self.data)

            # Print iter info
            self.print_iter_info(i)

        self.print_final_info()
        results = self.get_final_results()
        return results

    def print_iter_info(self, iter_idx):
        """Print information for a given iteration of Bayesian optimization."""
        x = self.data.x[-1]
        y = self.data.y[-1].item()
        bsf = np.min(self.data.y)
        print('i: {},    x: {},\ty: {:.4f},\tBSF: {:.4f}'.format(iter_idx, x, y, bsf))

    def print_final_info(self):
        """Print final information after Bayesian optimization is complete."""
        min_idx = np.argmin(self.data.y)
        min_x = self.data.x[min_idx]
        min_y = self.data.y[min_idx]
        print('Minimum x = {}'.format(min_x))
        print('Minimum y = {}'.format(min_y))
        print('Found at i = {}'.format(min_idx))

    def get_final_results(self):
        """Return final results of a run of Bayesian optimization."""
        results = Namespace()
        results.min_idx = np.argmin(self.data.y)
        results.min_x = self.data.x[results.min_idx]
        results.min_y = self.data.y[results.min_idx]
        return results

    def print_str(self):
        """Print a description string."""
        print('*SimpleBo with params={}'.format(self.params))
