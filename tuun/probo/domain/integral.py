"""
Classes for integral domains.
"""

from argparse import Namespace
import numpy as np

from ..util.misc_util import dict_to_namespace


class IntegralDomain:
    """Class for defining bounded sets of integers."""

    def __init__(self, params=None, verbose=True):
        """
        Parameters
        ----------
        params : Namespace_or_dict
            Namespace or dict of parameters.
        verbose : bool
            If True, print description string.
        """
        self._set_params(params)
        self._set_verbose(verbose)

    def _set_params(self, params):
        """Set parameters for the IntegralDomain."""
        params = dict_to_namespace(params)

        self.params = Namespace()
        self.params.min_max = getattr(params, 'min_max', [(0, 10)])
        self.params.dom_str = getattr(params, 'dom_str', 'integral')

        self.ndimx = len(self.params.min_max)

    def _set_verbose(self, verbose):
        """Set verbose options."""
        self.verbose = verbose
        if self.verbose:
            self._print_str()

    def is_in_domain(self, pt):
        """Check if pt is in domain, and return True or False."""
        pt = np.array(pt).reshape(-1)

        if pt.shape[0] != self.ndimx:
            ret = False
        else:
            int_bools = [float(c).is_integer() for c in pt]
            in_range_bools = [
                pt[i] >= self.params.min_max[i][0]
                and pt[i] <= self.params.min_max[i][1]
                for i in range(self.ndimx)
            ]

            ret = False if False in int_bools + in_range_bools else True

        return ret

    def unif_rand_sample(self, n=1):
        """Draws a sample uniformly at random from domain."""
        list_of_arr_per_dim = [
            np.random.randint(mm[0], mm[1] + 1, n) for mm in self.params.min_max
        ]
        list_of_list_per_sample = [list(l) for l in np.array(list_of_arr_per_dim).T]

        return list_of_list_per_sample

    def _print_str(self):
        """Print a description string."""
        print('*IntegralDomain with params = {}'.format(self.params))
