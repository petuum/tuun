"""
Classes for list (discrete set) domains.
"""

from argparse import Namespace
import copy
import numpy as np

from ..util.misc_util import dict_to_namespace


class ListDomain:
    """Class for defining discrete sets containing a list of elements."""

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
        self._init_domain_list()
        self._set_verbose(verbose)

    def _set_params(self, params):
        """Set parameters for the ListDomain."""
        params = dict_to_namespace(params)

        self.params = Namespace()
        self.params.set_domain_list_auto = getattr(
            params, 'set_domain_list_auto', False
        )
        self.params.domain_list_exec_str = getattr(params, 'domain_list_exec_str', '')
        self.params.domain_list = getattr(params, 'domain_list', [])
        self.params.dom_str = getattr(params, 'dom_str', 'list')

    def _set_verbose(self, verbose):
        """Set verbose options."""
        self.verbose = verbose
        if self.verbose:
            self._print_str()

    def _init_domain_list(self):
        """Initialize self.domain_list."""
        if self.params.set_domain_list_auto:
            self.set_domain_list_auto()
        else:
            self.set_domain_list(self.params.domain_list)

    def set_domain_list_auto(self):
        """
        Set self.domain_list automatically via self.params.domain_list_exec_str.
        """
        exec(self.params.domain_list_exec_str)

    def set_domain_list(self, domain_list):
        """Set self.domain_list, containing elements of domain."""
        self.domain_list = domain_list

    def is_in_domain(self, pt):
        """Check if pt is in domain, and return True or False."""
        return pt in self.domain_list

    def unif_rand_sample(self, n=1, replace=True):
        """
        Draws a sample uniformly at random from domain, returns as a list of
        len n, with (default) or without replacement.
        """
        if replace:
            randind = np.random.randint(len(self.domain_list), size=n)
        else:
            randind = np.arange(min(n, len(self.domain_list)))
        return [self.domain_list[i] for i in randind]

    def set_print_params(self):
        """Set self.print_params."""
        if not hasattr(self, 'print_params'):
            self.print_params = copy.deepcopy(self.params)
            delattr(self.print_params, "domain_list")

    def _print_str(self):
        """Print a description string."""
        print('*[INFO] ' + str(self))

    def __str__(self):
        self.set_print_params()
        return f'ListDomain with params={self.print_params}'
