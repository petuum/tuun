"""
Classes for list (discrete set) domains.
"""

from argparse import Namespace
import numpy as np


class ListDomain(object):
    """
    Class for defining sets defined by a list of elements.
    """

    def __init__(self, params=None, verbose=True):
        """
        Parameters
        ----------
        params : Namespace_or_dict
            Namespace or dict of parameters.
        verbose : bool
            If True, print description string.
        """
        self.set_params(params)
        self.init_domain_list()
        self.set_verbose(verbose)

    def set_params(self, params):
        """Set parameters for the ListDomain."""

        # If params is a dict, convert to Namespace
        if isinstance(params, dict):
            params = Namespace(**params)

        self.params = Namespace()
        self.params.set_domain_list_auto = getattr(
            params, 'set_domain_list_auto', False
        )
        self.params.domain_list_exec_str = getattr(params, 'domain_list_exec_str', '')
        self.params.set_domain_list = getattr(params, 'set_domain_list', False)
        self.params.domain_list = getattr(params, 'domain_list', [])

    def set_verbose(self, verbose):
        """Set verbose options."""
        self.verbose = verbose
        if self.verbose:
            self.print_str()

    def init_domain_list(self):
        """Initialize self.domain_list."""
        if getattr(self.params, 'set_domain_list_auto', False):
            self.set_domain_list_auto()
        elif getattr(self.params, 'set_domain_list', False):
            self.set_domain_list(self.params.domain_list)
        else:
            self.domain_list = None

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

    def print_str(self):
        """Print a description string."""
        print('*ListDomain with params = {}'.format(self.params))
