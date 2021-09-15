"""
Classes for product domains (Cartesian products of domains).
"""

from argparse import Namespace
import numpy as np

from .real import RealDomain
from .integral import IntegralDomain
from ..util.misc_util import dict_to_namespace


class ProductDomain:
    """
    Class for domains defined as a Cartesian product of other domains.
    """

    def __init__(self, params=None, domain_list=None, verbose=True):
        """
        Parameters
        ----------
        params : Namespace_or_dict
            Namespace or dict of parameters.
        domain_list : list
            List of instantiated Domain objects
        verbose : bool
            If True, print description string.
        """
        self.set_params(params)
        self.set_domain_list(domain_list)
        self.set_verbose(verbose)

    def set_params(self, params):
        """Set parameters for the ProductDomain."""
        params = dict_to_namespace(params)

        self.params = Namespace()
        self.params.dom_str = getattr(params, 'dom_str', 'product')

    def set_domain_list(self, domain_list):
        """Set self.domain_list."""

        # NOTE: For now, assume domain_list is a list of instantiated Domains.
        # TODO: if domain_list is a list of Namespaces/dicts with dom_str
        # field, instantiate Domains.

        # Default domain list is [RealDomain, IntegralDomain]
        if domain_list is None:
            domain_list = [RealDomain(verbose=False), IntegralDomain(verbose=False)]

        self.domain_list = domain_list

    def get_domain_list(self):
        """Return self.domain_list."""
        return self.domain_list

    def set_verbose(self, verbose):
        """Set verbose options."""
        self.verbose = verbose
        if self.verbose:
            self.print_str()

    def is_in_domain(self, pt):
        """Check if pt is in domain, and return True or False."""

        bool_list = [
            domain.is_in_domain(block) for block, domain in zip(pt, self.domain_list)
        ]
        ret = False if False in bool_list else True
        return ret

    def unif_rand_sample(self, n=1):
        """Draws a sample uniformly at random from domain."""
        sample_list = [dom.unif_rand_sample(n) for dom in self.domain_list]
        ret = [list(x) for x in zip(*sample_list)]
        return ret

    def print_str(self):
        """Print a description string."""
        print('*ProductDomain with domain_list:')
        for idx, domain in enumerate(self.domain_list):
            # print('   Sub-domain {}: '.format(idx+1), end='')
            print('   {}: '.format(idx), end='')
            domain.print_str()
