"""
Classes to perform acquisition function optimization.
"""

from argparse import Namespace
import copy
import numpy as np

from ..util.misc_util import dict_to_namespace
from ..domain import get_domain_from_config


def get_acqoptimizer_from_config(ao_params, verbose=False):
    """
    Return an AcqOptimizer instance given ao_params config.

    Parameters
    ----------
    ao_params : Namespace_or_dict
        Namespace or dict of parameters for the acqoptimizer.
    verbose : bool
        If True, the acqoptimizer will print description string.

    Returns
    -------
    AcqOptimizer
        An AcqOptimizer instance.
    """
    ao_params = dict_to_namespace(ao_params)
    return AcqOptimizer(params=ao_params, verbose=verbose)


class AcqOptimizer:
    """
    Class to perform acquisition function optimization.
    """

    def __init__(self, params=None, domain=None, verbose=True):
        """
        Parameters
        ----------
        params : Namespace_or_dict
            Namespace or dict of parameters.
        domain : Domain_or_Namespace_or_dict
            Domain instance, or Namespace/dict of parameters that specify one
            of the predefined Domains.
        verbose : bool
            If True, print description string.
        """
        self.set_params(params)
        self.set_domain(domain)
        self.set_verbose(verbose)

    def set_params(self, params):
        """Set parameters of AcqOptimizer."""
        params = dict_to_namespace(params)

        # Set defaults
        self.params = Namespace()
        self.params.opt_str = getattr(params, 'opt_str', 'rand')
        self.params.max_iter = getattr(params, 'max_iter', 100)
        self.params.n_opt = getattr(params, 'n_opt', 1)

    def set_domain(self, domain):
        """
        Set self.domain, the search space for acquisition optimization.
        
        Parameters
        ----------
        domain: Domain_or_Namespace_or_dict
            Domain instance, or Namespace/dict of parameters that specify one
            of the predefined Domains.
        """
        domain = dict_to_namespace(domain)

        if domain is None:
            # Set domain defaults based on data

            # TODO: base next line off of the data, instead of just setting to
            # a 1d domain with arbitrary range
            domain_params = Namespace(dom_str='real', min_max=[(-10, 10)])

            self.domain = get_domain_from_config(domain_params)
        elif isinstance(domain, Namespace):
            # Set defaults for any missing params
            domain_params = Namespace()
            domain_params.dom_str = getattr(domain, 'dom_str', 'real')
            domain_params.min_max = getattr(domain, 'min_max', [(-10, 10)])
            self.domain = get_domain_from_config(domain_params)
        else:
            # If not Namespace or dict, treat as Domain instance
            self.domain = domain

    def set_verbose(self, verbose):
        """Set verbose options."""
        self.verbose = verbose
        if self.verbose:
            self.print_str()

    def setup_optimize(self):
        """Setup for self.optimize method"""
        # Set self.xin_is_list
        batch_opt_str_tup = ('rand', 'mutate')
        opt_str = getattr(self.params, 'opt_str', '')
        self.xin_is_list = True if opt_str in batch_opt_str_tup else False

        # Set self.params.n_opt if not already in self.params
        if not hasattr(self.params, 'n_opt'):
            self.params.n_opt = 1

    def optimize(self, acqfunction, data=None):
        """Optimize acqfunction over x in domain"""
        # If there is no data, return a random sample from domain
        if data is None or data.x == []:
            return self.domain.unif_rand_sample(1)[0]

        if self.params.opt_str == 'rand':
            return self.optimize_rand(self.domain, acqfunction)
        elif self.params.opt_str == 'mutate':
            return self.optimize_mutate(self.domain, acqfunction, data)
        elif self.params.opt_str == 'rand_seq':
            return self.optimize_rand_seq(self.domain, acqfunction)
        elif self.params.opt_str == 'ea':
            return self.optimize_ea(self.domain, acqfunction, data)

    def optimize_rand(self, dom, acqmap):
        """Optimize acqmap(x) over domain via random search"""
        xin_list = dom.unif_rand_sample(self.params.max_iter)
        return self.get_min_xin_list(acqmap, xin_list, self.params.n_opt)

    def optimize_mutate(self, dom, acqmap, data):
        """Optimize acqmap(x) over domain via data mutation"""
        xin_list = dom.mutate_data(data, self.params.max_iter)
        return self.get_min_xin_list(acqmap, xin_list, self.params.n_opt)

    def get_min_xin_list(self, acqmap, xin_list, n_opt=1):
        """
        Return index (or indicies if n_opt>1) of min over xin list of the
        acqmap.
        """
        sort_idx_list = list(np.argsort(acqmap(xin_list)))
        return self.get_n_xin_list_from_sort_idx_list(xin_list, sort_idx_list)

    def get_n_xin_list_from_sort_idx_list(self, xin_list, sort_idx_list):
        """
        Return n_opt (list or single element) of xin_list based on
        sort_idx_list of sorted indices.
        """
        if self.params.n_opt > 1:
            trunc_sort_idx_list = sort_idx_list[: self.params.n_opt]
            return [xin_list[m] for m in trunc_sort_idx_list]
        else:
            return xin_list[sort_idx_list[0]]

    def optimize_rand_seq(self, dom, acqmap):
        """Optimize acqmap(x) over domain via sequential random search"""
        acq_list = []
        xin_list = []
        for i in range(self.params.max_iter):
            xin = dom.unif_rand_sample(1)[0]
            acq_list.append(acqmap(xin))
            xin_list.append(xin)
        sort_idx_list = list(np.argsort(acq_list))
        return self.get_n_xin_list_from_sort_idx_list(xin_list, sort_idx_list)

    def optimize_mutate(self, dom, acqmap, data):
        """Optimize acqmap(x) over domain via data mutation"""
        xin_list = dom.mutate_data(data, self.params.max_iter)
        return self.get_min_xin_list(acqmap, xin_list, self.params.n_opt)

    def optimize_ea(self, dom, acqmap, data):
        """Optimize acqmap(x) over domain via sequential data mutation"""
        n_rounds = getattr(self.params, 'n_rounds', 5)
        nmut = self.params.max_iter // n_rounds
        new_data = Namespace(X=list(copy.deepcopy(data.X)), y=copy.deepcopy(data.y))

        for r in range(n_rounds):
            # Choose subset of new_data of size nmut
            xidx = list(np.argsort(new_data.y))
            xidx = xidx[:nmut]
            new_data = Namespace(
                X=[new_data.X[i] for i in xidx], y=new_data.y[np.array(xidx)]
            )

            # Mutate subset of data
            mut_list = dom.mutate_data(new_data, nmut)

            # Re-make new_data namespace
            new_data = Namespace(X=mut_list, y=np.array([acqmap(x) for x in mut_list]))

        sort_idx_list = list(np.argsort(new_data.y))
        return self.get_n_xin_list_from_sort_idx_list(new_data.X, sort_idx_list)

    def print_str(self):
        """Print a description string."""
        print('*AcqOptimizer with params={}'.format(self.params))
