"""
An acqoptimizer for a product domain.
"""

from argparse import Namespace
import copy
import numpy as np

from ..domain import ProductDomain
from ..util.misc_util import dict_to_namespace


class ProductAcqOptimizer:
    """AcqOptimizer for ProductDomain."""

    def __init__(
        self, acqoptimizer_list, params=None, print_delta=False, verbose=True,
    ):
        """
        Parameters
        ----------
        acqoptimizer_list : list
            List of other AcqOptimizers
        params : Namespace_or_dict
            Namespace or dict of parameters.
        print_delta : bool
            If True, print acquisition function deltas at each iteration.
        verbose : bool
            If True, print description string.
        """
        self.ao_list = acqoptimizer_list
        self.set_params(params)
        self.set_product_domain()
        self.params.print_delta = print_delta
        self.set_verbose(verbose)

    def set_params(self, params):
        """Set self.params."""
        params = dict_to_namespace(params)

        self.params = params
        self.params.n_iter_bcd = getattr(params, 'n_iter_bcd', 3)
        self.params.rand_every = getattr(params, 'rand_every', None)
        self.params.rand_block_init = getattr(params, 'rand_block_init', False)
        self.params.n_init_rs = getattr(params, 'n_init_rs', 0)

    def set_product_domain(self):
        """Set self.product_domain."""
        domain_list = [ao.domain for ao in self.ao_list]
        self.product_domain = ProductDomain(domain_list=domain_list, verbose=False)

    def set_verbose(self, verbose):
        """Set verbose options."""
        self.verbose = verbose
        if self.verbose:
            self.print_str()

    def setup_optimize(self):
        # Run setup_optimize() for each AcqOptimizer in self.ao_list
        for ao in self.ao_list:
            ao.setup_optimize()

        self.xin_is_list = True

    def optimize(self, acqmap, data):

        # If there is no data, return a random sample from domain
        if data is None or not list(data.x):
            return self.product_domain.unif_rand_sample(1)[0]

        # Optionally initialize with random search
        if len(data.x) <= self.params.n_init_rs:
            return self.product_domain.unif_rand_sample(1)[0]

        # NOTE:
        # - Below I assume that the input xin to acqmap (and to the model) is a list,
        #   where each element is associated with one sub-domain (this restricts how
        #   model must be defined).

        domain_list = self.product_domain.get_domain_list()

        # Initialize nextpt
        nextpt = self.init_and_get_nextpt(self.product_domain, data)

        # Store initial point (used for printing acquisition function delta)
        initpt = copy.deepcopy(nextpt)

        for _ in range(self.params.n_iter_bcd):
            for j in range(len(domain_list)):

                # Select jth domain and acqoptimizer
                domain = domain_list[j]
                ao = self.ao_list[j]

                # Define function that returns a modified nextpt
                get_nextpt_mod = lambda x: self.list_replace_idx(nextpt, x, j)

                # Construct am (for ao) from acqmap, for list & non-list cases
                xin_is_list = getattr(ao, 'xin_is_list', False)

                if xin_is_list:
                    # NOTE:
                    # - xin_list is list of domain-pts for sub-domain.
                    # - acqmap is for full domain (and takes list of xin)
                    # - am is for sub-domain (and also takes list of xin)
                    am = lambda xin_list: acqmap(
                        [get_nextpt_mod(xin) for xin in xin_list]
                    )
                else:
                    # NOTE:
                    # - xin is a domain-pt for sub-domain.
                    # - acqmap is for full domain (and takes list of xin)
                    # - am is for sub-domain (and takes single xin)
                    am = lambda xin: acqmap([get_nextpt_mod(xin)])[0]

                # Convert data into correct form
                data_j = copy.deepcopy(data)
                data_j.x = [x[j] for x in data_j.x]

                # init_opt strategy
                data_j.init_opt = nextpt[j]

                # Checkpoint current nextpt
                nextpt_ckpt = copy.deepcopy(nextpt)

                # Update nextpt with ao.optimize
                nextpt[j] = ao.optimize(am, data_j)

                #if self.params.print_delta:
                    #acq_delta = acqmap([nextpt])[0] - acqmap([nextpt_ckpt])[0]
                    #print(('  Acq delta: {:.7f} = (final acq - init acq) ' +
                           #'[block ckpt]').format(acq_delta))

        if self.params.print_delta:
            self.print_acq_delta(acqmap, initpt, nextpt)

        return nextpt

    def list_replace_idx(self, alist, newitem, idx):
        """Replace alist[idx] with newitem."""
        newlist = copy.deepcopy(alist)
        newlist[idx] = newitem
        return newlist

    def print_acq_delta(self, acqmap, init_point, optima):
        """Print acquisition function delta for optima minus initial point."""
        init_acq = acqmap([init_point])[0]
        final_acq = acqmap([optima])[0]
        acq_delta = final_acq - init_acq
        print(
            ('  Acq delta: {:.7f} = (final acq - init acq) ' + '[product]').format(
                acq_delta
            )
        )

    def init_and_get_nextpt(self, product_domain, data):
        """Initialize and return nextpt for optimize."""

        if len(data.x) < 1:
            nextpt = product_domain.unif_rand_sample()[0]
        else:
            if self.params.rand_every is None:
                self.params.rand_every = len(data.x) + 1

            if (
                self.params.rand_block_init
                and len(data.x) % self.params.rand_every == 0
            ):
                # Randomize initialize only one block of nextpt (other blocks
                # set to best so far)

                min_idx = np.argmin(data.y)
                nextpt = data.x[min_idx]

                nextpt_rand = product_domain.unif_rand_sample()[0]
                rand_j_idx = np.random.randint(len(nextpt))

                nextpt[rand_j_idx] = nextpt_rand[rand_j_idx]
                if self.params.print_delta:
                    print('  RAND-BLOCK init for BCD')

            elif len(data.x) % self.params.rand_every == 0:
                # Randomly initialize full nextpt
                nextpt = product_domain.unif_rand_sample()[0]
                if self.params.print_delta:
                    print('  RAND init for BCD')

            else:
                # Initialize nextpt to best so far
                min_idx = np.argmin(data.y)
                nextpt = data.x[min_idx]
                if self.params.print_delta:
                    print('  BSF init for BCD')

        return nextpt

    def print_str(self):
        """Print a description string."""
        print(
            ('*ProductAcqOptimizer with params = {} ' + 'and ao_list:').format(
                self.params
            )
        )
        for idx, ao in enumerate(self.ao_list):
            print('   {}: {}'.format(idx, ao))
