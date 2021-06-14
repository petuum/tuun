"""
Classes for GP models with Stan, using distance matrices to define the kernel. 
"""

from argparse import Namespace
import numpy as np
import copy

from .gp_stan import StanGp
from .stan.gp_distmat import get_model as get_model_gp
from .stan.gp_distmat_fixedsig import get_model as get_model_gp_fixedsig
from .gp.gp_utils import kern_distmat, squared_euc_distmat, simple_list_distmat
from ..util.misc_util import dict_to_namespace


class StanDistmatGp(StanGp):
    """
    Hierarchical GP model, using distance matrices to define the kernel, implemented
    with Stan.
    """

    def set_params(self, params):
        """Set self.params, the parameters for this model."""
        params = dict_to_namespace(params)

        # Set self.params
        self.params = Namespace()
        self.params.model_str = getattr(params, 'model_str', 'optfixedsig')
        self.params.ig1 = getattr(params, 'ig1', 4.0)
        self.params.ig2 = getattr(params, 'ig2', 3.0)
        self.params.n1 = getattr(params, 'n1', 1.0)
        self.params.n2 = getattr(params, 'n2', 1.0)
        self.params.sigma = getattr(params, 'sigma', 1e-5)
        self.params.niter = getattr(params, 'niter', 70)
        self.params.trans_x = getattr(params, 'trans_x', False)
        self.params.print_warnings = getattr(params, 'print_warnings', False)
        self.params.domain_spec = getattr(params, 'domain_spec', 'real')
        distmat_function = getattr(
            params,
            'distmat_function',
            self.get_default_distmat_function(self.params.domain_spec),
        )
        self.params.distmat_function = distmat_function
        self.params.kernel = self.get_distmat_kernel(distmat_function)

    def get_default_distmat_function(self, domain_spec):
        """Return a default distmat_function."""

        # Treat list of one type as single (non-product) domain
        if type(domain_spec) is list:
            if len(domain_spec) == 1:
                domain_spec = domain_spec[0]

        # Single (non-product) domain
        if type(domain_spec) is not list:
            return self.get_default_distmat_function_single(domain_spec)

        # Product domain
        elif type(domain_spec) is list:
            distmat_function_list = []
            for i, domain_spec_single in enumerate(domain_spec):
                distfn = self.get_default_distmat_function_single(domain_spec_single)
                distfn = self.convert_distmat_function_for_sum(distfn, i)
                distmat_function_list.append(distfn)

            def default_distmat_function(a, b):
                return sum([distfn(a, b) for distfn in distmat_function_list])

            return default_distmat_function

    def get_default_distmat_function_single(self, domain_spec_single):
        """Return a default distmat_function for single (non-product) domains only."""
        assert domain_spec_single in ['real', 'list']

        if domain_spec_single is 'real':
            return lambda a, b: squared_euc_distmat(a, b, 1.0)
        elif domain_spec_single is 'list':
            return lambda a, b: simple_list_distmat(a, b, 0.1, additive=True)

    def convert_distmat_function_for_sum(self, distmat_function, i):
        """
        Convert a distmat_function to correct format for component i of sum for final
        distmat_function.
        """

        def new_distmat_function(xlist1, xlist2):
            """Modified distmat_function to operate on sublists."""
            new_xlist1 = [x[i] for x in xlist1]
            new_xlist2 = [x[i] for x in xlist2]
            return distmat_function(new_xlist1, new_xlist2)

        return new_distmat_function

    def get_distmat_kernel(self, distmat_function):
        """Return kernel for a given distmat_function"""
        return lambda a, b, c, d: kern_distmat(a, b, c, d, distmat_function)

    def get_model(self):
        """Returns GP stan model"""
        if (
            self.params.model_str == 'optfixedsig'
            or self.params.model_str == 'sampfixedsig'
        ):
            return get_model_gp_fixedsig(print_status=self.verbose)
        elif self.params.model_str == 'opt' or self.params.model_str == 'samp':
            return get_model_gp(print_status=self.verbose)
        elif self.params.model_str == 'fixedparam':
            return None

    def transform_xin_list(self, xin_list):
        """Transform list of xin (e.g. in data.x)."""
        # Ensure data.x is correct format (list of 1D numpy arrays)
        # xin_list = [np.array(xin).reshape(-1) for xin in xin_list]

        if self.params.trans_x:
            # apply transformation to xin_list
            xin_list_trans = xin_list  # TODO: define default transformation
        else:
            xin_list_trans = xin_list

        return xin_list_trans

    def get_stan_data_dict(self):
        """Return data dict for stan sampling method"""
        if (
            self.params.model_str == 'optfixedsig'
            or self.params.model_str == 'sampfixedsig'
        ):
            return {
                'ig1': self.params.ig1,
                'ig2': self.params.ig2,
                'n1': self.params.n1,
                'n2': self.params.n2,
                'sigma': self.params.sigma,
                'N': len(self.data.x),
                'y': np.array(self.data.y).reshape(-1),
                'distmat': self.params.distmat_function(self.data.x, self.data.x),
            }
        elif self.params.model_str == 'opt' or self.params.model_str == 'samp':
            return {
                'ig1': self.params.ig1,
                'ig2': self.params.ig2,
                'n1': self.params.n1,
                'n2': self.params.n2,
                'n3': self.params.n3,
                'n4': self.params.n4,
                'N': len(self.data.x),
                'y': np.array(self.data.y).reshape(-1),
                'distmat': self.params.distmat_function(self.data.x, self.data.x),
            }

    def __str__(self):
        return f'StanDistmatGp with params={self.params}'
