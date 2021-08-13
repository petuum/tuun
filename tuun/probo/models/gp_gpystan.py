"""
Classes for GP models with GPyStan.
"""

from argparse import Namespace
import copy
import numpy as np

from .gpystan.gp_stan import StanGp
from ..util.misc_util import dict_to_namespace, suppress_stdout_stderr


class GpystanGp:
    """
    Hierarchical GPs, implemented with GPyStan.
    """

    def __init__(self, params=None, verbose=True):
        """
        Parameters
        ----------
        params : Namespace_or_dict
            Namespace or dict of parameters for this model.
        verbose : bool
            If True, print description string.
        """
        self.set_params(params)
        self.set_model()
        self.set_verbose(verbose)

    def set_params(self, params):
        """Set self.params, the parameters for this model."""
        params = dict_to_namespace(params)

        # Set self.params
        self.params = Namespace()
        self.params.ndimx = params.ndimx
        self.params.model_str = getattr(params, 'model_str', 'optfixedsig')
        self.params.kernel_str = getattr(params, 'kernel_str', 'rbf')
        self.params.ig1 = getattr(params, 'ig1', 4.0)
        self.params.ig2 = getattr(params, 'ig2', 3.0)
        self.params.n1 = getattr(params, 'n1', 1.0)
        self.params.n2 = getattr(params, 'n2', 1.0)
        self.params.sigma = getattr(params, 'sigma', 1e-5)
        self.params.niter = getattr(params, 'niter', 70)
        self.params.trans_x = getattr(params, 'trans_x', False)
        self.params.trans_y = getattr(params, 'trans_x', True)
        self.params.print_inf = getattr(params, 'print_inf', False)

    def set_model(self):
        """Set self.model."""
        self.model = StanGp(self.params, verbose=False)

    def inf(self, data):
        """Set data, run inference, update self.sample_list."""
        self.model.infer_hypers(data)

    def postgen_list(self, x_list, s, nsamp):
        """
        Draw nsamp samples from posterior predictive distribution, given list
        of inputs x_list and seed s.

        Parameters
        ----------
        x_list : list
            List of numpy ndarrays each with shape=(self.params.ndimx,).
        s : int
            The seed, a positive integer.
        nsamp : int
            The number of samples to draw from the posterior predictive
            distribution.

        Returns
        -------
        list
            A list with len=len(x_list) of numpy ndarrays, each with
            shape=(nsamp,).
        """
        if self.model.params.trans_x:
            x_list = self.model.transform_xin_list(x_list)

        pred_list = self.model.sample_post_pred(
            nsamp, x_list, full_cov=True, nloop=np.min([50, nsamp])
        )

        if self.model.params.trans_y:
            pred_list = [self.model.dt.inv_transform_y_data(pr) for pr in pred_list]

        return pred_list

    def set_verbose(self, verbose):
        """Set verbose options."""
        self.verbose = verbose
        if self.verbose:
            self.print_str()

    def print_str(self):
        """Print a description string."""
        print('*[INFO] ' + str(self))

    def __str__(self):
        return f'GpystanGp with params={self.params}'
