"""
Classes for GP models with Stan that make use of a prior mean function.
"""

from argparse import Namespace
import numpy as np
import copy

from .gp_stan import StanGp
from ..util.misc_util import dict_to_namespace


class StanPriorMeanGp(StanGp):
    """
    Hierarchical GP model, implemented with Stan, which uses a prior mean function
    passed as an argument.
    """

    def set_params(self, params):
        """Set self.params, the parameters for this model."""
        super().set_params(params)
        params = dict_to_namespace(params)

        assert hasattr(params, 'prior_mean_f')
        self.prior_mean_f = params.prior_mean_f

    def transform_data_y(self):
        """Transform data.y using PriorMeanDataTransformer."""
        self.dt = PriorMeanDataTransformer(self.data, self.prior_mean_f, False)
        y_trans = self.dt.transform_y_data()
        self.data = Namespace(x=self.data.x, y=y_trans)

    def gen_list(self, x_list, z, s, nsamp):
        """
        Draw nsamp samples from generative process, given list of inputs
        x_list, posterior sample z, and seed s.

        Parameters
        ----------
        x_list : list
            List of numpy ndarrays each with shape=(self.params.ndimx,)
        z : Namespace
            Namespace of GP hyperparameters.
        s : int
            The seed, a positive integer.
        nsamp : int
            The number of samples to draw from generative process.

        Returns
        -------
        list
            A list with len=len(x_list) of numpy ndarrays, each with
            shape=(nsamp,).
        """
        x_list = self.transform_xin_list(x_list)
        pred_list = self.sample_gp_pred(nsamp, x_list)
        pred_list = [
            self.dt.inv_transform_y_data(pr, x) for pr, x in zip(pred_list, x_list)
        ]
        return pred_list

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
        x_list = self.transform_xin_list(x_list)
        pred_list = self.sample_gp_post_pred(
            nsamp, x_list, full_cov=True, nloop=np.min([50, nsamp])
        )
        pred_list = [
            self.dt.inv_transform_y_data(pr, x) for pr, x in zip(pred_list, x_list)
        ]
        return pred_list

    def __str__(self):
        return f'StanPriorMeanGp with params={self.params}'


class PriorMeanDataTransformer:
    """
    A class to transform (and inverse transform) data, based on a prior mean regression.
    """

    def __init__(self, data, prior_mean_f, verbose=True):
        """
        Parameters
        ----------
        data : Namespace
            Namespace containing data.
        prior_mean_f : function
            Prior mean function.
        verbose : bool
            If True, print description string.
        """
        self._set_data(data)
        self._set_prior_mean_f(prior_mean_f)
        self._set_verbose(verbose)

    def _set_data(self, data):
        """Set self.data"""
        self.data = data

    def _set_prior_mean_f(self, prior_mean_f):
        """Set self.prior_mean_f."""
        self.prior_mean_f = prior_mean_f

    def _set_verbose(self, verbose):
        """Set verbose options."""
        self.verbose = verbose
        if self.verbose:
            self._print_str()

    def transform_y_data(self, y_data=None, x_data=None):
        """Transform and return self.data.y"""

        # Transform self.data.y into new list
        y_trans = [y - self.prior_mean_f(x) for x, y in zip(self.data.x, self.data.y)]
        return y_trans

    def inv_transform_y_data(self, y_arr, x_single_arr):
        """Return inverse transform of y_arr."""

        # Compute prior mean val for the single input
        prior_mean_val = self.prior_mean_f(x_single_arr)

        # Inverse transform y_arr into list
        y_inv_trans_list = [y + prior_mean_val for y in list(y_arr)]

        # Transform back to array and return
        y_inv_trans = np.array(y_inv_trans_list).reshape(-1)
        return y_inv_trans

    def _print_str(self):
        """Print a description string."""
        print('*PriorMeanDataTransformer')
