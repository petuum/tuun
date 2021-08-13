"""
Classes for GP models with Stan that perform transfer optimization.
"""

from argparse import Namespace
import numpy as np
import copy

from .gp_stan import StanGp
from .regression.transfer_regression import TransferRegression
from ..util.misc_util import dict_to_namespace


class StanTransferGp(StanGp):
    """
    GP model with transferred prior mean based on a regression model.
    """
    def __init__(self, params=None, data=None, verbose=None):
        self.set_params(params)
        self.set_verbose(verbose)
        self.set_model(data)

    def set_params(self, params):
        """Set self.params, the parameters for this model."""
        super().set_params(params)
        params = dict_to_namespace(params)

        assert hasattr(params, 'transfer_config')
        self.params.transfer_config = params.transfer_config

    def set_model(self, data):
        """Set GP Stan model and regression model."""
        self.model = self.get_model()
        self.regressor = self.get_regressor(data)
        #self.regressor = self.get_proxy_regressor(data) # TODO

    def get_regressor(self, data):
        """Return transfer (prior mean) regressor."""

        # Define regressor
        regressor = TransferRegression(self.params.transfer_config)

        if len(data.x) < 1:
            regressor = None
        else:
            mean_errors = []

            # TODO: remove extra files such as .DS_STORE (or ignore files that break)
            for i, reg in enumerate(regressor.model_fnames):
                try:
                    val_acc = regressor.evaluate_model(reg, data.x)
                    neg_error = -1. * np.mean((data.y - val_acc) ** 2)
                    mean_errors.append((neg_error, i))
                except:
                    print(f'Transfer model file in tarball did not load: {reg}')
            mean_errors.sort(reverse=True)
            if mean_errors[0][0] > self.params.transfer_config.get('metric_threshold', 0.6):
                regressor.set_best_model(-1)
            else:
                regressor.set_best_model(mean_errors[0][1])

        return regressor

    def get_proxy_regressor(self, data):
        if not data:
            regressor = None
        else:
            def regressor(x): return np.linalg.norm(x)

        return regressor

    def transform_data_y(self):
        """Transform data.y using PriorMeanDataTransformer."""
        self.dt = PriorMeanDataTransformer(self.data, self.regressor, False)
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
        return f'StanTransferGp with params={self.params}'


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
        if prior_mean_f is None:
            # Default prior mean function is constant 0 function
            def prior_mean_f(x): return 0.

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
