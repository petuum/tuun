"""
Classes for Bayesian neural network (BNN) models implemented in Pyro.
"""

from argparse import Namespace
import copy
import numpy as np
import torch

from ..util.misc_util import dict_to_namespace
from .pyro.bnn import BNN


class PyroBnn:
    """
    Bayesian neural networks (BNNs) implemented in Pyro.
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
        self.set_torch_seed()
        self.set_verbose(verbose)
        self.set_model()

    def set_params(self, params):
        """Set self.params, the parameters for this model."""
        params = dict_to_namespace(params)

        self.params = Namespace()
        self.params.likelihood_scale = getattr(params, 'likelihood_scale', 0.5)
        self.params.trans_x = getattr(params, 'trans_x', False)
        self.params.seed = getattr(params, 'seed', -1)

    def set_torch_seed(self):
        """Set torch random number seed."""
        if self.params.seed > 0:
            torch.manual_seed(self.params.seed)

    def set_verbose(self, verbose):
        """Set verbose options."""
        self.verbose = verbose
        if self.verbose:
            self.print_str()

    def set_model(self):
        """Set pyro bnn regression model."""
        self.model = self.get_model()

    def get_model(self):
        """Returns BNN model object."""
        return BNN(self.params.likelihood_scale)

    def set_data(self, data):
        """Set self.data."""
        self.data_init = copy.deepcopy(data)
        self.data = copy.deepcopy(self.data_init)

        # Transform data.x
        self.data.x = self.transform_xin_list(self.data.x)

    def transform_xin_list(self, xin_list):
        """Transform list of xin (e.g. in data.x)."""
        # Ensure data.x is correct format (list of 1D numpy arrays)
        xin_list = [np.array(xin).reshape(-1) for xin in xin_list]

        if self.params.trans_x:
            # apply transformation to xin_list
            xin_list_trans = xin_list  # TODO: define default transformation
        else:
            xin_list_trans = xin_list

        return xin_list_trans

    def inf(self, data):
        """Set data, run inference."""
        self.set_data(data)
        self.bnn = self.get_model()

        x_train = torch.tensor(np.array(self.data.x)).float()
        y_train = torch.tensor(np.array(self.data.y).reshape(-1)).float()

        self.bnn.train(x_train, y_train)

    def postgen_list(self, x_list, s, nsamp):
        """
        Draw nsamp samples from posterior predictive distribution, given list of inputs
        x_list and seed s.

        Parameters
        ----------
        x_list : list
            List of numpy ndarrays each with shape=(-1,).
        s : int
            The seed, a positive integer.
        nsamp : int
            The number of samples to draw from the posterior predictive distribution.

        Returns
        -------
        list
            A list with len=len(x_list) of numpy ndarrays, each with shape=(nsamp,).
        """
        x_list = self.transform_xin_list(x_list)
        x_arr = np.array(x_list)
        pred_arr = self.bnn.postpred_nsamp_np(nsamp, x_arr).T
        pred_list = list(pred_arr)
        return pred_list

    def post(self, s):
        """Return one posterior sample."""
        return self.bnn.sample_post()

    def gen_list(self, x_list, z, s, nsamp):
        """
        Draw nsamp samples from generative process, given list of inputs x_list,
        posterior sample z, and seed s.

        Parameters
        ----------
        x_list : list
            List of numpy ndarrays each with shape=(-1,).
        z : Namespace
            Namespace of GP hyperparameters.
        s : int
            The seed, a positive integer.
        nsamp : int
            The number of samples to draw from generative process.

        Returns
        -------
        list
            A list with len=len(x_list) of numpy ndarrays, each with shape=(nsamp,).
        """
        x_list = self.transform_xin_list(x_list)
        x_arr = np.array(x_list)
        pred_arr = self.bnn.pred_nsamp_np(nsamp, x_arr, z).T
        pred_list = list(pred_arr)
        return pred_list

    def print_str(self):
        """Print a description string."""
        print('*[INFO] ' + str(self))

    def __str__(self):
        return f'PyroBnn with params={self.params}'
