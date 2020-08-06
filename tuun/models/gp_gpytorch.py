"""
Classes for GP models with GpyTorch.
"""

from argparse import Namespace
import copy
import numpy as np
import gpytorch
import torch

from ..util.misc_util import dict_to_namespace
from ..util.data_transform import DataTransformer
from .gp.gp_utils import sample_mvn


class GpytorchGp:
    """
    Gaussian processes implemented with GPyTorch.
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
        self.set_verbose(verbose)

    def set_params(self, params):
        """Set self.params, the parameters for this model."""
        params = dict_to_namespace(params)

        # Set self.params
        self.params = Namespace()
        self.params.n_train_iter = getattr(params, 'n_train_iter', 200)
        self.params.trans_x = getattr(params, 'trans_x', False)

    def set_verbose(self, verbose):
        """Set verbose options."""
        self.verbose = verbose
        if self.verbose:
            self.print_str()

    def set_data(self, data):
        """Set self.data."""
        self.data_init = copy.deepcopy(data)

        # Transform data.X. TODO: make consistent with transform data.y below
        self.data = Namespace(
            X=self.transform_xin_list(self.data_init.X), y=self.data_init.y
        )

        # Transform data.y
        self.transform_data_y()

    def transform_data_y(self):
        """Transform data.y using DataTransformer."""
        self.dt = DataTransformer(self.data, False)
        y_trans = self.dt.transform_y_data()
        self.data = Namespace(X=self.data.X, y=y_trans)

    def transform_xin_list(self, xin_list):
        """Transform list of xin (e.g. in data.X)."""
        if self.params.trans_x:
            # apply transformation to xin_list
            xin_list_trans = xin_list  # TODO: define default transformation
        else:
            xin_list_trans = xin_list

        return xin_list_trans

    def inf(self, data):
        """Run inference on data."""
        self.set_data(data)

        train_x = torch.from_numpy(np.array(self.data.X))
        train_y = torch.from_numpy(np.array(self.data.y).reshape(-1))

        # Initialize self.likelihood and self.model
        self.likelihood = gpytorch.likelihoods.GaussianLikelihood()
        self.model = ExactGPModel(train_x, train_y, self.likelihood)

        # Go to training mode
        self.model.train()
        self.likelihood.train()

        # Set adam optimizer
        optimizer = torch.optim.Adam( [{'params': self.model.parameters()}, ], lr=0.1)

        # Set GP loss
        mll = gpytorch.mlls.ExactMarginalLogLikelihood(self.likelihood, self.model)

        # Training procedure
        for i in range(self.params.n_train_iter):
            optimizer.zero_grad()
            output = self.model(train_x)
            loss = -mll(output, train_y)
            loss.backward()
            optimizer.step()

    def postgen_list(self, x_list, s, nsamp):
        """
        Draw nsamp samples from posterior predictive distribution, given list
        of inputs x_list and seed s.

        Parameters
        ----------
        x_list : list
            List of numpy ndarrays each with shape=(-1,).
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

        # Go to evaluation mode
        self.model.eval()
        self.likelihood.eval()

        # Make predictions by feeding model through self.likelihood
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            test_x = torch.from_numpy(np.array(x_list))
            observed_pred = self.likelihood(self.model(test_x))

            try:
                samp_torch = observed_pred.sample(sample_shape=torch.Size((nsamp,)))
                samp = samp_torch.numpy().T
                pred_list = list(samp)
            except:
                op_mean = observed_pred.mean.numpy()
                op_cov = observed_pred.covariance_matrix.numpy()
                samp = sample_mvn(op_mean, op_cov, nsamp).T
                pred_list = list(samp)

        return pred_list

    def print_str(self):
        """Print a description string."""
        print('*GpytorchGp with params={}'.format(self.params))


class ExactGPModel(gpytorch.models.ExactGP):
    """GPyTorch ExactGpModel class."""

    def __init__(self, train_x, train_y, likelihood):
        super(ExactGPModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)
