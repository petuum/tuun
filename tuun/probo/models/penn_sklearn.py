"""
Classes for probabilistic ensemble neural network (PENN) models implemented in
scikit-learn.
"""

from argparse import Namespace
import copy
import numpy as np
from sklearn.neural_network import MLPRegressor
from sklearn.utils._testing import ignore_warnings
from sklearn.exceptions import ConvergenceWarning

from ..util.misc_util import dict_to_namespace


class SklearnPenn:
    """
    Probabilistic ensemble neural network (PENN) model implemented in
    scikit-learn.
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
        if verbose:
            self.print_str()

    def set_params(self, params):
        """Set self.params, the parameters for this model."""
        params = dict_to_namespace(params)

        self.params = Namespace()
        self.params.n_ensemble = getattr(params, 'n_ensemble', 5)
        self.params.hls = getattr(params, 'hls', (20, 30, 40))
        self.params.max_iter = getattr(params, 'max_iter', 500)
        self.params.alpha = getattr(params, 'alpha', 0.01)
        self.params.trans_x = getattr(params, 'trans_x', False)

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

    @ignore_warnings(category=ConvergenceWarning)
    def inf(self, data):
        """Set data, run inference."""
        self.set_data(data)

        self.ensemble = [
            MLPRegressor(
                max_iter=self.params.max_iter,
                solver='lbfgs',
                activation='relu',
                alpha=self.params.alpha,
                hidden_layer_sizes=self.params.hls,
                random_state=i,
            )
            for i in range(self.params.n_ensemble)
        ]

        x_train = np.array(self.data.x)
        y_train = np.array(self.data.y).reshape(-1)

        # Train ensemble
        for idx, nn in enumerate(self.ensemble):
            nn.fit(x_train, y_train)

    def post(self, s):
        """Return one posterior sample"""
        return np.random.choice(self.ensemble)

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
        pred_list = [
            np.array([z.predict(x.reshape(1, -1)) for _ in range(nsamp)]).reshape(-1)
            for x in x_list
        ]  # TODO: predict on batch of x_list
        return pred_list

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
        pred_list = [
            np.array([nn.predict(x.reshape(1, -1)) for nn in self.ensemble]).reshape(-1)
            for x in x_list
        ]  # TODO: predict on batch of x_list
        return pred_list

    def print_str(self):
        """Print a description string."""
        print('*[INFO] ' + str(self))

    def __str__(self):
        return f'SklearnPenn with params={self.params}'
