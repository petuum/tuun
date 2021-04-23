"""
Classes for GP models with GpyTorch for a ProductDomain (i.e. Cartesian product) of real
domains.
"""

import numpy as np
import copy

from .gp_gpytorch import GpytorchGp


class GpytorchProductGp(GpytorchGp):
    """
    Gaussian processes implemented with GPyTorch, for a product of real domains.
    """

    def transform_xin_list(self, xin_list):
        """Transform list of xin."""

        xin_list = [np.concatenate(xin) for xin in xin_list]

        # Ensure data.x is correct format (list of 1D numpy arrays)
        xin_list = [np.array(xin).reshape(-1) for xin in xin_list]

        if self.params.trans_x:
            # Apply transformation to xin_list
            xin_list_trans = xin_list  # TODO: define default transformation
        else:
            xin_list_trans = xin_list

        return xin_list_trans

    def __str__(self):
        return f'GpytorchProductGp with params={self.params}'
