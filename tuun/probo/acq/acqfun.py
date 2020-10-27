"""
Classes to manage acquisition functions.
"""

from argparse import Namespace
import numpy as np
from inspect import getfullargspec

from .acqdef import Acquisitioner
from ..util.data_transform import DataTransformer
from ..util.misc_util import dict_to_namespace


def get_acqfunction_from_config(af_params, verbose=False):
    """
    Return an AcqFunction instance given af_params config.

    Parameters
    ----------
    af_params : Namespace_or_dict
        Namespace or dict of parameters for the acqfunction.
    verbose : bool
        If True, the acqfunction will print description string.

    Returns
    -------
    AcqFunction
        An AcqFunction instance.
    """
    af_params = dict_to_namespace(af_params)
    return AcqFunction(params=af_params, verbose=verbose)


class AcqFunction:
    """Class to manage acquisition functions."""

    def __init__(self, params=None, verbose=True):
        """
        Parameters
        ----------
        params : Namespace_or_dict
            Namespace or dict of parameters.
        verbose : bool
            If True, print description string.
        """
        self._set_params(params)
        self._set_verbose(verbose)

    def _set_params(self, params):
        """Set parameters for the AcqFunction."""
        params = dict_to_namespace(params)

        # Set defaults
        self.params = Namespace()
        self.params.acq_str = getattr(params, 'acq_str', 'ei')
        self.params.n_gen = getattr(params, 'n_gen', 10)
        self.params.trans_str = getattr(params, 'trans_str', '')

        assert self.params.acq_str in ['ei', 'ucb', 'pi', 'ts', 'mean', 'rand', 'null']
        assert self.params.trans_str in ['y', '']

    def _set_verbose(self, verbose):
        """Set verbose options."""
        self.verbose = verbose
        if self.verbose:
            self._print_str()

    def setup_function(self, data, model, acqoptimizer):
        """
        Setup acquisition function. Transform data and then call inf on model.
        Optionally: cache/compute quantities, and instantiate objects used in
        acquisition function (e.g. when doing sequential optimization).

        Parameters
        ----------
        data : Namespace
            Namespace of data.
        model : Model instance
            Model instance.
        acqoptimizer : AcqOptimizer
            An AcqOptimizer object that may contain attributes to get an appropriate
            acqfunction (e.g. acqoptimizer.xin_is_list, which, if True, gives a list of
            xin as input to acqfunction).
        """

        if model is None:
            # If there is no model, use random acquisition function
            self.params.acq_str = 'rand'

        if data is None or data.x == []:
            # If there is no data, do nothing (don't set self.acqf)
            pass
        else:
            self.data = self._set_acq_data(data)
            self.model = model

            # Run inf
            self._run_inf()

            # Check for attributes in acqoptimizer
            xin_is_list = getattr(acqoptimizer, 'xin_is_list', False)

            # Set self.acqf
            self._set_acquisitioner()
            self._set_pred_list_map(xin_is_list)
            self.acqf = self._acqmap_list if xin_is_list else self._acqmap_single

    def __call__(self, x):
        """Class is callable and returns self.acqf(x)."""
        return self.acqf(x)

    def _set_acq_data(self, data):
        """
        Convert data to correct format (data.x is a list and data.y is a 1d numpy
        array), apply any transforms, and set as self.acq_data.
        """

        # Convert data to correct format
        acq_data = Namespace
        acq_data.x = list(data.x)
        acq_data.y = np.array(data.y).reshape(-1)

        # Apply any transforms
        if self.params.trans_str == 'y':
            self.dt = DataTransformer(acq_data, False)
            acq_data.y = self.dt.transform_y_data()
        else:
            self.dt = None

        self.acq_data = acq_data

    def _run_inf(self):
        """Run inference in model."""
        if self.params.acq_str != 'null' and self.params.acq_str != 'rand':
            self.model.inf(self.acq_data)

    def _set_acquisitioner(self):
        """Set self.acquisitioner."""
        acqp = Namespace(acq_str=self.params.acq_str, ypred_str='sample')
        self.acquisitioner = Acquisitioner(self.acq_data, acqp, False)

    def _acqmap_list(self, xin_list):
        """Return acqmap for a list of xin."""
        if self.params.acq_str == 'null':
            return [0.0 for xin in xin_list]
        elif self.params.acq_str == 'rand':
            return [np.random.random() for xin in xin_list]
        else:
            pred_list = self.pred_list_map(self.params.n_gen, xin_list)

        return self._apply_acq_to_pred_list(pred_list)

    def _acqmap_single(self, xin):
        """Return acqmap for a single xin. Returns acqmap(xin) value, not list."""
        return self._acqmap_list([xin])[0]

    def _apply_acq_to_pred_list(self, pred_list):
        """Apply acquisition function to pred list."""
        return [self.acquisitioner.acqfn(p) for p in pred_list]

    def _set_pred_list_map(self, xin_is_list=True):
        """
        Set self.pred_list_map, which maps an xin_list to a list of yout np.arrays
        (shape (-1, )), based on methods in self.model, parameter self.params.acq_str,
        and xin_is_list.
        """

        if self.params.acq_str == 'ts':
            if xin_is_list and hasattr(self.model, 'gen_list'):
                self.pred_list_map = self._plm_ts_gen_list
                self.pred_list_map_id = 0
            elif hasattr(self.model, 'gen'):
                self.pred_list_map = self._plm_ts_gen
                self.pred_list_map_id = 1
            elif hasattr(self.model, 'gen_list'):
                self.pred_list_map = self._plm_ts_gen_list
                self.pred_list_map_id = 2

        else:
            if xin_is_list and hasattr(self.model, 'postgen_list'):
                self.pred_list_map = self._plm_pe_postgen_list
                self.pred_list_map_id = 3
            elif (
                xin_is_list
                and hasattr(self.model, 'gen_list')
                and hasattr(self.model, 'post')
            ):
                self.pred_list_map = self._plm_pe_gen_list
                self.pred_list_map_id = 4
            elif hasattr(self.model, 'postgen'):
                self.pred_list_map = self._plm_pe_postgen
                self.pred_list_map_id = 5
            elif hasattr(self.model, 'gen') and hasattr(self.model, 'post'):
                self.pred_list_map = self._plm_pe_gen
                self.pred_list_map_id = 6
            elif hasattr(self.model, 'postgen_list'):
                self.pred_list_map = self._plm_pe_postgen_list
                self.pred_list_map_id = 7
            elif hasattr(self.model, 'gen_list') and hasattr(self.model, 'post'):
                self.pred_list_map = self._plm_pe_gen_list
                self.pred_list_map_id = 8

    def _plm_ts_gen_list(self, nsamp, x_list):
        """One call to post, then calls to gen_list"""
        z = self.model.post(0)
        if 'nsamp' in getfullargspec(self.model.gen_list).args:
            return self.model.gen_list(x_list, z, 0, nsamp)
        else:
            samp_list = [self.model.gen_list(x_list, z, s) for s in range(nsamp)]
            return list(np.array(samp_list).T)

    def _plm_ts_gen(self, nsamp, x_list):
        """One call to post, then calls to gen for each x in x_list"""
        z = self.model.post(0)
        return [self.model.gen(x, z, s) for s, x in enumerate(x_list)]

    def _plm_pe_postgen_list(self, nsamp, x_list):
        """Calls to postgen_list"""
        if 'nsamp' in getfullargspec(self.model.postgen_list).args:
            return self.model.postgen_list(x_list, 0, nsamp)
        else:
            samp_list = [self.model.postgen_list(x_list, s) for s in range(nsamp)]
            return list(np.array(samp_list).T)

    def _plm_pe_gen_list(self, nsamp, x_list):
        """Calls to post then gen_list"""
        post_list = [self.model.post(s) for s in range(nsamp)]
        if 'nsamp' in getfullargspec(self.model.gen_list).args:
            # TODO: fix next line, given a gen_list with z_is_list option
            return self.model.gen_list(x_list, post_list[0], 0, nsamp)
        else:
            samp_list = [
                self.model.gen_list(x_list, z, s) for s, z in enumerate(post_list)
            ]
            return list(np.array(samp_list).T)

    def _plm_pe_postgen(self, nsamp, x_list):
        """Calls to postgen for each x in x_list"""
        return [self.model.postgen(x, s) for s, x in enumerate(x_list)]

    def _plm_pe_gen(self, nsamp, x_list):
        """Calls to post then gen for each x in x_list"""
        post_list = [self.model.post(s) for s in range(nsamp)]
        return [
            self.model.gen(x, z, s) for s, x, z in zip(range(nsamp), x_list, post_list)
        ]

    def _print_str(self):
        """Print a description string."""
        print('*[INFO] ' + str(self))

    def __str__(self):
        return f'AcqFunction with params={self.params}'
