"""
Classes for GP models with Stan.
"""

from argparse import Namespace
import copy
import numpy as np

from .stan.gp import get_model as get_model_gp
from .stan.gp_fixedsig import get_model as get_model_gp_fixedsig
from .gp.gp_utils import kern_exp_quad, sample_mvn, gp_post
from ..util.data_transform import DataTransformer
from ..util.misc_util import dict_to_namespace, suppress_stdout_stderr


class StanGp:
    """
    Hierarchical GPs, implemented with Stan.
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
        self.set_model()

    def set_params(self, params):
        """Set self.params, the parameters for this model."""
        params = dict_to_namespace(params)

        # Set self.params
        self.params = Namespace()
        self.params.ndimx = params.ndimx
        self.params.model_str = getattr(params, 'model_str', 'optfixedsig')
        self.params.ig1 = getattr(params, 'ig1', 4.0)
        self.params.ig2 = getattr(params, 'ig2', 3.0)
        self.params.n1 = getattr(params, 'n1', 1.0)
        self.params.n2 = getattr(params, 'n2', 1.0)
        self.params.sigma = getattr(params, 'sigma', 1e-5)
        self.params.niter = getattr(params, 'niter', 70)
        self.params.kernel = getattr(params, 'kernel', kern_exp_quad)
        self.params.trans_x = getattr(params, 'trans_x', False)
        self.params.print_warnings = getattr(params, 'print_warnings', False)

    def set_verbose(self, verbose):
        """Set verbose options."""
        self.verbose = verbose
        if self.verbose:
            self.print_str()

    def set_model(self):
        """Set GP stan model."""
        self.model = self.get_model()

    def get_model(self):
        """Returns GP stan model."""
        if (
            self.params.model_str == 'optfixedsig'
            or self.params.model_str == 'sampfixedsig'
        ):
            return get_model_gp_fixedsig(print_status=self.verbose)
        elif self.params.model_str == 'opt' or self.params.model_str == 'samp':
            return get_model_gp(print_status=self.verbose)
        elif self.params.model_str == 'fixedparam':
            return None

    def set_data(self, data):
        """Set self.data."""
        self.data_init = copy.deepcopy(data)
        self.data = copy.deepcopy(self.data_init)

        # Transform data.x
        self.data.x = self.transform_xin_list(self.data.x)

        # Transform data.y
        self.transform_data_y()

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

    def transform_data_y(self):
        """Transform data.y using DataTransformer."""
        self.dt = DataTransformer(self.data, False)
        y_trans = self.dt.transform_y_data()
        self.data = Namespace(x=self.data.x, y=y_trans)

    def inf(self, data):
        """Set data, run inference, update self.sample_list."""
        self.set_data(data)
        self.infer_post_and_update_samples()

    def post(self, s):
        """Return one posterior sample."""
        return np.random.choice(self.sample_list)

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
        pred_list = [self.dt.inv_transform_y_data(pr) for pr in pred_list]
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
        pred_list = [self.dt.inv_transform_y_data(pr) for pr in pred_list]
        return pred_list

    def infer_post_and_update_samples(self, seed=543210, print_result=False):
        """Update self.sample_list."""
        data_dict = self.get_stan_data_dict()
        if self.params.model_str == 'optfixedsig' or self.params.model_str == 'opt':

            def run_stan_optimizing(stan_opt_str):
                with suppress_stdout_stderr():
                    return self.model.optimizing(
                        data_dict,
                        iter=self.params.niter,
                        seed=seed,
                        as_vector=True,
                        algorithm=stan_opt_str,
                    )

            try:
                stanout = run_stan_optimizing('LBFGS')
            except RuntimeError:
                if self.params.print_warnings:
                    print(
                        '\t*Stan LBFGS optimizer failed. Running Newton '
                        + 'optimizer instead.'
                    )
                stanout = run_stan_optimizing('Newton')

        elif self.params.model_str == 'samp' or self.params.model_str == 'sampfixedsig':
            with suppress_stdout_stderr():
                stanout = self.model.sampling(
                    data_dict,
                    iter=self.params.niter + self.params.nwarmup,
                    warmup=self.params.nwarmup,
                    chains=1,
                    seed=seed,
                    refresh=1000,
                )

        elif self.params.model_str == 'fixedparam':
            stanout = None

        self.sample_list = self.get_sample_list_from_stan_out(stanout)
        if print_result:
            self.print_inference_result()

    def get_stan_data_dict(self):
        """Return data dict for stan sampling method."""
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
                'D': self.params.ndimx,
                'N': len(self.data.x),
                'x': self.data.x,
                'y': np.array(self.data.y).reshape(-1),
            }
        elif self.params.model_str == 'opt' or self.params.model_str == 'samp':
            return {
                'ig1': self.params.ig1,
                'ig2': self.params.ig2,
                'n1': self.params.n1,
                'n2': self.params.n2,
                'n3': self.params.n3,
                'n4': self.params.n4,
                'D': self.params.ndimx,
                'N': len(self.data.x),
                'y': np.array(self.data.y).reshape(-1),
                'x': self.data.x,
            }

    def get_sample_list_from_stan_out(self, stanout):
        """Convert stan output to sample_list."""
        if self.params.model_str == 'optfixedsig':
            return [
                Namespace(
                    ls=stanout['rho'], alpha=stanout['alpha'], sigma=self.params.sigma
                )
            ]
        elif self.params.model_str == 'opt':
            return [
                Namespace(
                    ls=stanout['rho'], alpha=stanout['alpha'], sigma=stanout['sigma']
                )
            ]
        elif self.params.model_str == 'sampfixedsig':
            sdict = stanout.extract(['rho', 'alpha'])
            return [
                Namespace(
                    ls=sdict['rho'][i], alpha=sdict['alpha'][i], sigma=self.params.sigma
                )
                for i in range(sdict['rho'].shape[0])
            ]
        elif self.params.model_str == 'samp':
            sdict = stanout.extract(['rho', 'alpha', 'sigma'])
            return [
                Namespace(
                    ls=sdict['rho'][i], alpha=sdict['alpha'][i], sigma=sdict['sigma'][i]
                )
                for i in range(sdict['rho'].shape[0])
            ]
        elif self.params.model_str == 'fixedparam':
            return [
                Namespace(
                    ls=self.params.ls, alpha=self.params.alpha, sigma=self.params.sigma
                )
            ]

    def print_inference_result(self):
        """Print results of stan inference."""
        if (
            self.params.model_str == 'optfixedsig'
            or self.params.model_str == 'opt'
            or self.params.model_str == 'fixedparam'
        ):
            print('*ls pt est = ' + str(self.sample_list[0].ls) + '.')
            print('*alpha pt est = ' + str(self.sample_list[0].alpha) + '.')
            print('*sigma pt est = ' + str(self.sample_list[0].sigma) + '.')
        elif self.params.model_str == 'samp' or self.params.model_str == 'sampfixedsig':
            ls_arr = np.array([ns.ls for ns in self.sample_list])
            alpha_arr = np.array([ns.alpha for ns in self.sample_list])
            sigma_arr = np.array([ns.sigma for ns in self.sample_list])
            print('*ls mean = ' + str(ls_arr.mean()) + '.')
            print('*ls std = ' + str(ls_arr.std()) + '.')
            print('*alpha mean = ' + str(alpha_arr.mean()) + '.')
            print('*alpha std = ' + str(alpha_arr.std()) + '.')
            print('*sigma mean = ' + str(sigma_arr.mean()) + '.')
            print('*sigma std = ' + str(sigma_arr.std()) + '.')
        print('-----')

    def sample_gp_pred(self, nsamp, input_list, lv=None):
        """
        Sample from GP predictive distribution given one posterior GP sample.

        Parameters
        ----------
        nsamp : int
            Number of samples from predictive distribution.
        input_list : list
            A list of numpy ndarray shape=(self.params.ndimx, ).
        lv : Namespace
            Namespace for posterior sample.

        Returns
        -------
        list
            A list of len=len(input_list) of numpy ndarrays shape=(nsamp, 1).
        """
        if lv is None:
            if (
                self.params.model_str == 'optfixedsig'
                or self.params.model_str == 'opt'
                or self.params.model_str == 'fixedparam'
            ):
                lv = self.sample_list[0]
            elif (
                self.params.model_str == 'samp'
                or self.params.model_str == 'sampfixedsig'
            ):
                lv = self.sample_list[np.random.randint(len(self.sample_list))]
        postmu, postcov = gp_post(
            self.data.x,
            self.data.y,
            input_list,
            lv.ls,
            lv.alpha,
            lv.sigma,
            self.params.kernel,
        )
        single_post_sample = sample_mvn(postmu, postcov, 1).reshape(-1)
        pred_list = [
            single_post_sample for _ in range(nsamp)
        ]  #### TODO: instead of duplicating this TS, sample nsamp times from generative process (given/conditioned-on this TS)
        return list(np.stack(pred_list).T)

    def sample_gp_post_pred(self, nsamp, input_list, full_cov=False, nloop=None):
        """
        Sample from GP posterior predictive distribution.

        Parameters
        ----------
        nsamp : int
            Number of samples from posterior predictive distribution.
        input_list : list
            A list of numpy ndarray shape=(self.params.ndimx, ).
        full_cov : bool
            If True, return covariance matrix, else return diagonal only.

        Returns
        -------
        list
            A list of len=len(input_list) of numpy ndarrays shape=(nsamp, 1).
        """
        if (
            self.params.model_str == 'optfixedsig'
            or self.params.model_str == 'opt'
            or self.params.model_str == 'fixedparam'
        ):
            nloop = 1
            sampids = [0]
        elif self.params.model_str == 'samp' or self.params.model_str == 'sampfixedsig':
            if nloop is None:
                nloop = nsamp
            nsamp = int(nsamp / nloop)
            sampids = np.random.randint(len(self.sample_list), size=(nloop,))
        ppred_list = []
        for i in range(nloop):
            samp = self.sample_list[sampids[i]]
            postmu, postcov = gp_post(
                self.data.x,
                self.data.y,
                input_list,
                samp.ls,
                samp.alpha,
                samp.sigma,
                self.params.kernel,
                full_cov,
            )
            if full_cov:
                ppred_list.extend(list(sample_mvn(postmu, postcov, nsamp)))
            else:
                ppred_list.extend(
                    list(
                        np.random.normal(
                            postmu.reshape(-1),
                            postcov.reshape(-1),
                            size=(nsamp, len(input_list)),
                        )
                    )
                )
        return list(np.stack(ppred_list).T)

    def print_str(self):
        """Print a description string."""
        print('*[INFO] ' + str(self))

    def __str__(self):
        return f'StanGp with params={self.params}'
