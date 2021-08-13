"""
Classes for GP models with Stan.
"""

from argparse import Namespace
import copy
import time
import numpy as np

from .base import Base
from .stan.gp import get_model as get_model_gp
from .stan.gp_fixedsig import get_model as get_model_gp_fixedsig
from .stan.gp_fixedsig_matern import get_model as get_model_gp_fixedsig_matern
from .stan.gp_fixedsig_ard import get_model as get_model_gp_fixedsig_ard
from .stan.gp_fixedsig_all import get_model as get_model_gp_fixedsig_all
from .gp.gp_utils import kern_exp_quad, kern_matern, sample_mvn, gp_post
from .util.data_transform import DataTransformer
from .util.misc_util import dict_to_namespace, suppress_stdout_stderr


class StanGp(Base):
    """Hierarchical GPs, implemented with Stan."""

    def set_params(self, params):
        """Set self.params, the parameters for this model."""
        super().set_params(params)
        params = dict_to_namespace(params)

        # Set self.params
        self.params.name = getattr(params, 'name', 'StanGp')
        self.params.ndimx = params.ndimx
        self.params.model_str = getattr(params, 'model_str', 'optfixedsig')
        self.params.kernel_str = getattr(params, 'kernel_str', 'rbf')
        self.params.kernel = getattr(params, 'kernel', None)
        self.params.ig1 = getattr(params, 'ig1', 4.0)
        self.params.ig2 = getattr(params, 'ig2', 3.0)
        self.params.n1 = getattr(params, 'n1', 1.0)
        self.params.n2 = getattr(params, 'n2', 1.0)
        self.params.ls = getattr(params, 'ls', 1.0)
        self.params.alpha = getattr(params, 'alpha', 1.0)
        self.params.sigma = getattr(params, 'sigma', 1e-5)
        self.params.niter = getattr(params, 'niter', 70)
        self.params.nmcmc = getattr(params, 'nmcmc', 100)
        self.params.nwarmup = getattr(params, 'nwarmup', 50)
        self.params.trans_y = getattr(params, 'trans_y', False)
        self.params.trans_x = getattr(params, 'trans_x', False)
        self.params.print_inf = getattr(params, 'print_inf', True)

        # Set kernel params
        self.set_kernel_params()

        # Set Stan GP model
        self.set_model()

    def set_kernel_params(self):
        """Set self.params relating to the kernel."""
        # Set self.params.kernel if it is None
        if not self.params.kernel:
            if self.params.kernel_str in ('rbf', 'rbf_ard'):
                self.params.kernel = kern_exp_quad
            elif self.params.kernel_str in ('mat12', 'mat32', 'mat52'):
                nu_dict = {'mat12': 0.5, 'mat32': 1.5, 'mat52': 2.5}
                nu = nu_dict[self.params.kernel_str]
                self.params.kernel = lambda a, b, c, d: kern_matern(a, b, c, d, nu=nu)

        # Set self.params.kernel_id based on self.params.kernel_str
        kernel_id_dict = {'rbf': 1, 'mat12': 2, 'mat32': 3, 'mat52': 4, 'rbf_ard': 5}
        self.params.kernel_id = kernel_id_dict[self.params.kernel_str]

    def set_model(self):
        """Return GP stan model."""
        if self.params.model_str in ('optfixedsig', 'sampfixedsig'):
            model = get_model_gp_fixedsig_all(print_status=self.params.verbose)
        elif self.params.model_str in ('opt', 'samp'):
            model = get_model_gp(print_status=self.params.verbose)
        elif self.params.model_str == 'fixedparam':
            model = None

        self.model = model

    def infer_hypers(self, data):
        """Set data, run inference on hyperparameters, update self.sample_list."""
        # Set data
        self.set_data(data)

        # Run inference, update self.sample_list, record time
        start_time = time.time()
        self.infer_post_and_update_sample_list()
        end_time = time.time()
        self.last_inf_time = end_time - start_time

        # Optionally print inference result
        if self.params.print_inf:
            self.print_inference_result()

    def set_data(self, data):
        """Set self.data."""
        # Copy data into self.data
        data = dict_to_namespace(data)
        self.data = copy.deepcopy(data)

        # Transform data.x
        self.data.x = self.transform_xin_list(self.data.x)

        # Transform data.y
        self.transform_data_y()

    def transform_xin_list(self, xin_list):
        """Transform list of xin (e.g. in data.x)."""
        # Ensure data.x is correct format (list of 1D numpy arrays)
        xin_list = [np.array(xin).reshape(-1) for xin in xin_list]

        if self.params.trans_x:
            # Apply transformation to xin_list
            # TODO: define default transformation
            xin_list_trans = xin_list
        else:
            xin_list_trans = xin_list

        return xin_list_trans

    def transform_data_y(self):
        """Transform data.y using DataTransformer."""
        if self.params.trans_y:
            self.dt = DataTransformer(self.data, False)
            y_trans = self.dt.transform_y_data()
            self.data = Namespace(x=self.data.x, y=y_trans)
        else:
            self.data.y = np.array(self.data.y).reshape(-1)

    def infer_post_and_update_sample_list(self, seed=543210):
        """Update self.sample_list."""
        data_dict = self.get_stan_data_dict()

        if self.params.model_str in ('optfixedsig', 'opt'):

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
                print(
                    f'{self.print_pre}Stan LBFGS optimizer failed. Running Newton '
                    + 'optimizer instead.'
                )
                stanout = run_stan_optimizing('Newton')

        elif self.params.model_str in ('sampfixedsig', 'samp'):
            with suppress_stdout_stderr():
                stanout = self.model.sampling(
                    data_dict,
                    iter=self.params.nmcmc + self.params.nwarmup,
                    warmup=self.params.nwarmup,
                    chains=1,
                    seed=seed,
                    refresh=1000,
                )

        elif self.params.model_str == 'fixedparam':
            stanout = None

        self.sample_list = self.get_sample_list_from_stan_out(stanout)

    def get_stan_data_dict(self):
        """Return data dict for stan sampling method."""
        if self.params.model_str in ('optfixedsig', 'sampfixedsig'):
            return {
                'D': self.params.ndimx,
                'N': len(self.data.x),
                'x': self.data.x,
                'y': self.data.y.flatten(),
                'ig1': self.params.ig1,
                'ig2': self.params.ig2,
                'n1': self.params.n1,
                'n2': self.params.n2,
                'sigma': self.params.sigma,
                'kernel_id': self.params.kernel_id,
            }
        elif self.params.model_str in ('opt', 'samp'):
            return {
                'D': self.params.ndimx,
                'N': len(self.data.x),
                'y': self.data.y.flatten(),
                'x': self.data.x,
                'ig1': self.params.ig1,
                'ig2': self.params.ig2,
                'n1': self.params.n1,
                'n2': self.params.n2,
                'n3': self.params.n3,
                'n4': self.params.n4,
                'kernel_id': self.params.kernel_id,
            }

    def get_sample_list_from_stan_out(self, stanout):
        """Convert stan output to sample_list."""
        ls_str = 'rhovec' if self.params.kernel_str in ('rbf_ard',) else 'rho'

        if self.params.model_str == 'optfixedsig':
            return [
                Namespace(
                    ls=stanout[ls_str], alpha=stanout['alpha'], sigma=self.params.sigma
                )
            ]
        elif self.params.model_str == 'opt':
            return [
                Namespace(
                    ls=stanout[ls_str], alpha=stanout['alpha'], sigma=stanout['sigma']
                )
            ]
        elif self.params.model_str == 'sampfixedsig':
            sdict = stanout.extract([ls_str, 'alpha'])
            return [
                Namespace(
                    ls=sdict[ls_str][i], alpha=sdict['alpha'][i], sigma=self.params.sigma
                )
                for i in range(sdict[ls_str].shape[0])
            ]
        elif self.params.model_str == 'samp':
            sdict = stanout.extract([ls_str, 'alpha', 'sigma'])
            return [
                Namespace(
                    ls=sdict[ls_str][i], alpha=sdict['alpha'][i], sigma=sdict['sigma'][i]
                )
                for i in range(sdict[ls_str].shape[0])
            ]
        elif self.params.model_str == 'fixedparam':
            return [
                Namespace(
                    ls=self.params.ls, alpha=self.params.alpha, sigma=self.params.sigma
                )
            ]

    def print_inference_result(self):
        """Print results of stan inference."""
        print(f'[INFO] {self.params.name} inference result:')
        pre = self.print_pre

        def get_ls_str(ls_arr):
            ls_str = str([float(f'{ls:.3f}') for ls in ls_arr])
            return ls_str[1:-1] if len(ls_arr) == 1 else ls_str

        if self.params.model_str in ('optfixedsig', 'opt', 'fixedparam'):
            ls_arr = np.array([self.sample_list[0].ls]).reshape(-1)
            ls_str = get_ls_str(ls_arr)

            inf_term = 'Estimated'
            print(f'{pre}{inf_term} ls = {ls_str}')
            print(f'{pre}{inf_term} alpha = {self.sample_list[0].alpha:.3f}')
            print(f'{pre}{inf_term} sigma = {self.sample_list[0].sigma}')

        elif self.params.model_str in ('samp', 'sampfixedsig'):
            ls_mat = np.array(
                [np.array([ns.ls]).reshape(-1) for ns in self.sample_list]
            )
            ls_mean_str = get_ls_str(ls_mat.mean(0))
            ls_std_str = get_ls_str(ls_mat.std(0))

            alpha_arr = np.array([ns.alpha for ns in self.sample_list])
            al_mean_str = f'{alpha_arr.mean():.3f}'
            al_std_str = f'{alpha_arr.std():.3f}'

            sigma_arr = np.array([ns.sigma for ns in self.sample_list])
            sig_mean_str = f'{sigma_arr.mean():.3e}'
            sig_std_str = f'{sigma_arr.std():.3e}'

            inf_term = 'Posterior'
            print(f'{pre}{inf_term} ls mean = {ls_mean_str}, std = {ls_std_str}')
            print(f'{pre}{inf_term} alpha mean = {al_mean_str}, std = {al_std_str}')
            print(f'{pre}{inf_term} sigma mean = {sig_mean_str}, std = {sig_std_str}')

        # Print timing
        print(f'{pre}Elapsed time: {self.last_inf_time:.3f}s.')

    def sample_pred(self, nsamp, input_list, lv=None):
        """
        Sample from GP predictive distribution given one posterior GP sample, given a
        hyperparameter lv (set to random hyperparameter in self.sample_list if None).

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
        x_pred = np.stack(input_list)
        if lv is None:
            if self.params.model_str in ('optfixedsig', 'opt', 'fixedparam'):
                lv = self.sample_list[-1]
            elif self.params.model_str in ('samp', 'sampfixedsig'):
                lv = self.sample_list[np.random.randint(len(self.sample_list))]
        mean, cov = gp_post(
            self.data.x,
            self.data.y,
            x_pred,
            lv.ls,
            lv.alpha,
            lv.sigma,
            self.params.kernel,
        )
        single_post_sample = sample_mvn(mean, cov, 1).reshape(-1)

        #### TODO: sample nsamp times from generative process (given single_post_sample)
        pred_list = [single_post_sample for _ in range(nsamp)]
        return list(np.stack(pred_list).T)

    def sample_pred_single(self, nsamp, x, lv=None):
        """Return samples from GP predictive distribution for a single input x."""
        sample_list = self.sample_pred(nsamp, [x], lv=lv)
        return sample_list[0]

    def sample_post_pred(self, nsamp, input_list, full_cov=False, nloop=None):
        """
        Sample from GP posterior predictive distribution. Uses hyperparameter samples
        from self.sample_list.

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
        if self.params.model_str in ('optfixedsig', 'opt', 'fixedparam'):
            nloop = 1
            sampids = [0]
        elif self.params.model_str in ('samp', 'sampfixedsig'):
            if nloop is None:
                nloop = nsamp
            nsamp = int(nsamp / nloop)
            sampids = np.random.randint(len(self.sample_list), size=(nloop))

        ppred_list = []
        for i in range(nloop):
            samp = self.sample_list[sampids[i]]
            mean, cov = gp_post(
                self.data.x,
                self.data.y,
                np.stack(input_list),
                samp.ls,
                samp.alpha,
                samp.sigma,
                self.params.kernel,
                full_cov,
            )
            if full_cov:
                ppred_list.extend(list(sample_mvn(mean, cov, nsamp)))
            else:
                ppred_list.extend(
                    list(
                        np.random.normal(
                            mean.reshape(-1),
                            cov.reshape(-1),
                            size=(nsamp, len(input_list)),
                        )
                    )
                )

        return list(np.stack(ppred_list).T)

    def sample_post_pred_single(self, nsamp, x, full_cov=False, nloop=None):
        """Return samples from GP posterior predictive for a single input x."""
        sample_list = self.sample_post_pred(nsamp, [x], full_cov=False, nloop=nloop)
        return sample_list[0]

    def get_prior_mean_cov(self, x_list, full_cov=True):
        """
        Return GP prior mean and covariance parameters.

        Parameters
        ----------
        x_list : list
            List of numpy ndarrays, each representing a domain point.
        full_cov : bool
            If True, return covariance matrix. If False, return list of standard
            deviations.

        Returns
        -------
        mean : ndarray
            A numpy 1d ndarray with len=len(x_list) of floats, corresponding to
            posterior mean for each x in x_list.
        cov : ndarray
            If full_cov is False, return a numpy 1d ndarray with len=len(x_list) of
            floats, corresponding to posterior standard deviations for each x in x_list.
            If full_cov is True, return the covariance matrix as a numpy ndarray
            (len(x_list) x len(x_list)).
        """
        # Set prior hypers
        if self.params.model_str == 'fixedparam':
            prior_ls = self.params.ls
            prior_alpha = self.params.alpha
        else:
            # TODO: determine strategy to set for other model_str
            pass

        # NOTE: assumes constant zero prior mean function.
        # TODO: support other mean functions.
        mean = np.zeros(len(x_list))
        cov = self.params.kernel(x_list, x_list, prior_ls, prior_alpha)

        if full_cov is False:
            cov = np.sqrt(np.diag(cov))

        return mean, cov

    def get_prior_mean_std_single(self, x):
        """For a single input x, return GP prior mean and std."""
        mean_arr, std_arr = self.get_prior_mean_cov([x], full_cov=False)
        return mean_arr[0], std_arr[0]

    def get_hp_from_sample_list(self):
        """
        Return index of a single hp from self.sample_list. Index is chosen based on
        model type.
        """
        if self.params.model_str in ('optfixedsig', 'opt', 'fixedparam'):
            idx = -1
        elif self.params.model_str in ('samp', 'sampfixedsig'):
            idx = np.random.randint(len(self.sample_list))

        hp = self.sample_list[idx]
        return hp

    def get_post_mean_cov(self, x_list, full_cov=True):
        """
        Return GP posterior mean and covariance parameters, using final hyperparameter
        in self.sample_list. If there is no data, return the GP prior parameters.

        Parameters
        ----------
        x_list : list
            List of numpy ndarrays, each representing a domain point.
        full_cov : bool
            If True, return covariance matrix. If False, return list of standard
            deviations.

        Returns
        -------
        mean : ndarray
            A numpy 1d ndarray with len=len(x_list) of floats, corresponding to
            posterior mean for each x in x_list.
        cov : ndarray
            If full_cov is False, return a numpy 1d ndarray with len=len(x_list) of
            floats, corresponding to posterior standard deviations for each x in x_list.
            If full_cov is True, return the covariance matrix as a numpy ndarray
            (len(x_list) x len(x_list)).
        """
        if len(self.data.x) == 0:
            return self.get_prior_mean_cov(x_list, full_cov)

        hp = self.get_hp_from_sample_list()

        mean, cov = gp_post(
            self.data.x,
            self.data.y,
            x_list,
            hp.ls,
            hp.alpha,
            hp.sigma,
            self.params.kernel,
            full_cov=full_cov,
        )
        return mean, cov

    def get_post_mean_std_single(self, x):
        """
        For a single input x, return GP posterior mean and std. Uses final
        hyperparameter in self.sample_list.
        """
        mean_arr, std_arr = self.get_post_mean_cov([x], full_cov=False)
        return mean_arr[0], std_arr[0]

    def sample_prior(self, x_list, n_samp, full_cov=True):
        """Return samples from GP prior for each input in x_list."""
        mean, cov = self.get_prior_mean_cov(x_list, full_cov)
        x_list_sample_list = self.get_normal_samples(mean, cov, n_samp, full_cov)
        return x_list_sample_list

    def sample_prior_single(self, x, n_samp):
        """Return samples from GP prior for a single input x."""
        x_list_sample_list = self.sample_prior([x], n_samp)
        return x_list_sample_list[0]

    def sample_post(self, x_list, n_samp, full_cov=True):
        """
        Return samples from GP posterior for each input in x_list. Uses final
        hyperparameter sample in self.sample_list.
        """
        if len(self.data.x) == 0:
            return self.sample_prior(x_list, n_samp, full_cov)

        # If data is not empty:
        mean, cov = self.get_post_mean_cov(x_list, full_cov)
        x_list_sample_list = self.get_normal_samples(mean, cov, n_samp, full_cov)
        return x_list_sample_list

    def sample_post_single(self, x, n_samp):
        """
        Return samples from GP posterior for a single input x. Uses final
        hyperparameter sample in self.sample_list.
        """
        x_list_sample_list = self.sample_post([x], n_samp, full_cov=False)
        return x_list_sample_list[0]

    def get_normal_samples(self, mean, cov, n_samp, full_cov):
        """Return normal samples."""
        if full_cov:
            sample_list = list(sample_mvn(mean, cov, n_samp))
        else:
            sample_list = list(
                np.random.normal(
                    mean.reshape(-1), cov.reshape(-1), size=(n_samp, len(mean))
                )
            )
        x_list_sample_list = list(np.stack(sample_list).T)
        return x_list_sample_list
