"""
Acquisition function definitions and approximations.
"""

from argparse import Namespace
import numpy as np
from scipy.stats import norm, mstats


class Acquisitioner:
    """Class to manage acquisition function definitions and approximations."""

    def __init__(self, data, params=None, verbose=True):
        """
        Parameters
        ----------
        data : Namespace
            Namespace of initial data.
        params : Namespace
            Namespace of parameters.
        verbose : bool
            If True, print description string.
        """
        self.data = data
        self._set_params(params)
        self._set_acqfn()
        self._set_verbose(verbose)

    def _set_params(self, params):
        """Set parameters for the Acquisitioner."""
        self.params = Namespace()
        self.params.acq_str = getattr(params, 'acq_str', 'ei')
        self.params.ypred_str = getattr(params, 'ypred_str', 'sample')

    def _set_acqfn(self):
        """Set the acquisition method."""
        if self.params.acq_str == 'ei':
            self.acqfn = self._ei
        elif self.params.acq_str == 'pi':
            self.acqfn = self._pi
        elif self.params.acq_str == 'ucb':
            self.acqfn = self._ucb
        elif self.params.acq_str == 'ts':
            self.acqfn = self._ts
        elif self.params.acq_str == 'mean':
            self.acqfn = self._mean
        elif self.params.acq_str == 'rand':
            self.acqfn = self._rand
        elif self.params.acq_str == 'null':
            self.acqfn = self._null
        else:
            raise ValueError(
                'Incorrect self.params.acq_str: {}'.format(self.params.acq_str)
            )

    def _set_verbose(self, verbose):
        """Set verbose options."""
        self.verbose = verbose
        if self.verbose:
            self._print_str()

    def _ei(self, ypred):
        """Expected improvement (EI)."""
        if self.params.ypred_str == 'sample':
            return self._ppl_ei(ypred)

    def _pi(self, ypred):
        """Probability of improvement (PI)."""
        if self.params.ypred_str == 'sample':
            return self._ppl_pi(ypred)

    def _ucb(self, ypred, beta=0.5):
        """Upper (lower) confidence bound (UCB)."""
        if self.params.ypred_str == 'sample':
            return self._ppl_ucb(ypred, beta)

    def _ts(self, ypred):
        """Thompson sampling (TS)."""
        if self.params.ypred_str == 'sample':
            return self._ppl_ts(ypred)

    def _mean(self, ypred):
        """Mean of posterior predictive."""
        if self.params.ypred_str == 'sample':
            return self._ppl_mean(ypred)

    def _rand(self, ypred):
        """Uniform random sampling."""
        return np.random.random()

    def _null(self, ypred):
        """Return constant 0."""
        return 0.0

    # PPL Acquisition Functions
    def _ppl_ei(self, ypred_samp, normal=True):
        """
        PPL-EI: PPL acquisition function algorithm for expected improvement (EI).

        Parameters
        ----------
        ypred_samp : ndarray
            A numpy ndarray with shape=(nsamp,).
        normal : bool
            If true, assume ypred_samp are Gaussian distributed.

        Returns
        -------
        float
            PPL-EI acquisition function value.
        """
        youts = np.array(ypred_samp).flatten()
        nsamp = youts.shape[0]
        y_min = self.data.y.min()
        if normal:
            mu = np.mean(youts)
            sig = np.std(youts)
            if sig < 1e-1:
                sig = 1e-1
            gam = (y_min - mu) / sig
            eiVal = -1 * sig * (gam * norm.cdf(gam) + norm.pdf(gam))
        else:
            diffs = y_min - youts
            ind_below_min = np.argwhere(diffs > 0)
            eiVal = (
                -1 * np.sum(diffs[ind_below_min]) / float(nsamp)
                if len(ind_below_min) > 0
                else 0
            )
        return eiVal

    def _ppl_pi(self, ypred_samp, normal=True):
        """
        PPL-PI: PPL acquisition function algorithm for probability of improvement (PI).

        Parameters
        ----------
        ypred_samp : ndarray
            A numpy ndarray with shape=(nsamp,).
        normal : bool
            If true, assume ypred_samp are Gaussian distributed.

        Returns
        -------
        float
            PPL-PI acquisition function value.
        """
        youts = np.array(ypred_samp).flatten()
        nsamp = youts.shape[0]
        y_min = self.data.y.min()
        if normal:
            mu = np.mean(youts)
            sig = np.std(youts)
            if sig < 1e-6:
                sig = 1e-6
            piVal = -1 * norm.logcdf(y_min, loc=mu, scale=sig)
        else:
            piVal = -1 * len(np.argwhere(youts < y_min)) / float(nsamp)
        return piVal

    def _ppl_ucb(self, ypred_samp, normal=True, beta=0.5):
        """
        PPL-UCB: PPL acquisition function algorithm for upper confidence bound (UCB).
        Note that this algorithm computes a lower confidence bound due to assumed
        minimization.

        Parameters
        ----------
        ypred_samp : ndarray
            A numpy ndarray with shape=(nsamp,).
        normal : bool
            If true, assume ypred_samp are Gaussian distributed.
        beta : float
            UCB tradeoff parameter.

        Returns
        -------
        float
            PPL-UCB acquisition function value.
        """
        youts = np.array(ypred_samp).flatten()
        if normal:
            ucb_val = np.mean(youts) - beta * np.std(youts)
        else:
            half_conf_mass = norm.cdf(beta / 2) - norm.cdf(-beta / 2)
            quantiles = mstats.mquantiles(
                youts, prob=[0.5 - half_conf_mass, 0.5, 0.5 + half_conf_mass]
            )
            ucb_val = quantiles[0]
        return ucb_val

    def _ppl_ts(self, ypred_samp):
        """
        PPL-TS: PPL acquisition function algorithm for Thompson sampling (TS).

        Parameters
        ----------
        ypred_samp : ndarray
            A numpy ndarray with shape=(nsamp,).

        Returns
        -------
        float
            PPL-TS acquisition function value.
        """
        return ypred_samp.mean()

    def _ppl_mean(self, ypred_samp):
        """
        PPL-mean: PPL acquisition function algorithm for the mean of the posterior
        predictive.

        Parameters
        ----------
        ypred_samp : ndarray
            A numpy ndarray with shape=(nsamp,).

        Returns
        -------
        float
            PPL-mean acquisition function value.
        """
        youts = np.array(ypred_samp).flatten()
        return np.mean(youts)

    def _print_str(self):
        """Print a description string."""
        print('*[INFO] ' + str(self))

    def __str__(self):
        return f'Acquisitioner with params={self.params}'
