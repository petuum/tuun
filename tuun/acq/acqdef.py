"""
Acquisition function definitions and approximations.
"""

from argparse import Namespace
import numpy as np
from scipy.stats import norm, mstats


class Acquisitioner:
    """
    Class to manage acquisition function definitions and approximations.
    """

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
        self.set_params(params)
        self.set_acqfn()
        self.set_verbose(verbose)

    def set_params(self, params):
        """Set parameters for the Acquisitioner."""
        self.params = Namespace()
        self.params.acq_str = getattr(params, 'acq_str', 'ei')
        self.params.pmout_str = getattr(params, 'pmout_str', 'sample')

    def set_acqfn(self):
        """Set the acquisition method."""
        if self.params.acq_str == 'ei':
            self.acqfn = self.ei
        if self.params.acq_str == 'pi':
            self.acqfn = self.pi
        if self.params.acq_str == 'ts':
            self.acqfn = self.ts
        if self.params.acq_str == 'ucb':
            self.acqfn = self.ucb
        if self.params.acq_str == 'mean':
            self.acqfn = self.mean
        if self.params.acq_str == 'rand':
            self.acqfn = self.rand
        if self.params.acq_str == 'null':
            self.acqfn = self.null

    def set_verbose(self, verbose):
        """Set verbose options."""
        self.verbose = verbose
        if self.verbose:
            self.print_str()

    def ei(self, pmout):
        """Expected improvement (EI)."""
        if self.params.pmout_str == 'sample':
            return self.ppl_acq_ei(pmout)

    def pi(self, pmout):
        """Probability of improvement (PI)."""
        if self.params.pmout_str == 'sample':
            return self.ppl_acq_pi(pmout)

    def ucb(self, pmout):
        """Upper (lower) confidence bound (UCB)."""
        if self.params.pmout_str == 'sample':
            return self.ppl_acq_ucb(pmout)

    def ts(self, pmout):
        """Thompson sampling (TS)."""
        if self.params.pmout_str == 'sample':
            return self.ppl_acq_ts(pmout)

    def mean(self, pmout):
        """Mean of posterior predictive."""
        if self.params.pmout_str == 'sample':
            return self.ppl_acq_mean(pmout)

    def rand(self, pmout):
        """Uniform random sampling."""
        return np.random.random()

    def null(self, pmout):
        """Return constant 0."""
        return 0.0

    # PPL Acquisition Functions
    def ppl_acq_ei(self, pmout_samp, normal=True):
        """
        PPL-EI: PPL acquisition function algorithm for expected improvement
        (EI).

        Parameters
        ----------
        pmout_samp : ndarray
            A numpy ndarray with shape=(nsamp,).
        normal : bool
            If true, assume pmout_samp are Gaussian distributed.

        Returns
        -------
        float
            PPL-EI acquisition function value.
        """
        youts = np.array(pmout_samp).flatten()
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

    def ppl_acq_pi(self, pmout_samp, normal=True):
        """
        PPL-PI: PPL acquisition function algorithm for probability of
        improvement (PI).

        Parameters
        ----------
        pmout_samp : ndarray
            A numpy ndarray with shape=(nsamp,).
        normal : bool
            If true, assume pmout_samp are Gaussian distributed.

        Returns
        -------
        float
            PPL-PI acquisition function value.
        """
        youts = np.array(pmout_samp).flatten()
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

    def ppl_acq_ucb(self, pmout_samp, normal=True, beta=0.5):
        """
        PPL-UCB: PPL acquisition function algorithm for upper confidence bound
        (UCB). Note that this algorithm computes a lower confidence bound due
        to assumed minimization.

        Parameters
        ----------
        pmout_samp : ndarray
            A numpy ndarray with shape=(nsamp,).
        normal : bool
            If true, assume pmout_samp are Gaussian distributed.
        beta : float
            UCB tradeoff parameter.

        Returns
        -------
        float
            PPL-UCB acquisition function value.
        """
        youts = np.array(pmout_samp).flatten()
        if normal:
            ucb_val = np.mean(youts) - beta * np.std(youts)
        else:
            half_conf_mass = norm.cdf(beta / 2) - norm.cdf(-beta / 2)
            quantiles = mstats.mquantiles(
                youts, prob=[0.5 - half_conf_mass, 0.5, 0.5 + half_conf_mass]
            )
            ucb_val = quantiles[0]
        return ucb_val

    def ppl_acq_ts(self, pmout_samp):
        """
        PPL-TS: PPL acquisition function algorithm for Thompson sampling (TS).

        Parameters
        ----------
        pmout_samp : ndarray
            A numpy ndarray with shape=(nsamp,).

        Returns
        -------
        float
            PPL-TS acquisition function value.
        """
        return pmout_samp.mean()

    def ppl_acq_mean(self, pmout_samp):
        """
        PPL-Mean: PPL acquisition function algorithm for the mean of the posterior
        predictive.

        Parameters
        ----------
        pmout_samp : ndarray
            A numpy ndarray with shape=(nsamp,).

        Returns
        -------
        float
            PPL-Mean acquisition function value.
        """
        youts = np.array(pmout_samp).flatten()
        return np.mean(youts)

    # Utilities
    def print_str(self):
        """Print a description string."""
        print('*Acquisitioner with params={}'.format(self.params))
