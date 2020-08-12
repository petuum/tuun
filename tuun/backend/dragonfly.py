"""
Code for using Dragonfly as the backend tuning system.
"""

from argparse import Namespace
import dragonfly

from .core import Backend


class DragonflyBackend(Backend):
    """Class for Dragonfly as the backend tuning system."""

    def __init__(
        self, domain_config=None, opt_config=None, dragonfly_config=None,
    ):
        """
        Parameters
        ----------
        domain_config : dict
            Config to specify Dragonfly acqoptimizer domain.
        opt_config : dict
            Config to specify Dragonfly optimizer details.
        dragonfly_config : dict
            Config for other Dragonfly details.
        """
        self.domain_config = domain_config
        self.opt_config = opt_config
        self.dragonfly_config = dragonfly_config

    def minimize_function(self, f, n_iter=10, verbose=True):
        """
        Run Dragonfly Bayesian optimization to minimize function f.

        Parameters
        ----------
        f : function
            Function to optimize.
        n_iter : int
            Number of iterations of Bayesian optimization.
        verbose : bool
            If True, print information.
        """

        domain = self._get_domain()
        opt_method = 'bo'

        opt_val, opt_pt, history = dragonfly.minimise_function(
            func=f, domain=domain, max_capital=n_iter, opt_method=opt_method
        )
        results = Namespace(opt_val=opt_val, opt_pt=opt_pt, history=history)

        if verbose:
            vals = results.history.query_vals
            min_vals_idx = min(range(len(vals)), key=lambda x: vals[x])

            print('Minimum y =  {}'.format(results.opt_val))
            print('Minimizer x =  {}'.format(results.opt_pt))
            print('Found at iter =  {}'.format(min_vals_idx + 1))

        return results

    def suggest_to_minimize(self, data=None, verbose=True):
        """
        Run Dragonfly AcqOptDesigner to suggest a design (i.e. a point to evaluate).

        Parameters
        ----------
        data : dict
            Dictionary with keys x (list) and y (1D numpy ndarray).
        verbose : bool
            If True, print information.
        """
        opt = self._get_opt()
        self._tell_opt_data(opt, data)
        suggestion = opt.ask()

        if verbose:
            print('Suggestion: {}'.format(suggestion))

        return suggestion

    def _get_domain(self):
        """
        Return Dragonfly domain based on self.domain_config.
        """
        name = self.domain_config['name']
        assert name in ['euc']

        if name == 'euc':
            bounds_list = self.domain_config['bounds_list']
            domain = dragonfly.exd.domains.EuclideanDomain(bounds_list)

        return domain

    def _get_opt(self):
        """
        Return Dragonfly optimizer based on self.opt_config.
        """
        name = self.opt_config['name']
        assert name in ['euc']

        domain = self._get_domain()
        func_caller = dragonfly.exd.experiment_caller.EuclideanFunctionCaller(
            None, domain
        )
        opt = dragonfly.opt.gp_bandit.EuclideanGPBandit(func_caller, ask_tell_mode=True)
        opt.initialise()

        return opt

    def _tell_opt_data(self, opt, data):
        """Tell opt all elements in data."""
        tell_list = [(data['x'][i], data['y'][i]) for i in range(len(data['x']))]
        opt.tell(tell_list)
