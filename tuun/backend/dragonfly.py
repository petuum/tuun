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
        parsed_config = self._get_parsed_config()

        opt_val, opt_pt, history = dragonfly.minimise_function(
            func=f,
            domain=domain,
            max_capital=n_iter,
            opt_method=opt_method,
            config=parsed_config,
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

    def _get_opt(self):
        """
        Return Dragonfly optimizer based on self.opt_config.
        """
        name = self.opt_config['name']
        assert name in ['real', 'product']

        if name == 'real':
            domain = self._get_domain()
            func_caller = dragonfly.exd.experiment_caller.EuclideanFunctionCaller(
                None, domain
            )
            opt = dragonfly.opt.gp_bandit.EuclideanGPBandit(func_caller, ask_tell_mode=True)

        elif name == 'product':
            domain, domain_orderings = self._get_cpgp_domain_and_orderings()
            func_caller = dragonfly.exd.experiment_caller.CPFunctionCaller(
                None, domain, domain_orderings=domain_orderings
            )
            opt = dragonfly.opt.gp_bandit.CPGPBandit(func_caller, ask_tell_mode=True)

        opt.initialise()
        return opt

    def _tell_opt_data(self, opt, data):
        """Tell opt all elements in data."""
        tell_list = [(data['x'][i], data['y'][i]) for i in range(len(data['x']))]
        opt.tell(tell_list)

    def _get_domain(self):
        """Return Dragonfly domain based on self.domain_config."""
        name = self.domain_config['name']
        assert name in ['real', 'product']

        if name == 'real':
            bounds_list = self.domain_config['bounds_list']
            domain = dragonfly.exd.domains.EuclideanDomain(bounds_list)
        elif name == 'product':
            domain, _ = self._get_cpgp_domain_and_orderings()

        return domain

    def _get_parsed_config(self):
        """Return Dragonfly parsed config."""
        name = self.domain_config['name']
        assert name in ['real', 'product']

        if name == 'real':
            parsed_config = None
        elif name == 'product':
            parsed_config = self._get_cpgp_parsed_config()

        return parsed_config

    def _get_cpgp_domain_and_orderings(self):
        """
        Return Dragonfly domain and domain_orderings based on self.domain_config.
        """
        name = self.domain_config['name']
        assert name in ['product']

        if name == 'product':
            parsed_config = self._get_cpgp_parsed_config()
            domain = parsed_config.domain
            domain_orderings = parsed_config.domain_orderings

        return domain, domain_orderings

    def _get_cpgp_parsed_config(self):
        """Parse self.domain_config into correct format for dragonfly."""

        name = self.domain_config['name']
        assert name in ['product']

        if name == 'product':
            parsed_dom_dict_list = []
            dom_config_list = self.domain_config['dom_config_list']
            for dom_idx, dom_config in enumerate(dom_config_list):
                parsed_dom_dict = {}
                if dom_config['name'] == 'real':
                    parsed_dom_dict['name'] = 'float' + str(dom_idx)
                    parsed_dom_dict['type'] = 'float'
                    min_max = dom_config['min_max']
                    assert len(min_max) == 1
                    parsed_dom_dict['min'] = min_max[0][0]
                    parsed_dom_dict['max'] = min_max[0][1]

                parsed_dom_dict_list.append(parsed_dom_dict)

            parsed_config = {}
            parsed_config['name'] = 'domain_config'
            parsed_config['domain'] = parsed_dom_dict_list

        parsed_config = dragonfly.exd.cp_domain_utils.load_config(parsed_config)
        return parsed_config
