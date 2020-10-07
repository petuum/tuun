"""
Code for using Dragonfly as the backend tuning system.
"""

from argparse import Namespace
import numpy as np
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

    def minimize_function(self, f, n_iter=10, verbose=True, seed=None):
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
        seed : int
            If not None, set the random seed to seed.
        """
        if seed is not None:
            np.random.seed(seed)

        domain = self._get_domain()
        opt_method = 'bo'
        parsed_config = self._get_parsed_config()
        options = self._get_options()

        opt_val, opt_pt, history = dragonfly.minimise_function(
            func=f,
            domain=domain,
            max_capital=n_iter,
            opt_method=opt_method,
            config=parsed_config,
            options=options,
        )
        results = Namespace(opt_val=opt_val, opt_pt=opt_pt, history=history)

        if verbose:
            vals = results.history.query_vals
            min_vals_idx = min(range(len(vals)), key=lambda x: vals[x])

            print('Minimum y =  {}'.format(results.opt_val))
            print('Minimizer x =  {}'.format(results.opt_pt))
            print('Found at iter =  {}'.format(min_vals_idx + 1))

        return results

    def suggest_to_minimize(self, data=None, verbose=True, seed=None):
        """
        Run Dragonfly AcqOptDesigner to suggest a design (i.e. a point to evaluate).

        Parameters
        ----------
        data : dict
            Dictionary with keys x (list) and y (1D numpy ndarray).
        verbose : bool
            If True, print information.
        seed : int
            If not None, set the random seed to seed.
        """
        if seed is not None:
            np.random.seed(seed)

        opt = self._get_opt()
        assert opt.options.num_init_evals == 0.0

        if isinstance(data, dict):
            data = Namespace(**data)

        if len(data.x) > 0:
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

        options = self._get_options()

        if name == 'real':
            domain = self._get_domain()
            func_caller = dragonfly.exd.experiment_caller.EuclideanFunctionCaller(
                None, domain
            )
            options = dragonfly.apis.api_utils.load_options_for_method(
                'bo', 'opt', domain, 'num_evals', options
            )
            opt = dragonfly.opt.gp_bandit.EuclideanGPBandit(
                func_caller, options=options, ask_tell_mode=True,
            )

        elif name == 'product':
            domain, domain_orderings = self._get_cpgp_domain_and_orderings()
            func_caller = dragonfly.exd.experiment_caller.CPFunctionCaller(
                None, domain, domain_orderings=domain_orderings
            )
            options = dragonfly.apis.api_utils.load_options_for_method(
                'bo', 'opt', domain, 'num_evals', options
            )
            opt = dragonfly.opt.gp_bandit.CPGPBandit(
                func_caller, options=options, ask_tell_mode=True,
            )

        opt.initialise()
        return opt

    def _tell_opt_data(self, opt, data):
        """Tell opt all elements in data."""
        # NOTE: To minimize, must multiply data.y by -1
        tell_list = [(data.x[i], -1 * data.y[i]) for i in range(len(data.x))]
        opt.tell(tell_list)

    def _get_domain(self):
        """Return Dragonfly domain based on self.domain_config."""
        name = self.domain_config['name']
        assert name in ['real', 'product']

        if name == 'real':
            bounds_list = self.domain_config['min_max']
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

    def _get_options(self):
        """Get additional Dragonfly options."""

        # Using options contained in self.dragonfly_config
        options_dict = self.dragonfly_config
        if isinstance(options_dict, Namespace):
            options_dict = vars(options_dict)

        if options_dict is not None:
            options = Namespace()
            if 'n_init_rs' in options_dict:
                options.init_capital = None
                options.num_init_evals = options_dict['n_init_rs']
            if 'acq_str' in options_dict:
                options.acq = options_dict['acq_str']
        else:
            options = None

        return options
