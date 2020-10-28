"""
Main interface for Tuun.
"""
from argparse import Namespace
import numpy as np

from .backend import ProboBackend, DragonflyBackend


class Tuun:
    """Main interface to Tuun."""

    def __init__(self, config_dict):
        """
        Parameters
        ----------
        config_dict : dict
            Config to specify Tuun options.
        """
        self._configure_tuun(config_dict)
        self._set_backend()

    def _configure_tuun(self, config_dict):
        """Configure Tuun based on config_dict and defaults."""
        config = Namespace(**config_dict)
        self.config = Namespace()

        # Set Tuun parameters
        self._set_seed(config=config)
        self.config.print_x_str_len = getattr(config, 'print_x_str_len', 30)
        self.config.normalize_real = getattr(config, 'normalize_real', False)

        # Set backend
        self.config.backend = getattr(config, 'backend', 'probo')
        assert self.config.backend in ['probo', 'dragonfly']

        # Set ProBO specific
        if self.config.backend == 'probo':
            self._configure_tuun_for_probo(config)

        # Set Dragonfly specific
        if self.config.backend == 'dragonfly':
            self._configure_tuun_for_dragonfly(config)

    def _set_seed(self, config=None, seed=None):
        """Set self.config.seed and numpy random seed."""
        if seed is not None:
            self.config.seed = seed
        elif config is not None:
            self.config.seed = getattr(config, 'seed', None)

        if not getattr(self.config, 'seed', None):
            self.config.seed = np.random.randint(13337)

        np.random.seed(self.config.seed)

    def _configure_tuun_for_probo(self, config):
        """Configure Tuun for Probo backend."""
        domain_config = getattr(config, 'domain_config', None)
        if domain_config is None:
            domain_config = {'name': 'real', 'min_max': [[0.0, 10.0]]}
        self.config.domain_config = domain_config

        model_config = getattr(config, 'model_config', None)
        if model_config is None:
            model_config = {'name': 'simplegp'}
        self.config.model_config = model_config

        acqfunction_config = getattr(config, 'acqfunction_config', None)
        if acqfunction_config is None:
            acqfunction_config = {'name': 'default', 'acq_str': 'ucb', 'n_gen': 500}
        self.config.acqfunction_config = acqfunction_config

        acqoptimizer_config = getattr(config, 'acqoptimizer_config', None)
        if acqoptimizer_config is None:
            acqoptimizer_config = {'name': 'default'}
        self.config.acqoptimizer_config = acqoptimizer_config

        probo_config = getattr(config, 'probo_config', None)
        self.config.probo_config = probo_config

    def _configure_tuun_for_dragonfly(self, config):
        """Configure Tuun for Dragonfly backend."""
        domain_config = getattr(config, 'domain_config', None)
        if domain_config is None:
            domain_config = {'name': 'real', 'min_max': [[0.0, 10.0]]}
        self.config.domain_config = domain_config

        opt_config = getattr(config, 'opt_config', None)
        if opt_config is None:
            opt_config = {'name': domain_config['name']}
        self.config.opt_config = opt_config

        dragonfly_config = getattr(config, 'dragonfly_config', None)
        self.config.dragonfly_config = dragonfly_config

    def _set_backend(self):
        """Set Tuun backend tuning system."""
        if self.config.backend == 'probo':
            self.backend = ProboBackend(
                model_config=self.config.model_config,
                acqfunction_config=self.config.acqfunction_config,
                acqoptimizer_config=self.config.acqoptimizer_config,
                domain_config=self.config.domain_config,
                probo_config=self.config.probo_config,
            )
        elif self.config.backend == 'dragonfly':
            self.backend = DragonflyBackend(
                domain_config=self.config.domain_config,
                opt_config=self.config.opt_config,
                dragonfly_config=self.config.dragonfly_config,
            )

    def set_config_from_list(self, search_space_list):
        """
        Automatically configure the Tuun search space given a search_space_list (a list
        of tuples each containing a domain type and a domain bounds specification).
        This method will overwrite of self.config and self.backend.

        Parameters
        ----------
        search_space_list : list
            List of tuples, where each element tuple corresponds to a separate
            optimization block which can have a unique domain type. Each element tuple
            consists of (domain type, domain bounds specification).
        """
        domain_types = [ss[0] for ss in search_space_list]
        assert all([dt in ['real', 'list'] for dt in domain_types])

        # Set backend
        self.config.backend = 'probo'

        # Set model_config
        self._set_model_config_from_list(domain_types)

        # Set acqoptimizer_config
        self._set_acqoptimizer_config_from_list(domain_types)

        # Set domain_config
        self._set_domain_config_from_list(search_space_list)

        # Set self.backend given the above updates
        self._set_backend()

    def _set_model_config_from_list(self, domain_types):
        """
        Helper function for self.set_config_from_list to update self.config.model_config
        attribute.
        """
        if self.config.model_config.get('name') == 'standistmatgp':
            self.config.model_config = {
                'name': 'standistmatgp',
                'model_str': 'optfixedsig',
                'ig1': 4.0,
                'ig2': 3.0,
                'n1': 1.0,
                'n2': 1.0,
                'sigma': 1e-5,
                'niter': 70,
                'domain_spec': domain_types,
            }
        else:
            self.config.model_config = {
                'name': 'simpleproductkernelgp',
                'ls': 3.0,
                'alpha': 1.85,
                'sigma': 1e-5,
                'domain_spec': domain_types,
            }

    def _set_acqoptimizer_config_from_list(self, domain_types):
        """
        Helper function for self.set_config_from_list to update
        self.config.acqoptimizer_config attribute.
        """
        n_init_rs = self.config.acqfunction_config.get('n_init_rs', 5)

        if len(domain_types) == 1:
            dt = domain_types[0]
            ao_config = self._get_acqoptimizer_config_from_domain_type(dt)
            self.config.acqoptimizer_config = ao_config
        else:
            pao_config_list = []
            for dt in domain_types:
                ao_config = self._get_acqoptimizer_config_from_domain_type(dt, True)
                pao_config_list.append(ao_config)
            self.config.acqoptimizer_config = {
                'name': 'product',
                'n_iter_bcd': 3,
                'n_init_rs': n_init_rs,
                'pao_config_list': pao_config_list,
            }

    def _get_acqoptimizer_config_from_domain_type(self, dt, in_product=False):
        """
        Helper function for self.set_acqoptimizer_config_from_list which returns
        acqoptimizer config for a single domain type dt.
        """
        assert dt in ['real', 'list']
        jitter = self.config.acqoptimizer_config.get('jitter', False)
        n_init_rs = self.config.acqoptimizer_config.get('n_init_rs', 5)
        acqoptimizer_name = self.config.acqoptimizer_config.get('name')

        if dt == 'real':
            if acqoptimizer_name == 'neldermead':
                ao_config = {
                    'name': 'neldermead',
                    'rand_every': 10,
                    'max_iter': 200,
                    'init_str': 'bsf',
                    'jitter': jitter,
                    'n_init_rs': n_init_rs,
                }
            else:
                ao_config = {
                    'name': 'cobyla',
                    'init_str': 'bsf',
                    'rand_every': 4,
                    'jitter': jitter,
                    'n_init_rs': n_init_rs,
                }

            if in_product:
                ao_config['init_str'] = 'init_opt'
                ao_config['rand_every'] = None
                ao_config['jitter'] = False
                ao_config['n_init_rs'] = 0

        elif dt == 'list':
            ao_config = {'name': 'default'}

        return ao_config

    def _set_domain_config_from_list(self, search_space_list):
        """
        Helper function for self.set_config_from_list to update
        self.config.domain_config attribute.
        """
        domain_types = [ss[0] for ss in search_space_list]
        if len(domain_types) == 1:
            dt = domain_types[0]
            bounds_spec = search_space_list[0][1]
            dom_config = self._get_domain_config_from_domain_type(dt, bounds_spec)
            self.config.domain_config = dom_config
        else:
            dom_config_list = []
            for i, dt in enumerate(domain_types):
                bounds_spec = search_space_list[i][1]
                dom_config = self._get_domain_config_from_domain_type(dt, bounds_spec)
                dom_config_list.append(dom_config)

            self.config.domain_config = {
                'name': 'product', 'dom_config_list': dom_config_list,
            }

    def _get_domain_config_from_domain_type(self, dt, bounds_spec):
        """
        Helper function for self.set_domain_config_from_list which returns
        domain config for a single domain type dt with given bounds_spec.
        """
        assert dt in ['real', 'list']

        if dt == 'real':
            dom_config = {'name': dt, 'min_max': bounds_spec}
        elif dt == 'list':
            dom_config = {'name': dt, 'domain_list': bounds_spec}

        return dom_config

    def _transform_domain_config(self):
        """Transform domain."""

        # Normalize
        if self.config.domain_config['name'] == 'product':
            for domain_config in self.config.domain_config['dom_config_list']:
                domain_config = self._normalize_domain_config_block(domain_config)
        else:
            domain_config = self.config.domain_config
            domain_config = self._normalize_domain_config_block(domain_config)

        # Reset self.backend
        self._set_backend()

    def _normalize_domain_config_block(self, domain_config):
        """Return domain config, possibly normalized to [0, 10]."""
        normalize_real = getattr(self.config, 'normalize_real', False)
        if domain_config['name'] == 'real' and normalize_real:
            domain_config['min_max_init'] = domain_config['min_max']
            domain_config['min_max'] = [[0, 10] for _ in domain_config['min_max']]
        return domain_config

    def _transform_data(self, data, inverse=False):
        """Return transformed data Namespace."""
        data.x = [self._transform_x(xi, inverse=inverse) for xi in data.x]
        return data

    def _transform_x(self, x, inverse=False):
        """Return transformed domain point x."""
        if self.config.domain_config['name'] == 'product':
            dom_config_list = self.config.domain_config['dom_config_list']
            for x_block, dom_config in zip(x, dom_config_list):
                x_block = self._normalize_x_block(x_block, dom_config, inverse=inverse)
        else:
            x = self._normalize_x_block(x, self.config.domain_config, inverse=inverse)
        return x

    def _normalize_x_block(self, x, domain_config, inverse=False):
        """Return x, possibly normalized to [0, 10]."""
        normalize_real = getattr(self.config, 'normalize_real', False)
        if domain_config['name'] == 'real' and normalize_real:
            for i, bounds in enumerate(domain_config['min_max_init']):
                if inverse is True:
                    scale_factor = (bounds[1] - bounds[0]) / 10.0
                    x[i] = x[i] * scale_factor + bounds[0]
                else:
                    scale_factor = 10.0 / (bounds[1] - bounds[0])
                    x[i] = (x[i] - bounds[0]) * scale_factor
        return x

    def suggest_to_minimize(self, data=None, verbose=True, seed=None):
        """
        Suggest a single design (i.e. a point to evaluate) to minimize.

        Parameters
        ----------
        data : dict
            Dictionary with keys x (list) and y (1D numpy ndarray).
        verbose : bool
            If True, print information.
        seed : int
            If not None, set the random seed to seed.
        """
        self._set_seed(seed=seed)
        data = self._format_data_input(data)

        # Transform data
        data = self._transform_data(data)

        # Set data-dependent subseed
        subseed = int(np.random.uniform(12345 * (len(data.x) + 1)))

        # Call backend suggest_to_minimize method
        suggestion = self.backend.suggest_to_minimize(
            data=data, verbose=verbose, seed=subseed
        )

        # Inverse transform data and suggestion
        data = self._transform_data(data, inverse=True)
        suggestion = self._transform_x(suggestion, inverse=True)

        return suggestion

    def suggest_to_maximize(self, data=None, verbose=True, seed=None):
        """
        Suggest a single design (i.e. a point to evaluate) to maximize.

        Parameters
        ----------
        data : dict
            Dictionary with keys x (list) and y (1D numpy ndarray).
        verbose : bool
            If True, print information.
        seed : int
            If not None, set the random seed to seed.
        """
        data = self._format_data_input(data)
        data.y = [-1*v for v in data.y]
        suggestion = self.suggest_to_minimize(data, verbose, seed)
        return suggestion

    def minimize_function(
        self,
        f,
        n_iter=10,
        data=None,
        data_update_fun=None,
        use_backend_minimize=False,
        verbose=False,
        seed=None,
    ):
        """
        Run tuning system to minimize function f.

        Parameters
        ----------
        f : function
            Function to optimize.
        n_iter : int
            Number of iterations of Bayesian optimization.
        data : dict
            Dictionary with keys x (list) and y (1D numpy ndarray).
        data_update_fun : function
            Function that will update internal dataset given a tuple (x, y)
        use_backend_minimize : bool
            If True, use self.backend.minimize_function method. Otherwise, will proceed
            via calls to self.suggest_to_minimize().
        verbose : bool
            If True, print information.
        seed : int
            If not None, set the random seed to seed.
        """
        self._set_seed(seed=seed)
        data = self._format_data_input(data)

        # Transform domain
        self._transform_domain_config()

        if use_backend_minimize:
            result = self._run_backend_minimize_function()
        else:
            # Minimize function via calls to self.suggest_to_minimize()
            n_data_init = 0 if data is None else len(data.x)

            # The BO Loop
            self._print_starting_info()
            for i in range(n_iter):
                x = self.suggest_to_minimize(data=data, verbose=verbose)
                y = self._format_function_output(f(x))

                # Update data
                data.x.append(x)
                data.y.append(y)

                # Print iter info
                self._print_iter_info(i, data)

            self._print_final_info(data, n_data_init)
            result = self._get_final_result(data)

        return result

    def _format_data_input(self, data):
        """Format and return data Namespace."""
        if data is None:
            data = Namespace(x=[], y=[])

        if isinstance(data, dict):
            data = Namespace(**data)

        if not hasattr(data, 'x'):
            raise Exception('Input data must contain x, a list.')

        if not hasattr(data, 'y'):
            raise Exception('Input data must contain y, a 1d numpy ndarray.')

        return data

    def _format_function_output(self, y_out):
        """Format output of function query."""

        # Ensure function output y_out is a float
        y_out = float(y_out)
        return y_out

    def _print_starting_info(self):
        """Print information before optimiztion run."""
        print(
            '*[KEY] i: iteration, x: design, y: objective, min_y: minimum '
            + 'objective so far (* indicates a new min_y)'
        )

    def _print_iter_info(self, iter_idx, data):
        """Print information for a given iteration of Bayesian optimization."""
        x_str = self._get_print_x_str(data.x[-1])
        y = data.y[-1]
        min_y = np.min(data.y)

        if len(data.y) > 1:
            ast_str = '*' if min_y != np.min(data.y[:-1]) else ''
        else:
            ast_str = '*'

        print(
            'i: {},    x: {} \ty: {:.4f},\tmin_y: {:.4f} {}'.format(
                iter_idx, x_str, y, min_y, ast_str
            )
        )

    def _get_print_x_str(self, x):
        """Return formatted x string to print at each iter."""
        x_str = str(x).ljust(self.config.print_x_str_len)
        x_str = x_str[: self.config.print_x_str_len] + '..'
        return x_str

    def _print_final_info(self, data, n_data_init):
        """Print final information after minimization is complete."""
        min_idx = np.argmin(data.y)
        min_x = data.x[min_idx]
        min_y = data.y[min_idx]
        print('Minimum y = {}'.format(min_y))
        print('Minimizer x = {}'.format(min_x))
        print('Found at i = {}'.format(min_idx - n_data_init))

    def _get_final_result(self, data):
        """Return final result of a run of minimization."""
        result = Namespace()
        result.min_idx = np.argmin(data.y)
        result.min_x = data.x[result.min_idx]
        result.min_y = data.y[result.min_idx]
        result.data = data
        return result

    def _run_backend_minimize_function(f, n_iter, data, data_update_fun, verbose, seed):
        """Run self.backend.minimize_function method."""

        if self.config.backend == 'dragonfly':
            if data is not None:
                #  TODO: print warning
                pass

            if data_update_fun is not None:
                #  TODO: print warning
                pass

            result = self.backend.minimize_function(
                f=f, n_iter=n_iter, verbose=verbose, seed=seed
            )

        elif self.config.backend == 'probo':
            result = self.backend.minimize_function(
                f=f,
                n_iter=n_iter,
                data=data,
                data_update_fun=data_update_fun,
                verbose=verbose,
                seed=seed,
            )

        return result
