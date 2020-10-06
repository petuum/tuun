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

        # Tuun parameters
        self.config.seed = getattr(config, 'seed', None)
        self.config.print_x_str_len = getattr(config, 'print_x_str_len', 30)

        self.config.backend = getattr(config, 'backend', 'probo')
        assert self.config.backend in ['probo', 'dragonfly']

        # ProBO specific
        if self.config.backend == 'probo':

            domain_config = getattr(config, 'domain_config', None)
            if domain_config is None:
                domain_config = {'name': 'real', 'min_max': [(0.0, 10.0)]}
            self.config.domain_config = domain_config

            model_config = getattr(config, 'model_config', None)
            if model_config is None:
                model_config = {'name': 'gpytorchgp'}
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

        # Dragonfly specific
        if self.config.backend == 'dragonfly':

            domain_config = getattr(config, 'domain_config', None)
            if domain_config is None:
                domain_config = {'name': 'real', 'min_max': [(0.0, 10.0)]}
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
        Set the Tuun search space given a search_space_list, a list of tuples each
        containing a domain type and a domain bounds specification.  This method will
        overwrite self.config and self.backend.

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

        # Set default model_config
        self.config.model_config = {
            'name': 'simpleproductkernelgp',
            'ls': 3.0,
            'alpha': 1.5,
            'sigma': 1e-5,
            'domain_spec': domain_types,
        }

        # Set default acqoptimizer_config
        pao_config_list = []
        for dt in domain_types:
            if dt == 'list':
                pao_config_list.append({'name': 'default'})
            elif dt == 'real':
                pao_config_list.append(
                        {'name': 'cobyla', 'init_str': 'init_opt', 'jitter': False}
                )

        self.config.acqoptimizer_config = {
            'name': 'product',
            'n_iter_bcd': 3,
            'n_init_rs': 5,
            'pao_config_list': pao_config_list,
        }

        # Set default domain_config
        dom_config_list = []
        for i, dt in enumerate(domain_types):
            bounds_spec = search_space_list[i][1]
            if dt == 'list':
                dom_config_list.append({'name': dt, 'domain_list': bounds_spec})
            elif dt == 'real':
                dom_config_list.append({'name': dt, 'min_max': bounds_spec})

        self.config.domain_config = {
            'name': 'product', 'dom_config_list': dom_config_list,
        }

        # Set self.backend with above settings
        self._set_backend()

    def _transform_domain_config(self):
        """Transform domain."""

        # Normalize domain for probo backend: all 'real' blocks normalized to [0, 10.0].
        # TODO: also for 'int'
        if self.config.backend == 'probo':
            if self.config.domain_config['name'] == 'product':
                dom_config_list = self.config.domain_config['dom_config_list']
                for block_idx, domain_config in enumerate(dom_config_list):
                    if domain_config['name'] == 'real':
                        min_max = domain_config['min_max']
                        domain_config['min_max_init'] = min_max
                        domain_config['min_max'] = [[0, 10] for _ in min_max]
                self.config.normalize_real = True
                self._set_backend()

    def _transform_data(self, data, inverse=False):
        """Return transformed data Namespace."""

        # Transform data for probo backend
        if self.config.backend == 'probo':
            if self.config.domain_config['name'] == 'product':
                data.x = [self._transform_x(xi, inverse=inverse) for xi in data.x]
        return data

    def _transform_x(self, x, inverse=False):
        """Return transformed domain point x."""

        # Transform each block of x for probo backend
        if self.config.backend == 'probo':
            if self.config.domain_config['name'] == 'product':
                dom_config_list = self.config.domain_config['dom_config_list']
                for block_idx, domain_config in enumerate(dom_config_list):
                    x = self._transform_x_block(x, block_idx, inverse=inverse)

        return x

    def _transform_x_block(self, x, block_idx, inverse=False):
        """Return domain point x with one block transformed."""
        domain_config = self.config.domain_config['dom_config_list'][block_idx]

        # Normalize domain: all 'real' blocks normalized to [0, 10.0].
        # TODO: also for 'int'
        normalize_real = getattr(self.config, 'normalize_real', False)
        if domain_config['name'] == 'real' and normalize_real:
            for i, bounds in enumerate(domain_config['min_max_init']):
                if inverse is True:
                    scale_factor = (bounds[1] - bounds[0]) / 10.0
                    x[block_idx][i] = x[block_idx][i] * scale_factor + bounds[0]
                else:
                    scale_factor = 10.0 / (bounds[1] - bounds[0])
                    x[block_idx][i] = (x[block_idx][i] - bounds[0]) * scale_factor

        return x

    def suggest_to_minimize(self, data=None, verbose=True, seed=None):
        """
        Suggest a single design (i.e. a point to evaluate).

        Parameters
        ----------
        data : dict
            Dictionary with keys x (list) and y (1D numpy ndarray).
        verbose : bool
            If True, print information.
        seed : int
            If not None, set the random seed to seed.
        """
        if seed is None:
            seed = self.config.seed

        # Convert data to Namespace
        if isinstance(data, dict):
            data = Namespace(**data)

        # Set subseed (depends on data)
        if seed is None:
            seed = np.random.randint(13337)
        subseed = seed if data is None else seed + len(data.x)

        # Transform data
        data = self._transform_data(data)

        # Call backend suggest_to_minimize method
        suggestion = self.backend.suggest_to_minimize(
            data=data, verbose=verbose, seed=subseed
        )

        # Inverse transform data and suggestion
        data = self._transform_data(data, inverse=True)
        suggestion = self._transform_x(suggestion, inverse=True)

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
        if seed is None:
            seed = self.config.seed

        # Convert data to Namespace
        if isinstance(data, dict):
            data = Namespace(**data)

        # Transform domain
        self._transform_domain_config()

        if use_backend_minimize:
            result = self._run_backend_minimize_function()
        else:
            # Minimize function via calls to self.suggest_to_minimize()
            if data is None:
                data = Namespace(x=[], y=[])

            n_data_init = 0 if data is None else len(data.x)

            for i in range(n_iter):
                x = self.suggest_to_minimize(data=data, verbose=verbose, seed=seed)
                y = self._format_function_output(f(x))

                # Update data
                data.x.append(x)
                data.y.append(y)

                # Print iter info
                self._print_iter_info(i, data)

            self._print_final_info(data, n_data_init)
            result = self._get_final_result(data)

        return result

    def _format_function_output(self, y_out):
        """Format output of function query."""

        # Ensure function output y_out is a float
        y_out = float(y_out)
        return y_out

    def _print_iter_info(self, iter_idx, data):
        """Print information for a given iteration of Bayesian optimization."""
        x_last = data.x[-1]
        x_str = self._get_print_x_str(x_last)
        y = data.y[-1]
        bsf = np.min(data.y)
        print(
            'i: {},    x: {} \ty: {:.4f},\tBSF: {:.4f}'.format(iter_idx, x_str, y, bsf)
        )

    def _get_print_x_str(self, x):
        """Return formatted string for x to print at each iter."""
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
