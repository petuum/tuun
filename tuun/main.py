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

        self.config.backend = getattr(config, 'backend', 'probo')
        assert self.config.backend in ['probo', 'dragonfly']

        # ProBO specific
        if self.config.backend == 'probo':

            domain_config = getattr(config, 'domain_config', None)
            assert domain_config is not None
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
            assert domain_config is not None
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

        # Set subseed (depends on data)
        if seed is None:
            seed = np.random.randint(13337)
        subseed = seed if data is None else seed + len(data.x)

        # Call backend suggest_to_minimize method
        suggestion = self.backend.suggest_to_minimize(
            data=data, verbose=verbose, seed=subseed
        )
        return suggestion

    def minimize_function(
        self,
        f,
        n_iter=10,
        data=None,
        data_update_fun=None,
        use_backend_minimize=False,
        verbose=True,
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

        if use_backend_minimize:
            result = self._run_backend_minimize_function()
        else:
            # Minimize function via calls to self.suggest_to_minimize()
            if data is None:
                data = Namespace(x=[], y=[])

            n_data_init = 0 if data is None else len(data.x)

            for i in range(n_iter):
                x = self.suggest_to_minimize(data=data, verbose=False, seed=seed)
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
        """Print information for a given iteration of minimization."""
        x_str_max_len = 14
        x_str = str(data.x[-1])
        x_str = x_str[: min(len(x_str), x_str_max_len)]
        y = data.y[-1]
        bsf = np.min(data.y)
        print(
            'i: {},    x: {},\ty: {:.4f},\tBSF: {:.4f}'.format(iter_idx, x_str, y, bsf)
        )

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
