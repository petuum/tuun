"""
Main interface for Tuun.
"""
from argparse import Namespace

from .backend import ProboBackend, DragonflyBackend


class Tuun:
    """Class for Tuun."""

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

        assert config.backend in ['probo', 'dragonfly']
        self.config.backend = getattr(config, 'backend', 'probo')

        # ProBO specific
        if config.backend == 'probo':

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
        if config.backend == 'dragonfly':

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
        elif self.config.bakend == 'dragonfly':
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

        suggestion = self.backend.suggest_to_minimize(
            data=data, verbose=verbose, seed=seed
        )
        return suggestion

    def minimize_function(
        self, f, n_iter=10, data=None, data_update_fun=None, verbose=True, seed=None
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
        verbose : bool
            If True, print information.
        seed : int
            If not None, set the random seed to seed.
        """
        if seed is None:
            seed = self.config.seed

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
