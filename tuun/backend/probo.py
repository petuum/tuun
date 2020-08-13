"""
Code for using ProBO as the backend tuning system.
"""

import tuun.probo as probo
from .core import Backend


class ProboBackend(Backend):
    """Class for ProBO as the backend tuning system."""

    def __init__(
        self,
        model_config=None,
        acqfunction_config=None,
        acqoptimizer_config=None,
        domain_config=None,
        probo_config=None,
    ):
        """
        Parameters
        ----------
        model_config : dict
            Config to specify ProBO model.
        acqfunction_config : dict
            Config to specify ProBO acqfunction.
        acqoptimizer_config : dict
            Config to specify ProBO acqoptimizer.
        domain_config : dict
            Config to specify ProBO acqoptimizer domain.
        probo_config : dict
            Config for other ProBO details.
        """
        self.model_config = model_config
        self.acqfunction_config = acqfunction_config
        self.acqoptimizer_config = acqoptimizer_config
        self.domain_config = domain_config
        self.probo_config = probo_config

    def minimize_function(
        self, f, n_iter=10, data=None, data_update_fun=None, verbose=True, seed=None
    ):
        """
        Run ProBO Bayesian optimization to minimize function f.

        Parameters
        ----------
        f : function
            Function to optimize.
        n_iter : int
            Number of iterations of Bayesian optimization.
        data : dict
            Dictionary with keys x (list) and y (1D numpy ndarray).
        data_update_fun : function
            Function that will update ProBO's dataset given a tuple (x, y)
        verbose : bool
            If True, print information.
        seed : int
            If not None, set ProBO random seed to seed.
        """
        model = self._get_model()
        acqfunction = self._get_acqfunction()
        acqoptimizer = self._get_acqoptimizer()

        bo_params = {'n_iter': n_iter}

        bo = probo.SimpleBo(
            f=f,
            model=model,
            acqfunction=acqfunction,
            acqoptimizer=acqoptimizer,
            data=data,
            data_update_fun=data_update_fun,
            params=bo_params,
            verbose=verbose,
            seed=seed,
        )

        results = bo.run()
        return results

    def suggest_to_minimize(self, data=None, verbose=True, seed=None):
        """
        Run ProBO AcqOptDesigner to suggest a design (i.e. a point to evaluate).

        Parameters
        ----------
        data : dict
            Dictionary with keys x (list) and y (1D numpy ndarray).
        verbose : bool
            If True, print information.
        seed : int
            If not None, set ProBO random seed to seed.
        """
        model = self._get_model()
        acqfunction = self._get_acqfunction()
        acqoptimizer = self._get_acqoptimizer()

        designer_params = None

        designer = probo.AcqOptDesigner(
            model=model,
            acqfunction=acqfunction,
            acqoptimizer=acqoptimizer,
            data=data,
            params=designer_params,
            verbose=verbose,
            seed=seed,
        )

        suggestion = designer.get()
        if verbose:
            print('Suggestion: {}'.format(suggestion))

        return suggestion

    def _get_model(self):
        """
        Return ProBO model based on self.model_config.
        """
        name = self.model_config['name']
        assert name in [
            'simplegp',
            'stangp',
            'stanproductgp',
            'gpytorchgp',
            'gpytorchproductgp',
            'sklearnpenn',
        ]

        if name == 'simplegp':
            model = probo.SimpleGp(self.model_config)
        elif name == 'stangp':
            model = probo.StanGp(self.model_config)
        elif name == 'stanproductgp':
            model = probo.StanProductGp(self.model_config)
        elif name == 'gpytorchgp':
            model = probo.GpytorchGp(self.model_config)
        elif name == 'gpytorchproductgp':
            model = probo.GpytorchProductGp(self.model_config)
        elif name == 'sklearnpenn':
            model = probo.SklearnPenn(self.model_config)

        return model

    def _get_acqfunction(self):
        """
        Return ProBO acqfunction based on self.acqfunction_config.
        """
        name = self.acqfunction_config['name']
        assert name in ['default']

        if name == 'default':
            acqfunction = probo.AcqFunction(self.acqfunction_config)

        return acqfunction

    def _get_acqoptimizer(self):
        """
        Return ProBO acqoptimizer based on self.acqoptimizer_config.
        """
        ao_name = self.acqoptimizer_config['name']
        pao_config_list = self.acqoptimizer_config.get('pao_config_list')

        dom_name = self.domain_config['name']
        dom_config_list = self.domain_config.get('dom_config_list')

        # For product acqoptimizer
        if (
            ao_name == 'product'
            or dom_name == 'product'
            or pao_config_list
            or dom_config_list
        ):

            # Ensure correct dom_config_list
            if type(dom_config_list) is not list:
                assert type(pao_config_list) is list
                dom_config_list = [self.domain_config] * len(pao_config_list)

            n_domain = len(dom_config_list)

            # Ensure correct pao_config_list
            if type(pao_config_list) is list:
                if len(pao_config_list) == n_domain:
                    pass
                elif len(pao_config_list) == 1:
                    pao_config_list = pao_config_list * n_domain
                else:
                    raise ValueError(
                        'self.acqoptimizer_config.pao_config_list list needs to have length either 1 or len(self.domain_config)'
                    )
            else:
                pao_config_list = [self.acqoptimizer_config] * n_domain

            acqoptimizer = self._get_product_acqoptimizer(
                dom_config_list, pao_config_list, self.acqoptimizer_config
            )

        # For single (non-product) acqoptimizer
        else:
            acqoptimizer = self._get_single_acqoptimizer(
                self.acqoptimizer_config, self.domain_config
            )

        return acqoptimizer

    def _get_product_acqoptimizer(self, dom_config_list, pao_config_list, pao_config):
        """Return ProductAcqOptimizer."""
        acqoptimizer_list = []
        for dom_config, ao_config in zip(dom_config_list, pao_config_list):
            acqoptimizer_single = self._get_single_acqoptimizer(ao_config, dom_config)
            acqoptimizer_list.append(acqoptimizer_single)

        acqoptimizer = probo.ProductAcqOptimizer(acqoptimizer_list, pao_config)
        return acqoptimizer

    def _get_single_acqoptimizer(self, ao_config, dom_config):
        """
        Return a single (non-product) ProBO acqoptimize.

        Parameters
        ----------
        ao_config : dict
            A dict containing config for a single (non-product) acqoptimizer.
        dom_config : dict
            A dict containing config for a single (non-product) domain.
        """
        name = ao_config['name']
        assert name in ['default', 'cobyla', 'neldermead']

        domain = self._get_domain(dom_config)

        if name == 'default':
            acqoptimizer = probo.AcqOptimizer(ao_config, domain)
        elif name == 'cobyla':
            acqoptimizer = probo.CobylaAcqOptimizer(ao_config, domain)
        elif name == 'neldermead':
            acqoptimizer = probo.NelderMeadAcqOptimizer(ao_config, domain)

        return acqoptimizer

    def _get_domain(self, dom_config):
        """Return Domain instance."""
        dom_name = dom_config['name']
        assert dom_name in ['real']

        if dom_name == 'real':
            domain = probo.RealDomain(dom_config)

        return domain
