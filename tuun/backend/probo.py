"""
Code for using ProBO as the backend tuning system.
"""
from argparse import Namespace
import copy

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

        # Set self.probo_config
        if not probo_config:
            probo_config = {}
        self.probo_config = probo_config

        # Transform domain
        self._transform_domain_config()

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
        # Convert data format
        data = self._convert_tuun_data_to_probo(data)

        # Transform data
        data = self._transform_data(data)

        # Set model, acqfunction, acqoptimizer
        model = self._get_model()
        acqfunction = self._get_acqfunction()
        acqoptimizer = self._get_acqoptimizer()

        bo_params = {'n_iter': n_iter}

        # TODO: if self.probo_config['normalize_real'], the following will optimize
        # original (untransformed) function f over transformed domain and with
        # transformed initial data. This is incorrect.
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
        # Convert data format
        data = self._convert_tuun_data_to_probo(data)

        # Transform data
        data = self._transform_data(data)

        # Set model, acqfunction, acqoptimizer
        model = self._get_model(verbose)
        acqfunction = self._get_acqfunction(verbose)
        acqoptimizer = self._get_acqoptimizer(verbose)

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

        # Inverse transform suggestion
        suggestion = self._transform_x(suggestion, inverse=True)

        suggestion = self._convert_probo_suggestion_to_tuun(suggestion)
        return suggestion

    def _get_model(self, verbose=True):
        """
        Return ProBO model based on self.model_config.
        """
        name = self.model_config['name']
        assert name in [
            'simplegp',
            'simpleproductkernelgp',
            'stangp',
            'stanproductgp',
            'standistmatgp',
            'gpystangp',
            'gpytorchgp',
            'gpytorchproductgp',
            'sklearnpenn',
        ]

        if name == 'simplegp':
            model = probo.SimpleGp(self.model_config, verbose=verbose)
        elif name == 'simpleproductkernelgp':
            model = probo.SimpleProductKernelGp(self.model_config, verbose=verbose)
        elif name == 'stangp':
            model = probo.StanGp(self.model_config, verbose=verbose)
        elif name == 'stanproductgp':
            model = probo.StanProductGp(self.model_config, verbose=verbose)
        elif name == 'standistmatgp':
            model = probo.StanDistmatGp(self.model_config, verbose=verbose)
        elif name == 'gpystangp':
            model = probo.GpystanGp(self.model_config, verbose=verbose)
        elif name == 'gpytorchgp':
            model = probo.GpytorchGp(self.model_config, verbose=verbose)
        elif name == 'gpytorchproductgp':
            model = probo.GpytorchProductGp(self.model_config, verbose=verbose)
        elif name == 'sklearnpenn':
            model = probo.SklearnPenn(self.model_config, verbose=verbose)

        return model

    def _get_acqfunction(self, verbose=True):
        """
        Return ProBO acqfunction based on self.acqfunction_config.
        """
        name = self.acqfunction_config['name']
        assert name in ['default']

        if name == 'default':
            acqfunction = probo.AcqFunction(self.acqfunction_config, verbose=verbose)

        return acqfunction

    def _get_acqoptimizer(self, verbose=True):
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
                dom_config_list, pao_config_list, self.acqoptimizer_config, verbose
            )

        # For single (non-product) acqoptimizer
        else:
            acqoptimizer = self._get_single_acqoptimizer(
                self.acqoptimizer_config, self.domain_config, verbose
            )

        return acqoptimizer

    def _get_product_acqoptimizer(
        self, dom_config_list, pao_config_list, pao_config, verbose=True
    ):
        """Return ProductAcqOptimizer."""
        acqoptimizer_list = []
        for dom_config, ao_config in zip(dom_config_list, pao_config_list):
            acqoptimizer_single = self._get_single_acqoptimizer(
                ao_config, dom_config, verbose
            )
            acqoptimizer_list.append(acqoptimizer_single)

        acqoptimizer = probo.ProductAcqOptimizer(
            acqoptimizer_list, pao_config, verbose=verbose
        )
        return acqoptimizer

    def _get_single_acqoptimizer(self, ao_config, dom_config, verbose=True):
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

        domain = self._get_domain(dom_config, verbose)

        if name == 'default':
            acqoptimizer = probo.AcqOptimizer(ao_config, domain, verbose=verbose)
        elif name == 'cobyla':
            acqoptimizer = probo.CobylaAcqOptimizer(ao_config, domain, verbose=verbose)
        elif name == 'neldermead':
            acqoptimizer = probo.NelderMeadAcqOptimizer(
                ao_config, domain, verbose=verbose
            )

        return acqoptimizer

    def _get_domain(self, dom_config, verbose=True):
        """Return Domain instance."""
        dom_name = dom_config['name']
        assert dom_name in ['real', 'list']

        if dom_name == 'real':
            domain = probo.RealDomain(dom_config, verbose=verbose)
        elif dom_name == 'list':
            domain = probo.ListDomain(dom_config, verbose=verbose)

        return domain

    def _convert_probo_suggestion_to_tuun(self, suggestion):
        """
        Given a suggestion from probo, convert to suggestion in Tuun format: a list with
        one element for each 1D type (including one element per real dimension).
        """
        real_idx = self.probo_config['real_idx']
        if not real_idx:
            # Only list types
            if isinstance(suggestion, list):
                new_suggestion = suggestion
            else:
                new_suggestion = [suggestion]
        elif self.probo_config['all_real']:
            # Only real types
            new_suggestion = suggestion
        else:
            # Blocks of multiple types, with all real in first block
            new_suggestion = suggestion[1:]
            for idx, real in zip(real_idx, suggestion[0]):
                new_suggestion.insert(idx, real)

        return new_suggestion

    def _convert_tuun_data_to_probo(self, data):
        """
        Given data Namespace in Tuun format (i.e. data.x is a list with each element
        corresponding to a 1D type, including one element per real dimension), convert
        to ProBO format (blocks of potentially multiple types, with all real in first
        block).
        """
        real_idx = self.probo_config['real_idx']
        data = copy.deepcopy(data)
        if not real_idx or self.probo_config['all_real']:
            new_data = data
        else:
            new_data_x = []
            for xi in data.x:
                real_list = [xi[idx] for idx in real_idx]
                new_xi = [real_list]

                _ = [xi.remove(real) for real in real_list]
                new_xi.extend(xi)
                new_data_x.append(new_xi)
            new_data = Namespace(x=new_data_x, y=data.y)

        return new_data

    def _transform_domain_config(self):
        """Transform domain."""

        # Normalize
        if self.domain_config['name'] == 'product':
            for domain_config in self.domain_config['dom_config_list']:
                domain_config = self._normalize_domain_config_block(domain_config)
        else:
            domain_config = self.domain_config
            domain_config = self._normalize_domain_config_block(domain_config)

    def _normalize_domain_config_block(self, domain_config):
        """Return domain config, possibly normalized to [0, 10]."""
        normalize_real = self.probo_config.get('normalize_real', False)
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
        if self.domain_config['name'] == 'product':
            dom_config_list = self.domain_config['dom_config_list']
            for x_block, dom_config in zip(x, dom_config_list):
                x_block = self._normalize_x_block(x_block, dom_config, inverse=inverse)
        else:
            x = self._normalize_x_block(x, self.domain_config, inverse=inverse)
        return x

    def _normalize_x_block(self, x, domain_config, inverse=False):
        """Return x, possibly normalized to [0, 10]."""
        normalize_real = self.probo_config.get('normalize_real', False)
        if domain_config['name'] == 'real' and normalize_real:
            for i, bounds in enumerate(domain_config['min_max_init']):
                if inverse is True:
                    scale_factor = (bounds[1] - bounds[0]) / 10.0
                    x[i] = x[i] * scale_factor + bounds[0]
                    #x = x * scale_factor + bounds[0]
                else:
                    scale_factor = 10.0 / (bounds[1] - bounds[0])
                    x[i] = (x[i] - bounds[0]) * scale_factor
                    #x = (x - bounds[0]) * scale_factor
        return x
