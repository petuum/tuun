"""
Main interface for Tuun.
"""
from argparse import Namespace
from tuun.util.database import MetricDatabase
import numpy as np
import os
from .backend import ProboBackend, DragonflyBackend
from .util.log import init_default_logger
from .util.database import init_db
from .util.misc_util import shortuid, dict_to_listspace


ENV_TUUN_LOG = os.getenv('TUUN_LOG', 'False')


class Tuun:
    """Main interface to Tuun."""

    def __init__(self, config_dict={}):
        """
        Parameters
        ----------
        config_dict : dict
            Config to specify Tuun options.
        """
        # Initialize stdout logger
        self.exp_id = shortuid()
        init_default_logger(self.exp_id, config_dict)

        # Initialize database
        if 'domain_config' in config_dict:
            self.db: MetricDatabase = init_db(
                'tuun.sqlite',
                self.exp_id,
                config_dict['domain_config'])
        else:
            self.db = None

        self._parameter_keys = None
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
        probo_config = getattr(config, 'probo_config', None)
        self.config.probo_config = probo_config

        domain_config = getattr(config, 'domain_config', None)
        if domain_config is None:
            domain_config = ('real', [0.0, 10.0])
        if type(domain_config) == dict:
            self._parameter_keys = list(domain_config.keys())
            domain_config = dict_to_listspace(domain_config)

        search_space = self._convert_search_space_to_probo(domain_config)
        self._set_domain_config_from_list(search_space)

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

    def set_config_from_list(self, search_space):
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
        if type(search_space) == dict:
            if ENV_TUUN_LOG.lower() in ('true', 't', '1') and not self.db:
                self.db = init_db('tuun.sqlite', self.exp_id, search_space)

            self._parameter_keys = list(search_space.keys())
            search_space = dict_to_listspace(search_space)

        search_space_list = self._format_search_space_list(search_space)

        # Convert search_space_list to ProBO format.
        # TODO: incorporate this into probo backend
        search_space_list = self._convert_search_space_to_probo(search_space_list)

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

    def _format_search_space_list(self, search_space_list):
        """Return search_space_list in correct format."""
        assert type(search_space_list) in [list, tuple]

        if type(search_space_list) is list:
            assert all([type(ss) is tuple for ss in search_space_list])
        elif type(search_space_list) is tuple:
            search_space_list = [search_space_list]

        return search_space_list

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
        jitter_val = self.config.acqoptimizer_config.get('jitter_val', 0.2)
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
                    'jitter_val': jitter_val,
                    'n_init_rs': n_init_rs,
                }
            else:
                ao_config = {
                    'name': 'cobyla',
                    'init_str': 'bsf',
                    'rand_every': 4,
                    'jitter': jitter,
                    'jitter_val': 0.2,
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

    def _convert_search_space_to_probo(self, search_space_list):
        """
        Given a search_space_list as a list of tuples, each representing a single
        dimension of the search space, combine 'real' types, form blocks, and record
        transformation.
        """
        search_space_list = self._format_search_space_list(search_space_list)

        # Instantiate self.config.probo_config if it does not exist
        if not self.config.probo_config:
            self.config.probo_config = {}

        real_idx = [i for i, ss in enumerate(search_space_list) if ss[0] == 'real']
        self.config.probo_config['real_idx'] = real_idx

        if len(real_idx) == len(search_space_list):
            self.config.probo_config['all_real'] = True
        else:
            self.config.probo_config['all_real'] = False

        if len(real_idx) > 0:
            ss_list = []
            real_tup = ('real', [search_space_list[i][1] for i in real_idx])
            ss_list.append(real_tup)
            for i, ss in enumerate(search_space_list):
                if i not in real_idx:
                    ss_list.append(ss)
        else:
            ss_list = search_space_list

        return ss_list

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
        data = self._format_initial_data(data)
        if data.x and ENV_TUUN_LOG.lower() in ('true', 't', '1'):
            self.db.add_metric(data)

        # Set data-dependent subseed
        subseed = int(np.random.uniform(12345 * (len(data.x) + 1)))

        # Call backend suggest_to_minimize method
        suggestion = self.backend.suggest_to_minimize(
            data=data, verbose=verbose, seed=subseed
        )

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
        data = self._format_initial_data(data)
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
        data = self._format_initial_data(data)

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

    def _format_initial_data(self, data):
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

    def _run_backend_minimize_function(self, f, n_iter, data, data_update_fun, verbose, seed):
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
