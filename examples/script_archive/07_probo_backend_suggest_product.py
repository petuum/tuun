from tuun.backend import ProboBackend
import numpy as np

model_config = {'name': 'gpytorchproductgp'}
acqfunction_config = {'name': 'default', 'acq_str': 'mean', 'n_gen': 5000}
acqoptimizer_config = {
    'name': 'product',
    'n_iter_bcd': 3,
    'pao_config_list': [
        {'name': 'default', 'max_iter': 1000},
        {'name': 'cobyla', 'rand_every': 4, 'jitter': True},
    ],
}
domain_config = {
    'name': 'product',
    'dom_config_list': [
        {'name': 'real', 'min_max': [(-0.1, 2.1)]},
        {'name': 'real', 'min_max': [(-0.2, 2.2)]},
    ],
}

data = {
    'x': [
        [[0.0], [0.0]],
        [[0.0], [2.0]],
        [[2.0], [0.0]],
        [[2.0], [2.0]],
        [[1.0], [1.0]],
        [[1.1], [1.1]],
    ],
    'y': [15.0, 6.0, 9.0, 12.0, 1.0, 1.5],
}

pb = ProboBackend(model_config, acqfunction_config, acqoptimizer_config, domain_config)
suggestion = pb.suggest_to_minimize(data)
