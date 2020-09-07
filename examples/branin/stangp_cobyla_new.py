from tuun.main import Tuun
from examples.branin.branin import branin

config = {
    'seed': 11,
    'model_config': {
        'name': 'stangp',
        'ndimx': 2,
        'model_str': 'optfixedsig',
        'ig1': 4.0,
        'ig2': 3.0,
        'n1': 1.0,
        'n2': 1.0,
        'sigma': 1e-5,
        'niter': 70,
    },
    'acqfunction_config': {'name': 'default', 'acq_str': 'ei', 'n_gen': 500},
    'acqoptimizer_config': {'name': 'cobyla', 'rand_every': 4, 'jitter': True},
    'domain_config': {'name': 'real', 'min_max': [[-5.0, 10.0], [0.0, 15.0]]},
}
tu = Tuun(config)

f = branin
result = tu.minimize_function(f, 30)
