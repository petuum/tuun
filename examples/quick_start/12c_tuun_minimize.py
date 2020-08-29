from tuun.main import Tuun

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
    'acqfunction_config': {'name': 'default', 'acq_str': 'ucb', 'n_gen': 500},
    'acqoptimizer_config': {'name': 'cobyla', 'rand_every': 4, 'jitter': True},
    'domain_config': {
        'name': 'product',
        'dom_config_list': [
            {'name': 'real', 'min_max': [(-10, 10)]},
            {'name': 'real', 'min_max': [(-10, 10)]},
        ],
    },
}
tu = Tuun(config)

f_s = lambda x: x ** 4 - x ** 2 + 0.1 * x
f = lambda x_list: f_s(x_list[0]) + f_s(x_list[1])
result = tu.minimize_function(f, 50)
