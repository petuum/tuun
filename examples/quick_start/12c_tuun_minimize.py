from tuun.main import Tuun

config = {
    'backend': 'dragonfly',
    'seed': 11,
    'domain_config': {'name': 'real', 'bounds_list': [[-5, 5]]},
    'opt_config': {'name': 'real'},
    'dragonfly_config': {'acq_str': 'ucb-ei', 'n_init_rs': 0}
}
tu = Tuun(config)

data = {'x': [[1.]],  'y': [.1]}

f = lambda x: x[0] ** 4 - x[0] ** 2 + 0.1 * x[0]
result = tu.minimize_function(f, 20, data)
