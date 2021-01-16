from tuun.main import Tuun

config = {
    'seed': 11,
    'model_config': {
        'name': 'simpleproductkernelgp',
        'ls': 3.0,
        'alpha': 1.5,
        'sigma': 1e-5,
        'domain_spec': ['list', 'real'],
    },
    'acqfunction_config': {'name': 'default', 'acq_str': 'ucb', 'n_gen': 500},
    'acqoptimizer_config': {
        'name': 'product',
        'n_iter_bcd': 3,
        #'rand_every': 4,
        'n_init_rs': 3,
        'pao_config_list': [
            {'name': 'default'},
            {'name': 'cobyla', 'init_str': 'init_opt'},
        ],
    },
    'domain_config': {
        'name': 'product',
        'dom_config_list': [
            {'name': 'list', 'domain_list': ['a', 'b', 'c', 'd', 'e']},
            {'name': 'real', 'min_max': [[-10, 10]]},
        ],
    },
}
tu = Tuun(config)

# Define function on list-real domain points
str_map = {'a': [4], 'b': [5], 'c': [0], 'd': [0.7], 'e': [-0.7]}
f_s = lambda x: x[0] ** 4 - x[0] ** 2 + 0.1 * x[0]
f = lambda x_list: f_s(str_map[x_list[0]]) + f_s(x_list[1])

result = tu.minimize_function(f, 50)
