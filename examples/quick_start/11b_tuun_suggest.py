from tuun.main import Tuun

config = {
    'backend': 'dragonfly',
    'seed': 11,
    'domain_config': {'name': 'real', 'bounds_list': [(-0.5, 2.5), (-0.6, 2.6)]},
    'opt_config': {'name': 'real'},
    'dragonfly_config': {'acq_str': 'ucb-ei', 'n_init_rs': 0}
}
tu = Tuun(config)

data = {
    'x': [
        [0.0, 0.0],
        [0.0, 2.0],
        [2.0, 0.0],
        [2.0, 2.0],
        [1.0, 1.0],
        [1.1, 1.1],
    ],
    'y': [15.0, 6.0, 9.0, 12.0, 1.0, 1.5],
}
suggestion = tu.suggest_to_minimize(data)
