from tuun.backend import DragonflyBackend

domain_config = {
    'name': 'product',
    'dom_config_list': [
        {'name': 'real', 'min_max': [(-0.1, 2.1)]},
        {'name': 'real', 'min_max': [(-0.2, 2.2)]},
    ],
}
opt_config = {'name': 'product'}
dragonfly_config = {'acq_str': 'ucb-ei', 'n_init_rs': 0}

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

db = DragonflyBackend(domain_config, opt_config, dragonfly_config)
suggestion = db.suggest_to_minimize(data)
