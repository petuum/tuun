from tuun.backend import DragonflyBackend

domain_config = {'name': 'real', 'min_max': [(-8.0, 7.0), (-7.0, 8.0)]}
opt_config = {'name': 'real'}
dragonfly_config = {'acq_str': 'ucb-ei', 'n_init_rs': 0}

data = {
    'x': [[0.0, 0.0], [0.0, 1.0], [1.0, 0.0]],
    'y': [6.0, 3.0, 4.0],
}

db = DragonflyBackend(domain_config, opt_config, dragonfly_config)
suggestion = db.suggest_to_minimize(data)
