from tuun.backend import DragonflyBackend

domain_config = {'name': 'real', 'bounds_list': [[-5, 5]]}
opt_config = {'name': 'real'}
dragonfly_config = {'acq_str': 'ucb-ei', 'n_init_rs': 0}

data = {
    'x': [0.0, 1.0, 2.0],
    'y': [6.0, 3.0, 4.0],
}

db = DragonflyBackend(domain_config, opt_config, dragonfly_config)
suggestion = db.suggest_to_minimize(data)
