from tuun.backend import DragonflyBackend

domain_config = {
    'name': 'product',
    'dom_config_list': [
        {'name': 'real', 'min_max': [(-13, -12)]},
        {'name': 'real', 'min_max': [(14, 17)]},
    ],
}
opt_config = {'name': 'product'}

data = {
    'x': [[0.0, 0.0], [0.0, 1.0], [1.0, 0.0]],
    'y': [6.0, 3.0, 4.0],
}

db = DragonflyBackend(domain_config, opt_config)
suggestion = db.suggest_to_minimize(data)
