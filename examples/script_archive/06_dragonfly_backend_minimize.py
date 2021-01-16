from tuun.backend import DragonflyBackend

domain_config = {'name': 'real', 'min_max': [[-5, 5]]}
opt_config = {'name': 'real'}

f = lambda x: x[0] ** 4 - x[0] ** 2 + 0.1 * x[0]

db = DragonflyBackend(domain_config, opt_config)
results = db.minimize_function(f, 20)
