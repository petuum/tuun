from tuun.backend import DragonflyBackend

domain_config = {'name': 'euc', 'bounds_list': [[-5, 5]]}
opt_config = {'name': 'euc'}

f = lambda x: x ** 4 - x ** 2 + 0.1 * x

db = DragonflyBackend(domain_config, opt_config)
results = db.minimize_function(f, 20)
