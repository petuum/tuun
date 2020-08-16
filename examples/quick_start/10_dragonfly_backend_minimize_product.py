from tuun.backend import DragonflyBackend

domain_config = {
    'name': 'product',
    'dom_config_list': [
        {'name': 'real', 'min_max': [(-10, 10)]},
        {'name': 'real', 'min_max': [(-4, 4)]},
    ],
}
opt_config = {'name': 'product'}

f_s = lambda x: x ** 4 - x ** 2 + 0.1 * x
f = lambda x_list: f_s(x_list[0]) + f_s(x_list[1])

db = DragonflyBackend(domain_config, opt_config)
results = db.minimize_function(f, 20)
