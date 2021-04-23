from tuun.main import Tuun

# instantiate Tuun
config = {
    'seed': 12,
    'acqfunction_config': {'name': 'default', 'acq_str': 'ucb'},
}
tu = Tuun(config)

# set search space
search_space = [
    ('real', [-10, 10]), ('list', ['a', 'b', 'c', 'd', 'e']), ('real', [-10, 10])
]
tu.set_config_from_list(search_space)

# define function to optimize
def f(x):
    str_map = {'a': 4, 'b': 5, 'c': 0, 'd': 0.7, 'e': -0.7}
    f_s = lambda x: x ** 4 - x ** 2 + 0.1 * x
    return f_s(x[0]) + f_s(str_map[x[1]])

# minimize function over search space
result = tu.minimize_function(f, 50)
