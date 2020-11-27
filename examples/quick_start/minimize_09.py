from tuun.main import Tuun

# instantiate Tuun
config = {
    'seed': 12,
    'acqfunction_config': {'name': 'default', 'acq_str': 'ucb'},
    'probo_config': {'normalize_real': True},
    'model_config': {'name': 'standistmatgp'},
}
tu = Tuun(config)

# set search space
search_space = [('real', [-25, 25])] * 2
tu.set_config_from_list(search_space)

# define function to optimize
def f(x):
    f_s = lambda x: x ** 4 - x ** 2 + 0.1 * x
    return f_s(x[0]) + f_s(x[1])

# minimize function over search space
result = tu.minimize_function(f, 50)
