from tuun.main import Tuun

# instantiate Tuun
config = {
    'seed': 12,
    'acqfunction_config': {'name': 'default', 'acq_str': 'ucb'},
}
tu = Tuun(config)

# set search space
search_space = ('real', [-10, 10])
tu.set_config_from_list(search_space)

# define function to optimize
f = lambda x: x[0] ** 4 - x[0] ** 2 + 0.1 * x[0]

# minimize function over search space
result = tu.minimize_function(f, 30)
