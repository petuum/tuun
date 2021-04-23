from tuun.main import Tuun

# instantiate Tuun
tu = Tuun()

# set search space
search_space = ('real', [-5, 5])
tu.set_config_from_list(search_space)

# define function to optimize
f = lambda x: x[0] ** 4 - x[0] ** 2 + 0.1 * x[0]

# minimize function over search space
result = tu.minimize_function(f, 20)
