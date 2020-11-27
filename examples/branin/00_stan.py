from tuun.main import Tuun
from branin import branin

# instantiate Tuun
config = {
    'seed': 11,
    'acqfunction_config': {'name': 'default', 'acq_str': 'ei', 'n_gen': 500},
    'model_config': {'name': 'standistmatgp'},
}
tu = Tuun(config)

# set search space
search_space = [('real', [-5, 10]), ('real', [0, 15])]
tu.set_config_from_list(search_space)

# define function to optimize
f = branin

# minimize function over search space
result = tu.minimize_function(f, 30)
