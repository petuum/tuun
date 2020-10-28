from tuun.main import Tuun
from hartmann import hartmann6

config = {
    'seed': 11,
    'acqfunction_config': {'name': 'default', 'acq_str': 'ei', 'n_gen': 500},
    'model_config': {'name': 'standistmatgp'},
}
tu = Tuun(config)

search_space_list = [('real', [[0, 1]] * 6)]
tu.set_config_from_list(search_space_list)

f = hartmann6

result = tu.minimize_function(f, 60)
