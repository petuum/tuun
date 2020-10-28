from tuun.main import Tuun
from branin import branin

config = {
    'seed': 11,
    'acqfunction_config': {'name': 'default', 'acq_str': 'ei', 'n_gen': 500},
    'model_config': {'name': 'standistmatgp'},
}
tu = Tuun(config)

search_space_list = [('real', [[-5, 10], [0, 15]])]
#search_space_list = [('real', [[-5, 10]]), ('real', [[0, 15]])]
tu.set_config_from_list(search_space_list)

f = branin

result = tu.minimize_function(f, 30)
