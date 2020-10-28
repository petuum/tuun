import numpy as np
from tuun.main import Tuun
from examples.branin.branin import branin

config = {
    'seed': 11,
    'acqfunction_config': {'name': 'default', 'acq_str': 'ei', 'n_gen': 500},
    'acqoptimizer_config': {'name': 'neldermead', 'n_init_rs': 10},
    'model_config': {'name': 'standistmatgp'},
}
tu = Tuun(config)

search_space_list = [('real', [[-5, 10], [0, 15]]  * 20)]
tu.set_config_from_list(search_space_list)

f = lambda x: np.sum([branin(x[2 * i : 2 * i + 2]) for i in range(20)])

result = tu.minimize_function(f, 120)
