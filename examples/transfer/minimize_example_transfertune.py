from tuun.main import Tuun
from parse_online_task import parse_online_task_json

# Load domain_list
domain_list, _, look_up_table = parse_online_task_json()

config = {
    # set seed
    'seed': 13,

    # configure tuning backend
    'backend': 'probo',

    # configure model
    'model_config': {
        'name': 'stantransfergp',
        'ls': 3.0,
        'alpha': 1.5,
        'sigma': 1e-5,
        'ndimx': 16,
        'transfer_config': {
            'local_path': "PATH/TO/GBR/WEIGHTS",
            'task_name': "imageClassification",
            'model_type': 'sklearn.linear_model.Ridge',
            'output_mode': 'val_error',
            'ndim': 16},
    },

    # configure acquisition function
    'acqfunction_config': {'name': 'default', 'acq_str': 'ei'},

    # configure acquisition optimizer
    'acqoptimizer_config': {'name': 'default', 'max_iter': 200},

    # configure domain
    'domain_config': {'x': ('list', domain_list)}
}
tu = Tuun(config)


def f(x):
    return look_up_table[tuple(x)]


result = tu.minimize_function(f, 20)
