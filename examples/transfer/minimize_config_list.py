from tuun.main import Tuun


# Define a domain_list
domain_list = [
    [-5, -4],
    [-3, -2],
    [-1.2, -0.2],
    [-1.1, -0.1],
    [-1, 0],
    [-0.9, 0.1],
    [-0.8, 0.2],
    [3, 4],
    [5, 6]
]

# Define Tuun configuration
config = {
    # set seed
    'seed': 12,

    # configure tuning backend
    'backend': 'probo',

    # configure model
    'model_config': {'name': 'stangp', 'ndimx': 2},

    # configure acquisition function
    'acqfunction_config': {'name': 'default', 'acq_str': 'ei'},

    # configure acquisition optimizer
    'acqoptimizer_config': {'name': 'default', 'max_iter': 200},

    # configure domain
    'domain_config': {'input_list': ('list', domain_list)}
}
tu = Tuun(config)


# Define function
def f(x):
    return x[0] ** 4 - x[0] ** 2 + 0.1 * x[0]


result = tu.minimize_function(f, 30, verbose=True)
