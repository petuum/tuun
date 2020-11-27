from tuun.main import Tuun

config = {
    'backend': 'probo',
    'seed': 11,
    'model_config': {'name': 'simplegp', 'ls': 3.0, 'alpha': 1.5, 'sigma': 1e-5},
    'acqfunction_config': {'name': 'default', 'acq_str': 'ei'},
    'acqoptimizer_config': {'name': 'default', 'max_iter': 200},
    'domain_config': {'name': 'real', 'min_max': [(-0.5, 2.5), (-0.6, 2.6)]},
}
tu = Tuun(config)

data = {
    'x': [
        [0.0, 0.0],
        [0.0, 2.0],
        [2.0, 0.0],
        [2.0, 2.0],
        [1.0, 1.0],
        [1.1, 1.1],
    ],
    'y': [15.0, 6.0, 9.0, 12.0, 1.0, 1.5],
}
suggestion = tu.suggest_to_minimize(data)

# maximization

data['y'] = [-1*v for v in data['y']]
suggestion_max = tu.suggest_to_maximize(data)

print(type(suggestion))
# <class 'list'>
assert suggestion == suggestion_max
