from tuun.main import Tuun

config = {
    'backend': 'probo',
    'seed': 11,
    'model_config': {'name': 'simplegp', 'ls': 3.0, 'alpha': 1.5, 'sigma': 1e-5},
    'acqfunction_config': {'name': 'default', 'acq_str': 'ei'},
    'acqoptimizer_config': {'name': 'default', 'max_iter': 200},
    'domain_config': {'name': 'real', 'min_max': [(-5, 5)]},
}
tu = Tuun(config)

data = {
    'x': [0.0, 1.0, 2.0],
    'y': [6.0, 3.0, 4.0],
}
suggestion = tu.suggest_to_minimize(data)
