from tuun.backend import ProboBackend

model_config = {'name': 'simplegp', 'ls': 3.0, 'alpha': 1.5, 'sigma': 1e-5}
acqfunction_config = {'name': 'default', 'acq_str': 'ei'}
acqoptimizer_config = {'name': 'default', 'max_iter': 200}
domain_config = {'min_max': [(-5, 5)]}

data = {
    'x': [0.0, 1.0, 2.0],
    'y': [6.0, 3.0, 4.0],
}

pb = ProboBackend(model_config, acqfunction_config, acqoptimizer_config, domain_config)
suggestion = pb.suggest_to_minimize(data)
