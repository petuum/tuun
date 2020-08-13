from tuun.backend import ProboBackend

model_config = {'name': 'gpytorchgp'}
acqfunction_config = {'name': 'default', 'acq_str': 'ucb', 'n_gen': 500}
acqoptimizer_config = {'name': 'neldermead', 'max_iter': 200, 'jitter': True}
domain_config = {'name': 'real', 'min_max': [(-8.0, 7.0), (-7.0, 8.0)]}

data = {
    'x': [[0.0, 0.0], [0.0, 1.0], [1.0, 0.0]],
    'y': [6.0, 3.0, 4.0],
}

pb = ProboBackend(model_config, acqfunction_config, acqoptimizer_config, domain_config)
suggestion = pb.suggest_to_minimize(data)
