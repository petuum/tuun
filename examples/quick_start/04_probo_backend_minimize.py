from tuun.backend import ProboBackend

model_config = {'name': 'simplegp', 'ls': 3.0, 'alpha': 1.5, 'sigma': 1e-5}
acqfunction_config = {'name': 'default', 'acq_str': 'ei'}
acqoptimizer_config = {'name': 'default', 'max_iter': 200}
domain_config = {'name': 'real', 'min_max': [(-5, 5)]}

f = lambda x: x ** 4 - x ** 2 + 0.1 * x

pb = ProboBackend(model_config, acqfunction_config, acqoptimizer_config, domain_config)
results = pb.minimize_function(f, 20)
