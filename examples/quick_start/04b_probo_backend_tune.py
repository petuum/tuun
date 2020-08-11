from tuun.backend import ProboBackend

model_config={
    'name': 'stangp',
    'ndimx': 1,
    'model_str': 'optfixedsig',
    'ig1': 4.0,
    'ig2': 3.0,
    'n1': 1.0,
    'n2': 1.0,
    'sigma': 1e-5,
    'niter': 70,
}
acqfunction_config = {'name': 'default', 'acq_str': 'ucb', 'n_gen': 500}
acqoptimizer_config = {'name': 'cobyla', 'rand_every': 4, 'jitter': True}
domain_config = {'min_max': [(-10, 10)]}

f = lambda x: x ** 4 - x ** 2 + 0.1 * x

pb = ProboBackend(model_config, acqfunction_config, acqoptimizer_config, domain_config)
results = pb.tune_function(f, 20, seed=11)
