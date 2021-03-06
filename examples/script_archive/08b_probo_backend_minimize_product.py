from tuun.backend import ProboBackend

model_config = {
    'name': 'stangp',
    'ndimx': 2,
    'model_str': 'optfixedsig',
    'ig1': 4.0,
    'ig2': 3.0,
    'n1': 1.0,
    'n2': 1.0,
    'sigma': 1e-5,
    'niter': 70,
}
acqfunction_config = {'name': 'default', 'acq_str': 'ucb', 'n_gen': 500}
acqoptimizer_config = {
    'name': 'product',
    'pao_config_list': [
        {'name': 'cobyla', 'rand_every': 4, 'jitter': True},
        {'name': 'cobyla', 'rand_every': 4, 'jitter': True},
    ],
}
domain_config = {'name': 'real', 'min_max': [(-10, 10)]}

f_s = lambda x: x[0] ** 4 - x[0] ** 2 + 0.1 * x[0]
f = lambda x_list: f_s(x_list[0]) + f_s(x_list[1])

pb = ProboBackend(model_config, acqfunction_config, acqoptimizer_config, domain_config)
results = pb.minimize_function(f, 50)
