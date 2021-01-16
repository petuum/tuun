from tuun.backend import ProboBackend

model_config = {'name': 'simplegp', 'ls': 3.0, 'alpha': 1.5, 'sigma': 1e-5}
acqfunction_config = {'name': 'default', 'acq_str': 'ei'}
acqoptimizer_config = {
    'name': 'product',
    'pao_config_list': [{'name': 'default'}, {'name': 'default'}],
}
domain_config = {'name': 'real', 'min_max': [(-5, 5)]}

f_s = lambda x: x[0] ** 4 - x[0] ** 2 + 0.1 * x[0]
f = lambda x_list: f_s(x_list[0]) + f_s(x_list[1])

pb = ProboBackend(model_config, acqfunction_config, acqoptimizer_config, domain_config)
results = pb.minimize_function(f, 50)
