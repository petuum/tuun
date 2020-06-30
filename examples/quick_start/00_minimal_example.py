from tuun import Tuun, AcqOptimizer, SimpleGp

# define initial dataset
data = {'x': [0., 1., 2.], 'y': [6., 3., 4.]}

# define model
model = SimpleGp({'ls': 3.7, 'alpha': 1.85, 'sigma': 1e-5})

# define acqfunction
acqfunction = {'acq_str': 'ei'}

# define acqoptimizer
acqoptimizer = AcqOptimizer(domain={'min_max': [(-5, 5)]})

# define tuun
tu = Tuun(data, model, acqfunction, acqoptimizer)

# get acquisition optima
acq_optima = tu.get()
print('acq_optima: {}'.format(acq_optima))
