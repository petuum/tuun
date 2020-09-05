from tuun import AcqOptDesigner, AcqOptimizer, SimpleGp

# define model
model = SimpleGp({'ls': 3.0, 'alpha': 1.5, 'sigma': 1e-5})

# define acqfunction
acqfunction = {'acq_str': 'ei'}

# define acqoptimizer
acqoptimizer = AcqOptimizer(domain={'min_max': [(-5, 5)]})

# define initial dataset
data = {'x': [[0.0], [1.0], [2.0]], 'y': [6.0, 3.0, 4.0]}

# define designer
designer = AcqOptDesigner(model, acqfunction, acqoptimizer, data)

# get acquisition optima
acq_optima = designer.get()
print('acq_optima: {}'.format(acq_optima))
