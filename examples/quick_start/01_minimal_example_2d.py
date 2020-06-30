from tuun import Tuun, AcqOptimizer, SimpleGp
import numpy as np

# define initial dataset
data = {'x': [np.array([0., 0.]), np.array([0., 1.]), np.array([1., 0.]),
              np.array([1., 1.]), np.array([.5, .5])],
        'y': [6., 3., 4., 5., 2.]}

# define model
model = SimpleGp({'ls': 3.7, 'alpha': 1.85, 'sigma': 1e-5})

# define acqfunction
acqfunction = {'acq_str': 'ei', 'n_gen': 100}

# define acqoptimizer
acqoptimizer = AcqOptimizer(domain={'min_max': [(-5, 5), (-5, 5)]})

# define tuun
tu = Tuun(data, model, acqfunction, acqoptimizer)

# get acquisition optima
acq_optima = tu.get()
print('acq_optima: {}'.format(acq_optima))
