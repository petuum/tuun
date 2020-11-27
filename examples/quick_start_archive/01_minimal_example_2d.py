from tuun import AcqOptDesigner, AcqOptimizer, SimpleGp
import numpy as np

# define model
model = SimpleGp({'ls': 3.0, 'alpha': 1.5, 'sigma': 1e-5})

# define acqfunction
acqfunction = {'acq_str': 'ei', 'n_gen': 100}

# define acqoptimizer
acqoptimizer = AcqOptimizer(domain={'min_max': [(-5, 5), (-5, 5)]})

# define initial dataset
data = {
    'x': [[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0], [0.5, 0.5]],
    'y': [6.0, 3.0, 4.0, 5.0, 2.0],
}

# define designer
designer = AcqOptDesigner(model, acqfunction, acqoptimizer, data)

# get acquisition optima
acq_optima = designer.get()
print('acq_optima: {}'.format(acq_optima))
