from tuun import Tuun, SimpleGp, AcqOptimizer, SimpleBo
from examples.hartmann.hartmann import hartmann6, get_hartmann6_domain

# define model
model = SimpleGp({'ls': 3., 'alpha': 1.5, 'sigma': 1e-5})

# define acqfunction
acqfunction = {'acq_str': 'ei', 'n_gen': 500}

# define acqoptimizer
acqoptimizer = AcqOptimizer(domain=get_hartmann6_domain())

# define tuun
tu = Tuun(model, acqfunction, acqoptimizer, data=None, seed=11)

# define function
f = hartmann6

# define and run BO
bo = SimpleBo(tu, f, params={'n_iter': 50})
results = bo.run()
