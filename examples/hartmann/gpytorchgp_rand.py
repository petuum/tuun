from tuun import Tuun, GpytorchGp, AcqOptimizer, SimpleBo
from examples.hartmann.hartmann import hartmann6, get_hartmann6_domain

# define seed
seed = 11

# define model
model = GpytorchGp({'seed': seed})

# define acqfunction
acqfunction = {'acq_str': 'ucb', 'n_gen': 500}

# define acqoptimizer
acqoptimizer = AcqOptimizer(domain=get_hartmann6_domain())

# define tuun
tu = Tuun(model, acqfunction, acqoptimizer, data=None, seed=seed)

# define function
f = hartmann6

# define and run BO
bo = SimpleBo(tu, f, params={'n_iter': 50})
results = bo.run()
