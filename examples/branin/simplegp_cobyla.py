from tuun import Tuun, SimpleGp, CobylaAcqOptimizer, SimpleBo
from examples.branin.branin import branin, get_branin_domain

# define dataset
data = {'x': [], 'y': []}

# define model
model = SimpleGp({'ls': 3., 'alpha': 1.5, 'sigma': 1e-5})

# define acqfunction
acqfunction = {'acq_str': 'ei', 'n_gen': 500}

# define acqoptimizer
acqoptimizer = CobylaAcqOptimizer({'rand_every': 4}, get_branin_domain())

# define tuun
tu = Tuun(data, model, acqfunction, acqoptimizer, seed=11)

# define function
f = branin

# define and run BO
bo = SimpleBo(tu, f, params={'n_iter': 50})
results = bo.run()
