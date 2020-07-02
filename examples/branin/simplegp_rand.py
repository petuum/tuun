from tuun import Tuun, SimpleGp, AcqOptimizer, SimpleBo
from examples.branin.branin import branin, get_branin_domain

# define model
model = SimpleGp({'ls': 3.0, 'alpha': 1.5, 'sigma': 1e-5})

# define acqfunction
acqfunction = {'acq_str': 'ei', 'n_gen': 500}

# define acqoptimizer
acqoptimizer = AcqOptimizer(domain=get_branin_domain())

# define tuun
tu = Tuun(model, acqfunction, acqoptimizer, data=None, seed=11)

# define function
f = branin

# define and run BO
bo = SimpleBo(tu, f, params={'n_iter': 50})
results = bo.run()
