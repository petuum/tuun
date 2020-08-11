from tuun import SimpleGp, AcqOptimizer, SimpleBo
from examples.branin.branin import branin, get_branin_domain

# define function
f = branin

# define model
model = SimpleGp({'ls': 3.0, 'alpha': 1.5, 'sigma': 1e-5})

# define acqfunction
acqfunction = {'acq_str': 'ei', 'n_gen': 500}

# define acqoptimizer
acqoptimizer = AcqOptimizer(domain=get_branin_domain())

# define and run BO
bo = SimpleBo(f, model, acqfunction, acqoptimizer, params={'n_iter': 50}, seed=11)
results = bo.run()
