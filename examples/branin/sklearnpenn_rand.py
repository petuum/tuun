from tuun import SklearnPenn, AcqOptimizer, SimpleBo
from examples.branin.branin import branin, get_branin_domain

# define function
f = branin

# define model
model = SklearnPenn()

# define acqfunction
acqfunction = {'acq_str': 'ucb', 'n_gen': 500}

# define acqoptimizer
acqoptimizer = AcqOptimizer(domain=get_branin_domain())

# define and run BO
bo = SimpleBo(f, model, acqfunction, acqoptimizer, params={'n_iter': 50}, seed=11)
results = bo.run()
