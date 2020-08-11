from tuun import AcqOptimizer, SimpleBo
from examples.branin.branin import branin, get_branin_domain

# define function
f = branin

# define acqfunction
acqfunction = {'acq_str': 'rand'}

# define acqoptimizer
acqoptimizer = AcqOptimizer({'max_iter': 1}, get_branin_domain())

# define and run BO
bo = SimpleBo(f, None, acqfunction, acqoptimizer, params={'n_iter': 50}, seed=11)
results = bo.run()
