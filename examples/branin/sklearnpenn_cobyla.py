from tuun import Tuun, SklearnPenn, CobylaAcqOptimizer, SimpleBo
from examples.branin.branin import branin, get_branin_domain

# define model
model = SklearnPenn()

# define acqfunction
acqfunction = {'acq_str': 'ucb', 'n_gen': 500}

# define acqoptimizer
acqoptimizer = CobylaAcqOptimizer(
    {'rand_every': 4, 'jitter': True}, get_branin_domain()
)

# define tuun
tu = Tuun(model, acqfunction, acqoptimizer, data=None, seed=11)

# define function
f = branin

# define and run BO
bo = SimpleBo(tu, f, params={'n_iter': 50})
results = bo.run()
