from tuun import GpytorchGp, CobylaAcqOptimizer, SimpleBo
from examples.branin.branin import branin, get_branin_domain

# define seed
seed = 11

# define function
f = branin

# define model
model = GpytorchGp({'seed': seed})

# define acqfunction
acqfunction = {'acq_str': 'ucb', 'n_gen': 500}

# define acqoptimizer
acqoptimizer = CobylaAcqOptimizer({'rand_every': 4}, get_branin_domain())

# define and run BO
bo = SimpleBo(f, model, acqfunction, acqoptimizer, params={'n_iter': 50}, seed=seed)
results = bo.run()
