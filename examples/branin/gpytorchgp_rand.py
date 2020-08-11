from tuun import GpytorchGp, AcqOptimizer, SimpleBo
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
acqoptimizer = AcqOptimizer(domain=get_branin_domain())

# define and run BO
bo = SimpleBo(f, model, acqfunction, acqoptimizer, params={'n_iter': 50}, seed=seed)
results = bo.run()
