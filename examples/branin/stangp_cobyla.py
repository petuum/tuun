from tuun import StanGp, CobylaAcqOptimizer, SimpleBo
from examples.branin.branin import branin, get_branin_domain

# define function
f = branin

# define model
model = StanGp(
    {
        'ndimx': 2,
        'model_str': 'optfixedsig',
        'ig1': 4.0,
        'ig2': 3.0,
        'n1': 1.0,
        'n2': 1.0,
        'sigma': 1e-5,
        'niter': 70,
    }
)

# define acqfunction
acqfunction = {'acq_str': 'ei', 'n_gen': 500}

# define acqoptimizer
acqoptimizer = CobylaAcqOptimizer({'rand_every': 4}, get_branin_domain())

# define and run BO
bo = SimpleBo(f, model, acqfunction, acqoptimizer, params={'n_iter': 50}, seed=11)
results = bo.run()
