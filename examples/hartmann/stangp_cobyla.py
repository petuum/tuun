from tuun import Tuun, StanGp, CobylaAcqOptimizer, SimpleBo
from examples.hartmann.hartmann import hartmann6, get_hartmann6_domain

# define model
model = StanGp(
    {
        'ndimx': 6,
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
acqoptimizer = CobylaAcqOptimizer({'rand_every': 4}, get_hartmann6_domain())

# define tuun
tu = Tuun(model, acqfunction, acqoptimizer, data=None, seed=11)

# define function
f = hartmann6

# define and run BO
bo = SimpleBo(tu, f, params={'n_iter': 50})
results = bo.run()
