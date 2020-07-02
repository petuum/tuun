from tuun import Tuun, StanGp, AcqOptimizer, SimpleBo
from examples.branin.branin import branin, get_branin_domain

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
acqoptimizer = AcqOptimizer(domain=get_branin_domain())

# define tuun
tu = Tuun(model, acqfunction, acqoptimizer, data=None, seed=11)

# define function
f = branin

# define and run BO
bo = SimpleBo(tu, f, params={'n_iter': 50})
results = bo.run()
