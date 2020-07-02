from tuun import Tuun, SimpleGp, AcqOptimizer, SimpleBo

# define model
model = SimpleGp({'ls': 3.0, 'alpha': 1.5, 'sigma': 1e-5})

# define acqfunction
acqfunction = {'acq_str': 'ei'}

# define acqoptimizer
acqoptimizer = AcqOptimizer(domain={'min_max': [(-5, 5)]})

# define tuun
tu = Tuun(model, acqfunction, acqoptimizer, data=None)

# define function
f = lambda x: x ** 4 - x ** 2 + 0.1 * x

# define and run BO
bo = SimpleBo(tu, f, params={'n_iter': 20})
results = bo.run()
