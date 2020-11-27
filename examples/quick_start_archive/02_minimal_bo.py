from tuun import SimpleGp, AcqOptimizer, SimpleBo

# define model
model = SimpleGp({'ls': 3.0, 'alpha': 1.5, 'sigma': 1e-5})

# define acqfunction
acqfunction = {'acq_str': 'ei'}

# define acqoptimizer
acqoptimizer = AcqOptimizer(domain={'min_max': [(-5, 5)]})

# define function
f = lambda x: x[0] ** 4 - x[0] ** 2 + 0.1 * x[0]

# define and run BO
bo = SimpleBo(f, model, acqfunction, acqoptimizer, params={'n_iter': 20})
results = bo.run()
