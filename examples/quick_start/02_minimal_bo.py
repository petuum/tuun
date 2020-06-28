from tuumbo import Tuumbo, RealDomain, AcqOptimizer, SimpleGp

# define dataset
data = {'x': [], 'y': []}

# define model
model = SimpleGp({'ls': 1.5, 'alpha': 2., 'sigma': 1e-5})

# define acqfunction
acqfunction = {'acq_str': 'ucb'}

# define acqoptimizer
acqoptimizer = AcqOptimizer(domain={'min_max': [(-5, 5)]})

# define tuumbo
tu = Tuumbo(data, model, acqfunction, acqoptimizer)

# define function
f = lambda x: x**4 - x**2 + 0.1*x

# BO loop
for i in range(30):

    # Choose next x and query f(x)
    x = tu.get()[0]
    y = f(x)

    # Update data and reset data in tu
    data['x'].append(x)
    data['y'].append(y)
    tu.set_data(data)

    # Print iter info
    print('iter: {}, x: {}, y: {}'.format(i, x, y))