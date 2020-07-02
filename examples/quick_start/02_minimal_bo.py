from tuun import Tuun, AcqOptimizer, SimpleGp

# define model
model = SimpleGp({'ls': 3., 'alpha': 1.5, 'sigma': 1e-5})

# define acqfunction
acqfunction = {'acq_str': 'ucb'}

# define acqoptimizer
acqoptimizer = AcqOptimizer(domain={'min_max': [(-5, 5)]})

# define dataset
data = {'x': [], 'y': []}

# define tuun
tu = Tuun(model, acqfunction, acqoptimizer, data)

# define function
f = lambda x: x ** 4 - x ** 2 + 0.1 * x

# BO loop
for i in range(20):

    # Choose next x and query f(x)
    x = tu.get()[0]
    y = f(x)

    # Update data and reset data in tu
    data['x'].append(x)
    data['y'].append(y)
    tu.set_data(data)

    # Print iter info
    bsf = min(data['y'])
    print('i: {},    x: {:.4f},\ty: {:.4f},\tBSF: {:.4f}'.format(i, x, y, bsf))
