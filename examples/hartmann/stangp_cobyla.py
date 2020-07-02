from tuun import Tuun, StanGp, CobylaAcqOptimizer
from examples.hartmann.hartmann import hartmann6, get_hartmann6_domain

# define dataset
data = {'x': [], 'y': []}

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
tu = Tuun(data, model, acqfunction, acqoptimizer, seed=11)

# define function
f = hartmann6

# BO loop
for i in range(50):

    # Choose next x and query f(x)
    x = tu.get()
    y = f(x)

    # Update data and reset data in tu
    data['x'].append(x)
    data['y'].append(y)
    tu.set_data(data)

    # Print iter info
    bsf = min(data['y'])
    print('i: {},    x: {},\ty: {:.4f},\tBSF: {:.4f}'.format(i, x, y, bsf))
