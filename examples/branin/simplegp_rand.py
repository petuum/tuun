from tuun import Tuun, AcqOptimizer, SimpleGp
from examples.branin.branin import branin, get_branin_domain

# define dataset
data = {'x': [], 'y': []}

# define model
model = SimpleGp({'ls': 3.7, 'alpha': 1.85, 'sigma': 1e-5})

# define acqfunction
acqfunction = {'acq_str': 'ei', 'n_gen': 500}

# define acqoptimizer
acqoptimizer = AcqOptimizer(domain=get_branin_domain())

# define tuun
tu = Tuun(data, model, acqfunction, acqoptimizer, seed=11)

# define function
f = branin

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
