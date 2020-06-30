from tuun import Tuun, RealDomain, StanGp
from examples.branin.branin import branin, get_branin_domain
from examples.branin.acqopt_spo import CobylaAcqOptimizer

# define dataset
data = {'x': [], 'y': []}

# define model
model = StanGp({'ndimx':2, 'model_str':'optfixedsig', 'ig1':4., 'ig2':3.,
                'n1':1., 'n2':1., 'sigma':1e-5, 'niter':70})

# define acqfunction
acqfunction = {'acq_str': 'ei', 'n_gen': 500}

# define acqoptimizer
acqoptimizer = CobylaAcqOptimizer({'rand_every': 4}, get_branin_domain())

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
