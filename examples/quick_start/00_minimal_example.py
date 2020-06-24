from tuumbo import Tuumbo, RealDomain


# define domain
domain = RealDomain(params={'min_max': [(-10, 10)]})

# define initial dataset
n = 5
data = {'x': domain.unif_rand_sample(n), 'y':list(range(n))}

# define model
model = None

# define acqfunction
acqfunction = None

# define acqoptimizer
acqoptimizer = None

# define params
params = None

tu = Tuumbo(data, model, acqfunction, acqoptimizer, params)

print(tu.__dict__)
