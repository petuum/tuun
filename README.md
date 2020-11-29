<p align="center"><img src="docs/images/tuun_logo.png" width=280 /></p>

**Tuun** is a toolkit for efficient hyperparameter tuning via uncertainty
modeling, with a focus on flexible model choice, scalability, and use in
distributed settings.

## Installation

Tuun requires Python 3.6+. To install all dependencies for development, run:
```
$ pip install -r requirements/requirements_dev.txt
```

For the full functionality of Tuun, a [Stan](https://mc-stan.org/) model must also be
compiled (this takes roughly 1 minute) by running:
```
$ python tuun/probo/models/stan/compile_models.py -m gp_distmat_fixedsig
```


## Quick Start
Here is a minimal working example, which uses Tuun to optimize a function over a
one-dimensional Euclidean search space with bounds [-5, 5].

```python
from tuun.main import Tuun

# instantiate Tuun
tu = Tuun()

# set search space
search_space = ('real', [-5, 5])
tu.set_config_from_list(search_space)

# define function to optimize
f = lambda x: x[0] ** 4 - x[0] ** 2 + 0.1 * x[0]

# minimize function over search space
result = tu.minimize_function(f, 20)
```
This should find a minima at roughly: 𝑥\*=[−0.73], where 𝑓(𝑥\*)=−0.32.  See [this docs
page](docs/search_space.rst) for more details on defining different search spaces for
Tuun.

Tuun also allows for fine-grained configuration of individual components and search
spaces.

```python
from tuun.main import Tuun

config = {
    # configure tuning backend
    'backend': 'probo',

    # configure model
    'model_config': {'name': 'simplegp', 'ls': 3.0, 'alpha': 1.5, 'sigma': 1e-5},

    # configure acquisition function
    'acqfunction_config': {'name': 'default', 'acq_str': 'ei'},

    # configure acquisition optimizer
    'acqoptimizer_config': {'name': 'default', 'max_iter': 200},

    # configure domain
    'domain_config': ('real', [-5, 5]),
}
tu = Tuun(config)

f = lambda x: x[0] ** 4 - x[0] ** 2 + 0.1 * x[0]
result = tu.minimize_function(f, 20)
```
This should also find a minima at roughly: 𝑥\*=[−0.73], where 𝑓(𝑥\*)=−0.32. See [this
docs page](docs/configure.rst) for more details on possible configurations.



## Examples
<p align="center">
    <img src="docs/images/hartmann6.png" alt="Hartmann 6 Dimensions" width="48%">
    &nbsp; &nbsp;
    <img src="docs/images/branin40.png" alt="Branin 40 Dimensions" width="48%">
</p>
