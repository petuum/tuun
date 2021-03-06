<p align="center"><img src="docs/images/tuun_logo.svg" width=280 /></p>

<p align="center">
    &emsp;&emsp;
    <a href="https://petuum.github.io/tuun">Documentation</a> and
    <a href="https://github.com/petuum/tuun/tree/master/examples">Examples</a>
</p>

**Tuun** is a toolkit for efficient hyperparameter tuning via uncertainty
modeling, with a focus on flexible model choice, scalability, and use in
distributed settings.


## Installation

Tuun requires Python 3.6+. To install all dependencies for development, clone and `cd`
into this repo, and run:
```
$ pip install -r requirements/requirements_dev.txt
```

For the full functionality of Tuun, a [Stan](https://mc-stan.org/) model must also be
compiled (this takes roughly 1 minute) by running:
```
$ python tuun/probo/models/stan/compile_models.py -m gp_distmat_fixedsig
```


## Quick Start

**Note:** Tuun is still in the early stages and the following API and functionality may
undergo changes.

[Here](examples/quick_start/minimize_01.py) is a minimal working example, which uses
Tuun to optimize a function over a one-dimensional Euclidean search space with bounds
[-5, 5].

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
page](https://petuum.github.io/tuun/search_space.html) for more details on defining
different search spaces for Tuun.

Tuun also allows for fine-grained configuration of individual components and search
spaces, as shown in the [example](examples/quick_start/minimize_11.py) below.

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
docs page](https://petuum.github.io/tuun/configure.html) for more details on possible
configurations.

[See here](examples/quick_start/) for a larger set of quick start examples for Tuun.


## Use in NNI

Tuun can be used as a custom tuner in [NNI (neural network
intelligence)](https://github.com/microsoft/nni), which allows for visualization and
experiment management. See [this docs page](https://petuum.github.io/tuun/nni.html) for
more details, and [this directory](examples/nni_simple_2d) for a minimal example.


## Examples

See a few examples of Tuun [here](examples/), including the setup for various benchmark
functions, different types of search spaces, and use within NNI.

The plots below show a couple examples of Tuun, along with other tuning algorithms, on
benchmark functions.
<p align="center">
    <img src="docs/images/hartmann6.svg" alt="Hartmann 6 Dimensions" width="48%">
    &nbsp; &nbsp;
    <img src="docs/images/branin40.svg" alt="Branin 40 Dimensions" width="48%">
</p>


## Affiliations

Tuun is part of the [CASL project](http://casl-project.ai/).
<p align="left"><img src="docs/images/casl_logo.svg" width=280></p>

Companies and universities using and developing Tuun.
<p align="top">
    <img src="docs/images/cmu_logo.png" width="15%">
    &nbsp; &nbsp; &nbsp; &nbsp;
    <img src="docs/images/petuum_logo.svg" width="24%">
    &nbsp; &nbsp;
    <img src="docs/images/stanford_logo.png" width="20%">
</p>
