![tuun](docs/images/tuun_header.png)

**Tuun** is a toolkit for efficient hyperparameter tuning via uncertainty
modeling, with a focus on flexible model choice, scalability, and use in
distributed settings.

## Installation

Tuun requires Python 3.6+. To install Python dependencies, run:
```
$ pip install -r requirements/requirements.txt
```

Certain models in [`tuun/models/`](tuun/models/) may require [additional
installation](tuun/models/README.md).


## Quick Start
[Here](examples/quick_start/02_minimal_bo.py) is a minimal working example, which uses
Tuun to optimize a function via Bayesian optimization using a Gaussian process (GP)
model.
```python
from tuun import Tuun, SimpleGp, AcqOptimizer, SimpleBo

# define model
model = SimpleGp({'ls': 3.0, 'alpha': 1.5, 'sigma': 1e-5})

# define acqfunction
acqfunction = {'acq_str': 'ei'}

# define acqoptimizer
acqoptimizer = AcqOptimizer(domain={'min_max': [(-5, 5)]})

# define tuun
tu = Tuun(model, acqfunction, acqoptimizer)

# define function
f = lambda x: x ** 4 - x ** 2 + 0.1 * x

# define and run BO
bo = SimpleBo(tu, f, params={'n_iter': 20})
results = bo.run()
```
This should find a minima at roughly: 𝑥\*=−0.73, 𝑓(𝑥\*)=−0.32.


## Example

To run a simple Tuun example:
```
$ python examples/quick_start/02_minimal_bo.py
```
