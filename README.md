![tuun](docs/images/tuun_header.png)

**Tuun** is a toolkit for efficient hyperparameter tuning via uncertainty
modeling, with a focus on flexible model choice, scalability, and use in
distributed settings.

## Installation

Tuun requires Python 3.6+. To install Python dependencies, run:
```
$ pip install -r requirements/requirements.txt
```

Certain models in [`tuun/models/`](tuun/model/) may require [additional
installation](tuun/model/README.md).


## Example

To run a simple Tuun example:
```
$ python examples/quick_start/02_minimal_bo.py
```
