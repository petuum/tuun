![tuun](docs/images/tuun_header.png)

**TUUN** is a toolkit for efficient hyperparameter tuning via uncertainty
modeling, with a focus on flexible model choice, scalability, and use in
distributed settings.

## Installation

Dependencies:
* Python 3.6+
* numpy
* scipy
* matplotlib
* scikit-learn
* pytest

Certain models in [`tuun/models/`](tuun/model/) may require [additional
dependencies](tuun/model/README.md).

## Example

To run a simple TUUN example:
```
$ python examples/quick_start/02_minimal_bo.py
```
