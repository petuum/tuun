# Models for use in ProBO

This directory contains models for use in ProBO.

Certain models may require additional dependencies, for example:
* `StanGp` dependencies:
    - pystan
* `GpytorchGp` dependencies:
    - gpytorch
* `PyroBnn` dependencies:
    - pyro
* `SklearnPenn` dependencies:
    - scikit-learn 
* `SklearnHpnn` dependencies:
    - scikit-learn 
* `SklearnPerf` dependencies:
    - scikit-learn 

To install dependencies for all of the above ProBO models, run:
```
$ pip install -r requirements/requirements_all.txt
```

To install dependencies for individual models separately, follow the instructions below.


## Stan Models

To install ProBO with dependencies for Stan models, run:
```
$ pip install -r requirements/requirements_stan.txt
```
Additionally, Stan models must be compiled on a given machine before use. To
compile the Stan models used in this directory, see the model code in
[`stan/`](stan/) and [compilation instructions](stan/README.md).


## GPyTorch Models

To install ProBO with dependencies for GPyTorch models, run:
```
$ pip install -r requirements/requirements_gpytorch.txt
```
