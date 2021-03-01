# Models for use in ProBO

This directory contains models for use in ProBO.

Certain models may require additional dependencies, for example:
* [PyStan](https://pystan.readthedocs.io/)
    - Used by `StanGp`, `StanProductGp`, and `StanDistmatGp`.
* [GPyTorch](https://docs.gpytorch.ai/)
    - Used by `GpytorchGp` and `GpytorchProductGp`.
* [Pyro](http://docs.pyro.ai/)
    - Used by `PyroBnn`.
* [Scikit-learn](https://scikit-learn.org/stable/user_guide.html)
    - Used by `SklearnPenn`, `SklearnHpnn`, and `SklearnPerf`.
* [TensorFlow Probability](https://www.tensorflow.org/probability/api_docs/python/tfp)
    - Used by `TfpBnn`.

To install dependencies for all of the above ProBO models, run:
```bash
$ pip install -r requirements/requirements_all.txt
```

To install dependencies for individual models separately, follow the instructions below.


## Stan Models

To install ProBO with dependencies for Stan models, run:
```bash
$ pip install -r requirements/requirements_stan.txt
```
Additionally, Stan models must be compiled on a given machine before use. To compile the
Stan models used in this directory, see the model code in [`stan/`](stan/) and
[compilation instructions](stan/README.md).

As an example, to compile models for the `StanGp` class, run:
```bash
$ python probo/models/stan/compile_models.py -m gp_fixedsig
```


## GPyTorch Models

To install ProBO with dependencies for GPyTorch models, run:
```bash
$ pip install -r requirements/requirements_gpytorch.txt
```


## Pyro Models

To install ProBO with dependencies for Pyro models, run:
```bash
$ pip install -r requirements/requirements_pyro.txt
```


## TensorFlow Probability Models

To install ProBO with dependencies for TensorFlow Probability models, run:
```bash
$ pip install -r requirements/requirements_tfp.txt
```
