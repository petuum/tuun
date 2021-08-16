# Compiling Stan Models for ProBO

This directory contains code for models implemented in the Stan language.

Note that Stan models must first be compiled in order to be used on a given
machine.

To compile a given Stan model, use [`compile_models.py`](compile_models.py). This will
compile the model (which takes around one minute) and then save the model as a pickle
file in the [`model_pkls/`](model_pkls/) directory. These pickled models are then
automatically used by code in the [`../models/`](../) directory.

As an example, run
```bash
$ python probo/models/stan/compile_models.py -m gp_fixedsig
```
to compile the model contained in [`gp_fixedsig.py`](gp_fixedsig.py) (a hierarchical GP
Stan model), which will save this compiled model as `model_pkls/gp_fixedsig.pkl`, and
will be used by the [`StanGp`](../gp_stan.py) class.
