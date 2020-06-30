This directory contains code for models implemented in the Stan language.

Note that Stan models must first be compiled in order to be used on a given
machine.

To compile a given Stan model (e.g. a hierarchical GP Stan model), use
`compile_models.py`. This will compile (takes about a minute) and save the
model as a pickle file in the `tuun/model/stan/model_pkls` directory. These
pickled models are then automatically used by code in the `models` directory.
As an example, run
```
$ python tuun/model/stan/compile_models.py -m gp_fixedsig
```
to compile the model contained in `tuun/model/stan/gp_fixedsig.py`, and save
this compiled model as `tuun/model/stan/model_pkls/gp_fixedsig.pkl`.
