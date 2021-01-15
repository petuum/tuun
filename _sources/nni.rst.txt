Use in NNI
==========

This page describes how Tuun can be used as a *custom tuner* in `NNI (neural network
intelligence) <https://github.com/microsoft/nni>`_.

***************
Overview
***************

Similar to the default tuners provided in NNI for hyperparameter tuning through
maximizing/minimizing a training objective, Tuun is a tuner that implements efficient
Bayesian hyperparameter optimization via uncertainty modeling and can be easily adapted
to the NNI environment.

***************
Configuration
***************

Tuun adopts a personalized configuration that is specified in the optional
:code:`classArgs` field with several default arguments (e.g., :code:`optimization_mode`)
as specified in the built-in tuners/advisors in `NNI
<https://nni.readthedocs.io/en/latest/Overview.html>`_. When using Tuun with NNI,
instead of specifying a domain specification or search space with Tuun, we specify the
range/specification of hyperparameters in a file :code:`search_space.json`. Please
see `this link <https://nni.readthedocs.io/en/stable/Tutorial/SearchSpaceSpec.html>`_
for a more general instruction about how to set up this file.  Here is a working example
of how configuration is set for one hyperparameter (e.g., learning rate) in the config
file of NNI, which uses Tuun to optimize a function via Bayesian optimization with a
Gaussian process (GP) model. Note that the only changes involve the tuner section in the
config file.

.. code-block:: yaml

    tuner:
      codeDir: $PATHTUUN/tuun-dev/tuun # PATHTUUN is the path where you have the source code of tuun
      classFileName: nni_tuner.py
        className: TuunTuner
        classArgs:
        optimization_mode: minimize
        tuun_config: {
            'seed': 1234,
            'backend': 'probo',
            'acqfunction_config': {'name': 'default', 'acq_str': 'ei'},
            } 


To configure Tuun to use a Stan model, you can instead using the following in the tuner
section:

.. code-block:: yaml

    tuner:
      codeDir: $PATHTUUN/tuun-dev/tuun # PATHTUUN is the path where you have the source code of tuun
      classFileName: nni_tuner.py
        className: TuunTuner
        classArgs:
        optimization_mode: minimize
        tuun_config: {
            'seed': 1234,
            'backend': 'probo',
            'model_config': {'name': 'standistmatgp'},
            'acqfunction_config': {'name': 'default', 'acq_str': 'ei'},
            }


Alternative acquisition optimization algorithms can also be specified, as follows:

.. code-block:: yaml

    tuner:
      codeDir: $PATHTUUN/tuun-dev/tuun # PATHTUUN is the path where you have the source code of tuun
      classFileName: nni_tuner.py
        className: TuunTuner
        classArgs:
        optimization_mode: minimize
        tuun_config: {
            'seed': 1234,
            'backend': 'probo',
            #'model_config': {'name': 'standistmatgp'}, # for Stan model
            'acqfunction_config': {'name': 'default', 'acq_str': 'ei'},
            'acqoptimizer_config': {'name': 'neldermead', 'n_init_rs': 5}, # Nelder-Mead algorithm with 5 initial uniform random samples
            }


***************
Get started
***************

After downloading the source code of Tuun to the specific directory, you only need to change the python path by:

.. code-block:: console

    export PYTHONPATH=<$PATHTUUN> # PATHTUUN is the path where you have the source code of Tuun 


Currently in the training code, we follow the NNI examples to report the results and read the hyperparameters in each trial (`link <https://nni.readthedocs.io/en/stable/TrialExample/Trials.html>`_). In our above example, for reading hyperparameters in each trial you can do:

.. code-block:: python

    hypers_dict = nni.get_next_parameter()
    hypers_list = list(hypers_dict.values())
    params['bert_model'] = hypers_list[0]   # params is a dict that stores all the hyperparameters
    params['learning_rate'] = hypers_list[1]


With this modification, you can follow the `guidelines of NNI
<https://nni.readthedocs.io/en/stable/Tutorial/QuickStart.html>`_ to launch the full
experiment.

***************
Limitations
***************

