Use in NNI
==========

This page describes how Tuun can be used as a *custom tuner* in `NNI (neural network
intelligence) <https://github.com/microsoft/nni>`_.

***************
Overview
***************

Like the default tuners provided in NNI, Tuun is a tuner for efficient Bayesian hyperparameter optimization via uncertainty modeling and can be easily adapted to the NNI environment.

***************
Configuration
***************

Tuun adopts a personalized configuration that is specified in the optional 'classArgs' field with several default arguments (e.g., 'optimization_mode') as specified in the built-in tuners/advisors in [NNI](https://nni.readthedocs.io/en/latest/Overview.html). Here is a working example of how configuration is set for one hyperparameter (e.g., learning rate in a range beween 1e-5 and 5e-4) in the config file of NNI, which uses Tuun to optimize a function via Bayesian optimization with a Gaussian process (GP) model. Note that the only changes are all about the tuner section in the config file. 

.. code-block:: yaml

    tuner:
      codeDir: $PATHTUUN/tuun-dev/tuun # PATHTUUN is the path where you have the source code of tuun
      classFileName: nni_tuner.py
        className: TuunTuner
        classArgs:
        optimization_mode: minimize
        tuun_config: {
            'backend': 'probo',
            'model_config': {'name': 'simplegp', 'ls': 3.0, 'alpha': 1.5, 'sigma': 1e-5},
            'acqfunction_config': {'name': 'default', 'acq_str': 'ei'},
            'acqoptimizer_config': {'name': 'default', 'max_iter': 200},
            } 


If you have more than one hyperparameter (e.g., we addionally tune on model type, which is in a categorical domain), in addition to the change made on the 'search_space' json file, the configuration argument can be tweaked like:

.. code-block:: yaml

    tuner:
      codeDir: /home/zeya.wang/ers/tuun/tuun-dev/tuun
      classFileName: nni_tuner.py
      className: TuunTuner
        className: TuunTuner
        classArgs:
        optimization_mode: minimize
        tuun_config: {
            'seed': 11,
            'model_config': {
                'name': 'simpleproductkernelgp',
                'ls': 3.0,
                'alpha': 1.5,
                'sigma': 1e-5,
                'domain_spec': ['list', 'real'],
            },
            'acqfunction_config': {'name': 'default', 'acq_str': 'ucb', 'n_gen': 500},
            'acqoptimizer_config': {
                'name': 'product',
                'n_iter_bcd': 3,
                'n_init_rs': 3,
                'pao_config_list': [
                    {'name': 'default'},
                    {'name': 'cobyla', 'init_str': 'init_opt'},
                ],
            },
        } 

Notice we change the name of 'acqoptimizer_config' to 'product' and include a list of 'pao_config_list' to specify the pao config corresponding to each hyperparameter. 

***************
Get started
***************

After downloading the source code of TUUN to the specific directory, you only need to change the python path by:

.. code-block:: console

    export PYTHONPATH=<$PATHTUUN> # PATHTUUN is the path where you have the source code of tuun 


Currently in the training code, we follow the NNI examples to read the hyperparameters in each trial. In our above example, you can do:

.. code-block:: python

    tuner_params = nni.get_next_parameter()
    params['bert_model'] = hyper_params[0]   # params is a dict that store all the hyperparameters
    params['learning_rate'] = hyper_params[1]


With this modification, the user can follow the guideline of `NNI <https://nni.readthedocs.io/en/latest/Overview.html>`_ to launch the whole experiment.

***************
Limitations
***************

