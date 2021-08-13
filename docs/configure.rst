Configure Components
====================

Backend
-------
Tuun supports two backend hyperparameter optimzation libraries, Dragonfly and ProBO.

Dragonfly
^^^^^^^^^
`Dragonfly <https://github.com/dragonfly/dragonfly/>`_ is an open source Bayesian Optimization library for scalable machine learning. Refer to its documentation `here <https://dragonfly-opt.readthedocs.io/en/master/>`_.

ProBO
^^^^^
ProBO is a library for flexible Bayesian optimization with customizable models. It
allows a user to define a custom model in a probabilistic programming language (PPL) or
modeling package, and automatically deploy this model in Bayesian optimization.
Features include support for models implemented with many popular modeling packages,
acquisition optimization with a variety of optimizers, and modular components for
diverse BO tasks over common and custom domains.

Recommendation Algorithm
------------------------
Tuun also provides transfer-learning-based recommendation algorithm for improving search using the ProBO backend.

Amortized Auto Tuning
^^^^^^^^^^^^^^^^^^^^^
Amortized Auto Tuning (AT2) is based on the paper `Amortized Auto Tuning. <https://arxiv.org/abs/2106.09179/>`_ It relies on a Gaussian Process model with shifted prior mean based on a regressor trained on offline data of similar tasks.
Note that this only works with the ProBO backend.
For configuring AT2, specify a :code:`model_config` with the correct GP model and a :code:`transfer_config` for the regression model to be used.

.. code-block:: yaml

    model_config:  {
        'name': 'stantransfergp',
        'ls': 3.0,
        'alpha': 1.5,
        'transfer_config': {
            'local_path': RegressorWeightsPath,
            'task_name': OnlineTaskName,
            'model_type': TypeOfModel,
            'output_mode': ModeOfOutput,
            'ndim': NumberOfInputDimensions
        }
    }

Ensure that the correct task name is specified and the regression model weights are stored as a tarfile.
For further details on AT2 refer to its page here.

