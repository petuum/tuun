Amortized Auto Tuning
=====================

Overview
--------
Amortized Auto-Tuning (AT2) aims to address the problem of automatic hyperparameter tuning using knowledge
transfer from previously trained similar tasks. The key idea is machine learing training tasks that are
similar to each other would also have similarities in their hyperparameter landscapes. Consequently, training
metadata of hyperparameters could be used to bootstrap new but similar training tasks. This has been previously
proposed in the paper  `Amortized Auto Tuning <https://arxiv.org/abs/2106.09179/>`_ where low-fidelity tuning
observations were leveraged to measure inter-task similarity and transfer knowledge from existing to new tasks.

The current implementation in Tuun uses this idea to provide transfer learning with the ProBO backend. It provides
a transfer learning model that incorporates a regression model for predicting the similarity of the new task with
existing tasks and a gaussian process model with a shifted prior mean based on the prediction from the regression
model.

Main Components
---------------

Task Templates
^^^^^^^^^^^^^^
A task template defines a set of tasks with high similarity for which information transfer can lead to improvement
in the optimization process. For each task defined in a task template, a regression model is trained on the
corresponding dataset and model parameters. Tuun can then choose the best regression model for a new online task
and use it to bootstrap the GP model used for hyperparameter recommendation. A task in a task template is defined by
a dataset-model pair corresponding to the model and dataset used for training.

Transfer GP Model
^^^^^^^^^^^^^^^^^
To conduct transfer optimization, Tuun makes use of a Gaussian process (GP) model with a
transfer-learned prior mean function. Specifically, point-prediction regression models
are trained in an offline fashion on previously-collected tuning data from previous
tuning tasks.  During online tuning of a new task, the compatibility of each regression
model with the new task is assessed, and these regression models are then used to define
a single function, which is used as the prior mean function within a GP model. We refer
to this full framework as a GP model with transfer-learned prior mean. Given this prior
mean function, the GP is then used within a Bayesian optimization procedure for
efficient hyperparameter tuning.  This procedure allows for few-shot recommendation of
hyperparameters using only a handful of tuning steps, but also yields a full
optimization procedure that searches for and returns more-accurate global optima given a
larger budget of tuning steps.

Example
-------
We demonstrate an example of using AT2 for choosing the best hyperparameters for image classification on the
`CALTECH256 <https://authors.library.caltech.edu/7694/>`_ dataset. Refer to its code
`here <https://github.com/willieneis/tuun-dev/blob/priormean/examples/transfer/minimize_example_transfertune.py>`_. 
For the image classification task, we use the image classification task template detailed
`here <https://github.com/willieneis/tuun-dev/tree/priormean/templates/image_classification>`_.
The task template consists of 26 image classification datasets that were used to train ResNet34 and ResNet50 models. 
For each of the dataset-model pairs, a Gradient Boosting Regressor (GBR) was trained to predict the validation accuracy given
a hyperparameter configuration. The model weights of the GBR are included as a tar file in the templates directory. The
objective function to be minimized is a lookup table over hyperparameter configurations and the validation error observed. 

.. code-block:: python

    def objective(look_up_table, x):
        """
        Parameters
        ----------
        look_up_table: dict
            dictionary with keys as hyperparamter configuration and
            values as the validation accuracies
        x: list
            hyperparamter configuration
        """
        return look_up_table[tuple(x)]

Next we define a transfer config parameter as part of the config passed to initialize the Tuun object.

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

The :code:`local_path` correpsond to the GBM model weights stored locally. In case the model weights are stored on a remote
repository, add :code:`remote_url` path with the remote path.

Having configured the Tuun object, simply run :code:`minimize_function` to get the best parameter.
