Tuning Benchmarks
=================

This tutorial demonstrates how to use Tuun to optimize a few popular benchmark
functions.

The following script shows how to run Tuun on the 6 dimensional Hartmann function. The
full code for this example can be `found here
<https://github.com/petuum/tuun-dev/blob/master/examples/hartmann/00_stan.py>`_.

.. code-block:: python

  from tuun.main import Tuun
  from hartmann import hartmann6

  # configure Tuun
  config = {
      'seed': 11,
      'acqfunction_config': {'name': 'default', 'acq_str': 'ei', 'n_gen': 500},
      'model_config': {'name': 'standistmatgp'},
  }
  tu = Tuun(config)

  # set search space
  search_space_list = [('real', [0, 1])] * 6
  tu.set_config_from_list(search_space_list)

  # define function to optimize
  f = hartmann6

  # minimize function over search space
  result = tu.minimize_function(f, 60)


The following script shows how to run Tuun on a 40 dimensional version of the Branin
function. The full code for this example can be `found here
<https://github.com/petuum/tuun-dev/blob/master/examples/branin40/00_stan.py>`_.

.. code-block:: python

  import numpy as np
  from tuun.main import Tuun
  from examples.branin.branin import branin

  config = {
      'seed': 11,
      'acqfunction_config': {'name': 'default', 'acq_str': 'ei', 'n_gen': 500},
      'acqoptimizer_config': {'name': 'neldermead', 'n_init_rs': 10},
      'model_config': {'name': 'standistmatgp'},
  }
  tu = Tuun(config)

  search_space_list = [('real', [-5, 10]), ('real', [0, 15])] * 20
  tu.set_config_from_list(search_space_list)

  f = lambda x: np.sum([branin(x[2 * i : 2 * i + 2]) for i in range(20)])

  result = tu.minimize_function(f, 120)


The plots below show a couple examples of Tuun, along with other tuning algorithms, on
the above benchmark functions.

.. image:: ../images/hartmann6.png
   :width: 300
   :alt: Hartmann6 benchmark function
.. image:: ../images/branin40.png
   :width: 300
   :alt: Hartmann6 benchmark function
