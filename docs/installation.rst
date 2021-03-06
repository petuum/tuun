Installation
============

Tuun requires Python 3.6+. To install all dependencies for development, first clone Tuun:

.. code-block:: console

  $ git clone https://github.com/petuum/tuun.git

Then :code:`cd` into this repo, and run:

.. code-block:: console

  $ pip install -r requirements/requirements_dev.txt


For the full functionality of Tuun, a `Stan <https://mc-stan.org>`_ model must also be
compiled (this takes roughly 1 minute) by running:

.. code-block:: console

  $ python tuun/probo/models/stan/compile_models.py -m gp_distmat_fixedsig
