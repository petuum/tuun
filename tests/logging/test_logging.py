import os
import logging
import shutil
import argparse
from unittest import TestCase, mock
from tuun.main import Tuun


class LoggingTests(TestCase):
    @mock.patch.dict(os.environ, {"TUUN_LOG": "true", "TUUN_LOG_PATH": "/tmp/tuun-log", "TUUN_LOG_STD": "false"})
    def test_config_log(self):
        try:
            config = {
                'seed': 11,
                'model_config': {
                    'name': 'standistmatgp',
                    'ndimx': 2,
                    'model_str': 'optfixedsig',
                    'ig1': 4.0,
                    'ig2': 3.0,
                    'n1': 1.0,
                    'n2': 1.0,
                    'sigma': 1e-5,
                    'niter': 70,
                },
                'acqfunction_config': {'name': 'default', 'acq_str': 'ei', 'n_gen': 500},
                'acqoptimizer_config': {'name': 'cobyla', 'rand_every': 4, 'jitter': True},
                'domain_config': {'x1': ('real', [-5.0, 10.0]), 'x2': ('real', [0.0, 15.0])},
            }
            tu = Tuun(config)
            assert os.getenv("TUUN_LOG_PATH") == "/tmp/tuun-log"
            assert os.path.exists(os.path.join("/tmp/tuun-log", tu.exp_id))
            assert os.path.exists(os.path.join("/tmp/tuun-log", tu.exp_id, "config.json"))
        finally:
            shutil.rmtree("/tmp/tuun-log")

    @mock.patch.dict(os.environ, {"TUUN_LOG": "true", "TUUN_LOG_PATH": "/tmp/tuun-log", "TUUN_LOG_STD": "false"})
    def test_db_write(self):
        try:
            config = {
                'seed': 11,
                'model_config': {
                    'name': 'standistmatgp',
                    'ndimx': 2,
                    'model_str': 'optfixedsig',
                    'ig1': 4.0,
                    'ig2': 3.0,
                    'n1': 1.0,
                    'n2': 1.0,
                    'sigma': 1e-5,
                    'niter': 70,
                },
                'acqfunction_config': {'name': 'default', 'acq_str': 'ei', 'n_gen': 500},
                'acqoptimizer_config': {'name': 'cobyla', 'rand_every': 4, 'jitter': True},
                'domain_config': {'x1': ('real', [-5.0, 10.0]), 'x2': ('real', [0.0, 15.0])},
            }
            tu = Tuun(config)
            assert tu.db is not None
            metric_data = argparse.Namespace()
            metric_data.x = [[0.4, 0.5]]
            metric_data.y = [0.6]
            tu.db.add_metric(metric_data)
            assert os.path.exists(os.path.join("/tmp/tuun-log", tu.exp_id, "tuun.sqlite"))
        finally:
            shutil.rmtree("/tmp/tuun-log")

    @mock.patch.dict(os.environ, {"TUUN_LOG": "true", "TUUN_LOG_PATH": "/tmp/tuun-log", "TUUN_LOG_STD": "true"})
    def test_std_logger(self):
        try:
            config = {
                'seed': 11,
                'model_config': {
                    'name': 'standistmatgp',
                    'ndimx': 2,
                    'model_str': 'optfixedsig',
                    'ig1': 4.0,
                    'ig2': 3.0,
                    'n1': 1.0,
                    'n2': 1.0,
                    'sigma': 1e-5,
                    'niter': 70,
                },
                'acqfunction_config': {'name': 'default', 'acq_str': 'ei', 'n_gen': 500},
                'acqoptimizer_config': {'name': 'cobyla', 'rand_every': 4, 'jitter': True},
                'domain_config': {'x1': ('real', [-5.0, 10.0]), 'x2': ('real', [0.0, 15.0])},
            }
            tu = Tuun(config)
            assert logging.getLogger("__tuun__.stdout").hasHandlers()
        finally:
            shutil.rmtree("/tmp/tuun-log")
 
