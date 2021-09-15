import sqlite3
import os
from sqlite_utils import Database
from pathlib import Path


def init_db(db_name, exp_id, table_config):
    """Initialize and return MetricDatabase database."""
    ENV_TUUN_LOG = os.getenv('TUUN_LOG', 'False')
    ENV_TUUN_LOG_PATH = os.path.expanduser(os.getenv('TUUN_LOG_PATH', '~/tuun-log'))

    table_config = configure_config(table_config)
    if ENV_TUUN_LOG.lower() in ('true', 't', '1'):
        log_path = os.path.join(ENV_TUUN_LOG_PATH, exp_id)
        Path(log_path).mkdir(parents=True, exist_ok=True)
        db_path = os.path.join(log_path, db_name)
        return MetricDatabase(db_path, table_config)


def configure_config(table_config):
    """Configure and return the config dict."""
    updated_config = {}
    for key, value in table_config.items():
        if value[0] == 'real':
            updated_config[key] = float
        elif value[0] == 'integral':
            updated_config[key] = int
        elif value[0] == 'list':
            updated_config[key] = str
        else:
            raise TypeError(f"Invalid type specified for parameter {key}")
    if 'pk' not in table_config:
        updated_config.update({'pk': int})
    else:
        raise ValueError('Parameter name cannot be pk')
    return updated_config


class MetricDatabase:
    """
    SQLite Database class for storing Tuun metrics.
    """
    def __init__(self, db_path, table_config):
        self._db_path = db_path
        self._parameter_keys = table_config.keys()
        if 'prediction' not in table_config:
            table_config['prediction'] = float
        with sqlite3.connect(self._db_path) as conn:
            db = Database(conn)
            db['metricData'].create(table_config, pk='pk')

    def add_metric(self, metric_data):
        metric_data = self._format_data(metric_data)
        with sqlite3.connect(self._db_path) as conn:
            db = Database(conn)
            db['metricData'].insert_all(metric_data, ignore=True)

    def _format_data(self, data):
        """
        Format metric data into SQL compatible form.

        Parameters
        ----------
        data: argparse.Namespace
            A dict of data with input as 'x' and suggestion as 'y'.

        Returns
        -------
        formatted_data: list
            SQL compatible data.
        """
        formatted_data = []
        for i in range(len(data.x)):
            entry = dict(zip(self._parameter_keys, data.x[i]))
            entry['prediction'] = data.y[i]
            entry['pk'] = hash(tuple(data.x[i]))
            formatted_data.append(entry)
        return formatted_data
