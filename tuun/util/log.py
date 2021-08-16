import os
import sys
import json
import logging
from logging import FileHandler, Formatter, StreamHandler
from pathlib import Path


file_format = '[%(asctime)s] %(levelname)s (%(name)s/%(threadName)s) %(message)s'
stream_format = "%(levelname)s %(message)s"
file_formatter = Formatter(file_format)
stream_formatter = Formatter(stream_format)


class StreamToLogger(object):
    """
    Fake file-like stream object that redirects writes to a logger instance.
    """
    def __init__(self, logger, level):
        self.logger = logger
        self.level = level
        self.linebuf = ''

    def write(self, buf):
        for line in buf.rstrip().splitlines():
            self.logger.log(self.level, line.rstrip())

    def flush(self):
        pass


def init_default_logger(exp_id, config_dict):
    """
    Experiment level logger that stores experiment configuration and initializes sqlite
    db.

    Parameters
    ----------
    exp_id: str
        The experiment ID.
    config_dict: dict
        Configuration dict for Tuun.
    """
    ENV_TUUN_LOG = os.getenv('TUUN_LOG', 'False')
    ENV_TUUN_LOG_PATH = os.path.expanduser(os.getenv('TUUN_LOG_PATH', "~/tuun-log"))
    ENV_TUUN_LOG_STD = os.getenv('TUUN_LOG_STD', 'False')

    if ENV_TUUN_LOG.lower() in ('true', '1', 't'):
        log_path = os.path.join(ENV_TUUN_LOG_PATH, exp_id)
        Path(log_path).mkdir(parents=True, exist_ok=True)
        with open(os.path.join(log_path, 'config.json'), 'w') as f:
            json.dump(config_dict, f)

        if ENV_TUUN_LOG_STD.lower() in ('true', '1', 't'):
            stdout_logger = _get_logger("__tuun__.stdout",
                                        logging.INFO,
                                        os.path.join(log_path, 'stdout'))
            stderr_logger = _get_logger("__tuun__.stderr",
                                        logging.ERROR,
                                        os.path.join(log_path, 'stderr'))
            sys.stdout = StreamToLogger(stdout_logger, logging.INFO)
            sys.stderr = StreamToLogger(stderr_logger, logging.ERROR)
        print(f"Logging to {log_path}")


def _get_logger(name, level, file_path) -> logging.Logger:
    """
    Get specified logger.

    Parameters
    ----------
    name: str
        Name of the logger.
    level: enum
        One of the log levels from logging.
    file_path: str
        Path to save log files.

    Returns
    -------
    logging.Logger
        The logger.
    """
    logger = logging.getLogger(name)
    logger.handlers.clear()
    file_handler = FileHandler(file_path)
    file_handler.setFormatter(file_formatter)
    file_handler.setLevel(level)
    logger.addHandler(file_handler)
    stream_handler = StreamHandler()
    stream_handler.setLevel(level)
    stream_handler.setFormatter(stream_formatter)
    logger.addHandler(stream_handler)
    logger.propagate = False
    return logger
