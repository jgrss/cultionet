import logging

from joblib import Parallel
from tqdm import tqdm


class ColorFormatter(logging.Formatter):
    """Reference:
    https://stackoverflow.com/questions/384076/how-can-i-color-python-logging-output
    """

    grey = "\x1b[38;20m"
    yellow = "\x1b[33;20m"
    red = "\x1b[31;20m"
    bold_red = "\x1b[31;1m"
    reset = "\x1b[0m"
    format = "%(asctime)s [%(levelname)s] [%(name)s.py:%(lineno)d] %(message)s"

    FORMATS = {
        logging.DEBUG: grey + format + reset,
        logging.INFO: grey + format + reset,
        logging.WARNING: yellow + format + reset,
        logging.ERROR: red + format + reset,
        logging.CRITICAL: bold_red + format + reset,
    }

    def format(self, record):
        log_fmt = self.FORMATS.get(record.levelno)
        formatter = logging.Formatter(log_fmt, datefmt='%Y-%m-%d:%H:%M:%S')

        return formatter.format(record)


def set_color_logger(logger_name):
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.INFO)
    formatter = ColorFormatter()

    file_handler = logging.FileHandler(
        "cultionet.log", mode="w", encoding="utf-8"
    )
    file_handler.setLevel(logging.WARNING)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(logging.INFO)
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    return logger


class ParallelProgress(Parallel):
    """
    Source:
        https://stackoverflow.com/questions/37804279/how-can-we-use-tqdm-in-a-parallel-execution-with-joblib
    """

    def __init__(self, tqdm_params: dict, **kwargs):
        self.tqdm_params = tqdm_params

        super().__init__(**kwargs)

    def __call__(self, *args, **kwargs):
        with tqdm(**self.tqdm_params) as self._pbar:
            return Parallel.__call__(self, *args, **kwargs)

    def print_progress(self):
        self._pbar.n = self.n_completed_tasks
        self._pbar.refresh()
