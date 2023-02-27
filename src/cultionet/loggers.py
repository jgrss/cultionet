from pathlib import Path
import typing as T

from pytorch_lightning.loggers.logger import Logger
import csv


class BatchMetricsLogger(Logger):
    def __init__(self, log_path: Path):
        super().__init__()

        self.log_path = log_path
        self.header = ['id', 'epoch', 'step', 'metric']
        self._write_row(self.header, mode='w')

    def _write_row(self, row: T.Sequence[str], mode: str) -> None:
        with open(self.log_path, mode=mode) as f:
            writer = csv.writer(f)
            writer.writerow(row)

    @property
    def name(self):
        return 'batch_metric_logs'

    @property
    def version(self):
        # Return the experiment version, int or str.
        return "0.1"

    def log_hyperparams(self, params):
        # params is an argparse.Namespace
        # your code to record hyperparameters goes here
        pass

    def log_metrics(self, metrics: dict, step: int):
        # metrics is a dictionary of metric names and values
        # your code to record metrics goes here
        if 'val_loss' in metrics:
            import ipdb; ipdb.set_trace()
        # output_path = Path(self.logger.save_dir) / ''
        # self._write_row()
