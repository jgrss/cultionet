from pathlib import Path
from dataclasses import dataclass
import shutil
import typing as T
import enum


class Destinations(enum.Enum):
    train = 'train'
    test = 'test'


@dataclass
class ProjectPaths:
    project_path: Path
    image_path: Path
    composite_path: Path
    proba_path: Path
    figure_path: Path
    data_path: Path
    process_path: Path
    test_process_path: Path
    ckpt_path: Path
    train_path: Path
    test_path: Path
    predict_path: Path
    edge_training_path: Path
    ckpt_file: Path
    loss_file: Path
    norm_file: Path

    def remove_train_path(self):
        if self.process_path.is_dir():
            for fn in self.process_path.glob('*.pt'):
                fn.unlink()
            shutil.rmtree(str(self.process_path))
        self.process_path.mkdir(exist_ok=True, parents=True)

    def get_process_path(self, destination: str) -> Path:
        return self.process_path if destination == Destinations.train.name else self.test_process_path


def setup_paths(
    project_path: T.Union[str, Path, bytes],
    append_ts: T.Optional[bool] = True,
    ckpt_name: T.Optional[str] = 'last.ckpt'
) -> ProjectPaths:
    project_path = Path(project_path)
    image_path = project_path / 'time_series_vars' if append_ts else project_path
    composite_path = project_path.parent / 'composites'
    proba_path = project_path.parent / 'composites_probas'
    figure_path = project_path / 'figures'
    data_path = project_path / 'data'
    ckpt_path = project_path / 'ckpt'
    train_path = data_path / 'train'
    test_path = data_path / 'test'
    process_path = train_path / 'processed'
    test_process_path = test_path / 'processed'
    predict_path = data_path / 'predict'
    edge_training_path = project_path / 'user_train'
    ckpt_file = ckpt_path / ckpt_name
    loss_file = ckpt_path / 'losses.npy'
    norm_file = ckpt_path / 'last.norm'

    for p in [
        proba_path,
        figure_path,
        data_path,
        process_path,
        test_process_path,
        ckpt_path
    ]:
        p.mkdir(exist_ok=True, parents=True)

    return ProjectPaths(
        project_path=project_path,
        image_path=image_path,
        composite_path=composite_path,
        proba_path=proba_path,
        figure_path=figure_path,
        data_path=data_path,
        process_path=process_path,
        test_process_path=test_process_path,
        ckpt_path=ckpt_path,
        train_path=train_path,
        test_path=test_path,
        predict_path=predict_path,
        edge_training_path=edge_training_path,
        ckpt_file=ckpt_file,
        loss_file=loss_file,
        norm_file=norm_file
    )
