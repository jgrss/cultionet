from pathlib import Path
from dataclasses import dataclass
import shutil
import typing as T

from ..enums import Destinations, ModelNames


@dataclass
class ProjectPaths:
    project_path: Path
    image_path: Path
    composite_path: Path
    proba_path: Path
    figure_path: Path
    data_path: Path
    classes_info_path: Path
    process_path: Path
    test_process_path: Path
    predict_process_path: Path
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
        return self.data_path / destination / 'processed'


def setup_paths(
    project_path: T.Union[str, Path, bytes],
    append_ts: T.Optional[bool] = True,
    ckpt_name: T.Optional[str] = ModelNames.CKPT_NAME.value,
) -> ProjectPaths:
    project_path = Path(project_path)
    image_path = (
        project_path / Destinations.TIME_SERIES_VARS.value
        if append_ts
        else project_path
    )
    composite_path = project_path.parent / 'composites'
    proba_path = project_path.parent / 'composites_probas'
    figure_path = project_path / Destinations.FIGURES.value
    data_path = project_path / Destinations.DATA.value
    ckpt_path = project_path / Destinations.CKPT.value
    classes_info_path = data_path / ModelNames.CLASS_INFO.value
    train_path = data_path / Destinations.TRAIN.value
    test_path = data_path / Destinations.TEST.value
    predict_path = data_path / Destinations.PREDICT.value
    process_path = train_path / Destinations.PROCESSED.value
    test_process_path = test_path / Destinations.PROCESSED.value
    predict_process_path = predict_path / Destinations.PROCESSED.value
    edge_training_path = project_path / Destinations.USER_TRAIN.value
    ckpt_file = ckpt_path / ckpt_name
    loss_file = ckpt_path / 'losses.npy'
    norm_file = ckpt_path / ModelNames.NORM.value

    for p in [
        proba_path,
        figure_path,
        data_path,
        process_path,
        test_process_path,
        predict_process_path,
        ckpt_path,
    ]:
        p.mkdir(exist_ok=True, parents=True)

    return ProjectPaths(
        project_path=project_path,
        image_path=image_path,
        composite_path=composite_path,
        proba_path=proba_path,
        figure_path=figure_path,
        data_path=data_path,
        classes_info_path=classes_info_path,
        process_path=process_path,
        test_process_path=test_process_path,
        predict_process_path=predict_process_path,
        ckpt_path=ckpt_path,
        train_path=train_path,
        test_path=test_path,
        predict_path=predict_path,
        edge_training_path=edge_training_path,
        ckpt_file=ckpt_file,
        loss_file=loss_file,
        norm_file=norm_file,
    )
