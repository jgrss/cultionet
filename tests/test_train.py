import json
import subprocess
import tempfile
from pathlib import Path

import lightning as L
import numpy as np
import torch

import cultionet
from cultionet.data import Data
from cultionet.data.datasets import EdgeDataset
from cultionet.enums import AttentionTypes, ModelTypes, ResBlockTypes
from cultionet.model import CultionetParams
from cultionet.utils.project_paths import setup_paths

L.seed_everything(100)
RNG = np.random.default_rng(200)


def create_data(group: int) -> Data:
    num_channels = 2
    num_time = 12
    height = 100
    width = 100

    x = torch.rand(
        (1, num_channels, num_time, height, width),
        dtype=torch.float32,
    )
    bdist = torch.rand((1, height, width), dtype=torch.float32)
    y = torch.randint(low=0, high=3, size=(1, height, width))

    lat_left = RNG.uniform(low=-180, high=180)
    lat_bottom = RNG.uniform(low=-90, high=90)
    lat_right = RNG.uniform(low=-180, high=180)
    lat_top = RNG.uniform(low=-90, high=90)

    batch_data = Data(
        x=x,
        y=y,
        bdist=bdist,
        left=torch.tensor([lat_left], dtype=torch.float32),
        bottom=torch.tensor([lat_bottom], dtype=torch.float32),
        right=torch.tensor([lat_right], dtype=torch.float32),
        top=torch.tensor([lat_top], dtype=torch.float32),
        batch_id=[group],
    )

    return batch_data


def test_train():
    num_data = 10
    with tempfile.TemporaryDirectory() as tmp_path:
        ppaths = setup_paths(tmp_path)
        for i in range(num_data):
            data_path = (
                ppaths.process_path / f'data_{i:06d}_2021_{i:06d}_none.pt'
            )
            batch_data = create_data(i)
            batch_data.to_file(data_path)

        dataset = EdgeDataset(
            ppaths.train_path,
            processes=0,
            random_seed=100,
        )

        cultionet_params = CultionetParams(
            ckpt_file=ppaths.ckpt_file,
            model_name="cultionet",
            dataset=dataset,
            val_frac=0.2,
            batch_size=2,
            load_batch_workers=0,
            hidden_channels=16,
            num_classes=1,
            edge_class=2,
            model_type=ModelTypes.TOWERUNET,
            res_block_type=ResBlockTypes.RESA,
            activation_type="SiLU",
            dilations=[1, 2],
            dropout=0.2,
            pool_by_max=True,
            epochs=1,
            device="cpu",
            devices=1,
            precision="32",
        )

        try:
            cultionet.fit(cultionet_params)
        except Exception as e:
            raise RuntimeError(e)


# def test_train_cli():
#     num_data = 10
#     with tempfile.TemporaryDirectory() as tmp_dir:
#         tmp_path = Path(tmp_dir)
#         ppaths = setup_paths(tmp_path)
#         for i in range(num_data):
#             data_path = (
#                 ppaths.process_path / f'data_{i:06d}_2021_{i:06d}_none.pt'
#             )
#             batch_data = create_data(i)
#             batch_data.to_file(data_path)

#         with open(tmp_path / "data/classes.info", "w") as f:
#             json.dump({"max_crop_class": 1, "edge_class": 2}, f)

#         command = (
#             f"cultionet train -p {str(tmp_path.absolute())} "
#             "--val-frac 0.2 --augment-prob 0.5 --epochs 1 --hidden-channels 16 "
#             "--processes 1 --load-batch-workers 0 --batch-size 2 --dropout 0.2 "
#             "--deep-sup --dilations 1 2 --pool-by-max --learning-rate 0.01 "
#             "--weight-decay 1e-4 --attention-weights spatial_channel --device cpu"
#         )

#         try:
#             subprocess.run(
#                 command,
#                 shell=True,
#                 check=True,
#                 capture_output=True,
#                 universal_newlines=True,
#             )
#         except subprocess.CalledProcessError as e:
#             raise NameError(
#                 "Exit code:\n{}\n\nstdout:\n{}\n\nstderr:\n{}".format(
#                     e.returncode, e.output, e.stderr
#                 )
#             )
