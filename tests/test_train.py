import tempfile

import joblib
import pytorch_lightning as pl
import torch

import cultionet
from cultionet.data.data import Data
from cultionet.data.datasets import EdgeDataset
from cultionet.enums import AttentionTypes, ModelTypes, ResBlockTypes
from cultionet.model import CultionetParams
from cultionet.utils.project_paths import setup_paths

pl.seed_everything(100)


def create_data() -> Data:
    num_channels = 2
    num_time = 12
    height = 10
    width = 10

    x = torch.rand(
        (1, num_channels, num_time, height, width),
        dtype=torch.float32,
    )
    bdist = torch.rand((1, height, width), dtype=torch.float32)
    y = torch.randint(low=0, high=3, size=(1, height, width))

    lat_left, lat_bottom, lat_right, lat_top = 1, 2, 3, 4

    batch_data = Data(
        x=x,
        y=y,
        bdist=bdist,
        left=torch.tensor([lat_left], dtype=torch.float32),
        bottom=torch.tensor([lat_bottom], dtype=torch.float32),
        right=torch.tensor([lat_right], dtype=torch.float32),
        top=torch.tensor([lat_top], dtype=torch.float32),
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
            batch_data = create_data()
            batch_data.to_file(data_path)

        dataset = EdgeDataset(
            ppaths.train_path,
            processes=0,
            threads_per_worker=1,
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
            num_classes=2,
            edge_class=2,
            model_type=ModelTypes.TOWERUNET,
            res_block_type=ResBlockTypes.RESA,
            attention_weights=AttentionTypes.SPATIAL_CHANNEL,
            activation_type="SiLU",
            dilations=[1, 2],
            dropout=0.2,
            deep_supervision=True,
            pool_attention=False,
            pool_by_max=True,
            repeat_resa_kernel=False,
            batchnorm_first=True,
            epochs=1,
            device="cpu",
            devices=1,
            precision="16-mixed",
        )
        cultionet.fit(cultionet_params)
