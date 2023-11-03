import tempfile
from pathlib import Path

import joblib
import torch
import pytorch_lightning as pl
from torch_geometric.data import Data

import cultionet
from cultionet.data.datasets import EdgeDataset
from cultionet.utils.project_paths import setup_paths


pl.seed_everything(100)


def create_data(site_id: int) -> Data:
    in_channels = 4
    in_time = 12
    height = 10
    width = 10

    x = torch.rand(
        (height * width, in_channels * in_time),
        dtype=torch.float32,
    )
    bdist = torch.rand((height * width,), dtype=torch.float32)
    y = torch.randint(low=0, high=3, size=(height * width,))

    batch_data = Data(
        x=x,
        y=y,
        bdist=bdist,
        height=height,
        width=width,
        ntime=in_time,
        nbands=in_channels,
        zero_padding=0,
        start_year=2020,
        end_year=2021,
        res=10.0,
        train_id=f'{site_id:06d}_2021_1_none',
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
            joblib.dump(batch_data, str(data_path), compress=5)
        dataset = EdgeDataset(
            ppaths.train_path,
            processes=1,
            threads_per_worker=1,
            random_seed=100,
        )
        cultionet.fit(
            dataset=dataset,
            ckpt_file=ppaths.ckpt_file,
            val_frac=0.2,
            batch_size=2,
            load_batch_workers=1,
            filters=32,
            model_type="ResUNet3Psi",
            activation_type="SiLU",
            dilations=[2],
            res_block_type="res",
            attention_weights="spatial_channel",
            deep_sup_dist=False,
            deep_sup_edge=False,
            deep_sup_mask=False,
            learning_rate=1e-3,
            epochs=5,
            device="cpu",
            devices=1,
            precision=32,
        )
