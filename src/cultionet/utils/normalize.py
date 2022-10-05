import typing as T
from dataclasses import dataclass

from ..data.datasets import EdgeDataset
from ..data.modules import EdgeDataModule

from tqdm import tqdm
import torch


@dataclass
class NormValues:
    mean: torch.Tensor
    std: torch.Tensor
    max: torch.Tensor


def add_dims(d: torch.Tensor) -> torch.Tensor:
    return d.unsqueeze(0)


def inverse_transform(x: torch.Tensor, data_values: NormValues) -> torch.Tensor:
    """Transforms the inverse of the z-scores"""
    return data_values.std*x + data_values.mean


def get_norm_values(
    dataset: T.Union[EdgeDataset, torch.utils.data.Dataset],
    batch_size: int,
    num_workers: int = 0,
    mean_color: str = 'ffffff',
    sse_color: str = 'ffffff'
) -> NormValues:
    """Normalizes a dataset to z-scores
    """
    if not isinstance(dataset, EdgeDataset):
        data_loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=0
        )

        data_maxs = torch.zeros(3, dtype=torch.float)
        data_sums = torch.zeros(3, dtype=torch.float)
        sse = torch.zeros(3, dtype=torch.float)
        pix_count = 0.0
        with tqdm(
            total=int(len(dataset)/batch_size),
            desc='Calculating means',
            colour=mean_color
        ) as pbar:
            for x, y in data_loader:
                channel_maxs = torch.tensor([x[0, c, ...].max() for c in range(0, x.shape[1])], dtype=torch.float)
                data_maxs = torch.where(channel_maxs > data_maxs, channel_maxs, data_maxs)
                # Sum over all data
                data_sums += x.sum(dim=(0, 2, 3))
                pix_count += (x.shape[2] * x.shape[3])

                pbar.update(1)

        data_means = data_sums / float(pix_count)
        with tqdm(
            total=int(len(dataset)/batch_size),
            desc='Calculating SSEs',
            colour=sse_color
        ) as pbar:
            for x, y in data_loader:
                sse += ((x - data_means.unsqueeze(0)[..., None, None]).pow(2)).sum(dim=(0, 2, 3))

                pbar.update(1)

        data_stds = torch.sqrt(sse / pix_count)

    else:
        # Calculate the means and standard deviations for each channel
        data_module = EdgeDataModule(
            train_ds=dataset,
            batch_size=batch_size,
            num_workers=num_workers
        )

        data_sums = torch.zeros(dataset[0].x.shape[1], dtype=torch.float)
        sse = torch.zeros(dataset[0].x.shape[1], dtype=torch.float)
        pix_count = 0.0

        with tqdm(
            total=int(len(dataset)/batch_size),
            desc='Calculating means',
            colour=mean_color
        ) as pbar:
            for batch in data_module.train_dataloader():
                data_sums += batch.x.sum(dim=0)
                pix_count += batch.x.shape[0]

                pbar.update(1)

        data_means = data_sums / float(pix_count)
        with tqdm(
            total=int(len(dataset)/batch_size),
            desc='Calculating SSEs',
            colour=sse_color
        ) as pbar:
            for batch in data_module.train_dataloader():
                sse += ((batch.x - add_dims(data_means)).pow(2)).sum(dim=0)

                pbar.update(1)

        data_stds = torch.sqrt(sse / pix_count)
        data_maxs = torch.zeros_like(data_means)

    norm_values = NormValues(
        mean=data_means,
        std=data_stds,
        max=data_maxs
    )

    return norm_values
