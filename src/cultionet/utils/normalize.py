import typing as T
from dataclasses import dataclass
from functools import partial

from cultionet.data.const import FALLOW_CLASS, EDGE_CLASS

from ..data.datasets import EdgeDataset
from ..data.modules import EdgeDataModule
from ..utils.model_preprocessing import TqdmParallel

from tqdm import tqdm
import torch
from joblib import delayed, parallel_backend


@dataclass
class NormValues:
    mean: torch.Tensor
    std: torch.Tensor
    max: torch.Tensor


def add_dim(d: torch.Tensor) -> torch.Tensor:
    return d.unsqueeze(0)


def inverse_transform(x: torch.Tensor, data_values: NormValues) -> torch.Tensor:
    """Transforms the inverse of the z-scores"""
    return data_values.std*x + data_values.mean


# def get_norm_values(
#     dataset: T.Union[EdgeDataset, torch.utils.data.Dataset],
#     batch_size: int,
#     num_workers: int = 0,
#     processes: int = 1,
#     threads_per_worker: int = 1,
#     mean_color: str = 'ffffff',
#     sse_color: str = 'ffffff'
# ) -> NormValues:
#     """Normalizes a dataset to z-scores
#     """
#     if not isinstance(dataset, EdgeDataset):
#         data_loader = torch.utils.data.DataLoader(
#             dataset,
#             batch_size=batch_size,
#             shuffle=True,
#             num_workers=0
#         )

#         data_maxs = torch.zeros(3, dtype=torch.float)
#         data_sums = torch.zeros(3, dtype=torch.float)
#         sse = torch.zeros(3, dtype=torch.float)
#         pix_count = 0.0
#         with tqdm(
#             total=int(len(dataset)/batch_size),
#             desc='Calculating means',
#             colour=mean_color
#         ) as pbar:
#             for x, y in data_loader:
#                 channel_maxs = torch.tensor([x[0, c, ...].max() for c in range(0, x.shape[1])], dtype=torch.float)
#                 data_maxs = torch.where(channel_maxs > data_maxs, channel_maxs, data_maxs)
#                 # Sum over all data
#                 data_sums += x.sum(dim=(0, 2, 3))
#                 pix_count += (x.shape[2] * x.shape[3])

#                 pbar.update(1)

#         data_means = data_sums / float(pix_count)
#         with tqdm(
#             total=int(len(dataset)/batch_size),
#             desc='Calculating SSEs',
#             colour=sse_color
#         ) as pbar:
#             for x, y in data_loader:
#                 sse += ((x - data_means.unsqueeze(0)[..., None, None]).pow(2)).sum(dim=(0, 2, 3))

#                 pbar.update(1)

#         data_stds = torch.sqrt(sse / pix_count)

#     else:
#         # Calculate the means and standard deviations for each channel
#         data_module = EdgeDataModule(
#             train_ds=dataset,
#             batch_size=batch_size,
#             num_workers=num_workers
#         )

#         def get_info(x: torch.Tensor) -> T.Tuple[torch.Tensor, int]:
#             # class_counts = torch.zeros(EDGE_CLASS)
#             # edge_counts = torch.zeros(2)
#             # class_counts[0] = ((y == 0) | (y == EDGE_CLASS)).sum()
#             # class_counts[1] = (y == 1).sum()
#             # if FALLOW_CLASS != -999:
#             #     for i in range(2, EDGE_CLASS):
#             #         class_counts[i] = (y == i).sum()
#             # edge_counts[0] = (y != EDGE_CLASS).sum()
#             # edge_counts[1] = (y == EDGE_CLASS).sum()

#             return x.sum(dim=0), x.shape[0]#, class_counts, edge_counts

#         with parallel_backend(
#             backend='loky',
#             n_jobs=processes,
#             inner_max_num_threads=threads_per_worker
#         ):
#             with TqdmParallel(
#                 tqdm_kwargs={
#                     'total': int(len(dataset) / batch_size),
#                     'desc': 'Calculating means',
#                     'colour': mean_color
#                 }
#             ) as pool:
#                 results = pool(
#                     delayed(get_info)(batch.x) for batch in data_module.train_dataloader()
#                 )
#         data_sums, pix_count = list(map(list, zip(*results)))

#         data_sums = torch.stack(data_sums).sum(dim=0)
#         pix_count = torch.tensor(pix_count).sum()
#         # class_counts = torch.stack(class_counts).sum(dim=0)
#         # edge_counts = torch.stack(edge_counts).sum(dim=0)
#         data_means = data_sums / float(pix_count)

#         def get_sse(x_mu: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
#             return ((x - x_mu).pow(2)).sum(dim=0)

#         sse_partial = partial(get_sse, add_dim(data_means))

#         with parallel_backend(
#             backend='loky',
#             n_jobs=processes,
#             inner_max_num_threads=threads_per_worker
#         ):
#             with TqdmParallel(
#                 tqdm_kwargs={
#                     'total': int(len(dataset) / batch_size),
#                     'desc': 'Calculating SSEs',
#                     'colour': sse_color
#                 }
#             ) as pool:
#                 sses = pool(
#                     delayed(sse_partial)(batch.x) for batch in data_module.train_dataloader()
#                 )

#         sses = torch.stack(sses).sum(dim=0)
#         data_stds = torch.sqrt(sses / float(pix_count))
#         data_maxs = torch.zeros_like(data_means)

#     norm_values = NormValues(
#         mean=data_means,
#         std=data_stds,
#         max=data_maxs
#     )

#     return norm_values


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
                sse += ((batch.x - add_dim(data_means)).pow(2)).sum(dim=0)

                pbar.update(1)

        data_stds = torch.sqrt(sse / pix_count)
        data_maxs = torch.zeros_like(data_means)

    norm_values = NormValues(
        mean=data_means,
        std=data_stds,
        max=data_maxs
    )

    return norm_values
