import typing as T
from functools import partial
from pathlib import Path

import joblib
import torch
from einops import rearrange
from joblib import delayed, parallel_backend
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from ..data.data import Data
from ..data.utils import collate_fn
from .model_preprocessing import TqdmParallel
from .stats import Quantile, Variance, cache_load_enabled, tally_stats


def add_dim(d: torch.Tensor) -> torch.Tensor:
    return d.unsqueeze(0)


class NormValues:
    def __init__(
        self,
        dataset_mean: torch.Tensor,
        dataset_std: torch.Tensor,
        dataset_crop_counts: torch.Tensor,
        dataset_edge_counts: torch.Tensor,
        num_channels: int,
    ):
        self.dataset_mean = dataset_mean
        self.dataset_std = dataset_std
        self.dataset_crop_counts = dataset_crop_counts
        self.dataset_edge_counts = dataset_edge_counts
        self.num_channels = num_channels

    def __repr__(self):
        return (
            "NormValues("
            f"  dataset_mean={self.dataset_mean},"
            f"  dataset_std={self.dataset_std},"
            f"  dataset_crop_counts={self.dataset_crop_counts},"
            f"  dataset_edge_counts={self.dataset_edge_counts},"
            f"  num_channels={self.num_channels},"
            ")"
        )

    def __call__(self, batch: Data) -> Data:
        return self.transform(batch)

    def transform(self, batch: Data) -> Data:
        r"""Normalizes data to z-scores.

        Args:
            batch (Data): A `torch_geometric` data object.
            data_means (Tensor): The data feature-wise means.
            data_stds (Tensor): The data feature-wise standard deviations.

        z = (x - μ) / σ
        """
        batch_copy = batch.copy()
        batch_copy.x = (
            batch_copy.x - self.dataset_mean.to(device=batch_copy.x.device)
        ) / self.dataset_std.to(device=batch_copy.x.device)

        return batch_copy

    def inverse_transform(self, batch: Data) -> Data:
        """Transforms the inverse of the z-scores."""
        batch_copy = batch.copy()
        batch_copy.x = self.dataset_std.to(
            device=batch_copy.x.device
        ) * batch_copy.x + self.dataset_mean.to(device=batch_copy.x.device)

        return batch_copy

    @property
    def data_dict(self) -> dict:
        return {
            'dataset_mean': self.dataset_mean,
            'dataset_std': self.dataset_std,
            'dataset_crop_counts': self.dataset_crop_counts,
            'dataset_edge_counts': self.dataset_edge_counts,
            'num_channels': self.num_channels,
        }

    def to_file(
        self, filename: T.Union[Path, str], compress: str = 'zlib'
    ) -> None:
        joblib.dump(
            self.data_dict,
            filename,
            compress=compress,
        )

    @classmethod
    def from_file(cls, filename: T.Union[Path, str]) -> "NormValues":
        return cls(**joblib.load(filename))

    @classmethod
    def from_dataset(
        cls,
        dataset: Dataset,
        batch_size: int,
        class_info: T.Dict[str, int],
        num_workers: int = 0,
        processes: int = 1,
        threads_per_worker: int = 1,
        centering: str = 'median',
        mean_color: str = '#ffffff',
        sse_color: str = '#ffffff',
    ) -> "NormValues":
        """Normalizes a dataset to z-scores."""

        if not isinstance(dataset, Dataset):
            data_loader = DataLoader(
                dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=0,
                collate_fn=collate_fn,
            )

            data_maxs = torch.zeros(3, dtype=torch.float)
            data_sums = torch.zeros(3, dtype=torch.float)
            sse = torch.zeros(3, dtype=torch.float)
            pix_count = 0.0
            with tqdm(
                total=int(len(dataset) / batch_size),
                desc='Calculating means',
                colour=mean_color,
            ) as pbar:
                for x, y in data_loader:
                    channel_maxs = torch.tensor(
                        [x[0, c, ...].max() for c in range(0, x.shape[1])],
                        dtype=torch.float,
                    )
                    data_maxs = torch.where(
                        channel_maxs > data_maxs, channel_maxs, data_maxs
                    )
                    # Sum over all data
                    data_sums += x.sum(dim=(0, 2, 3))
                    pix_count += x.shape[2] * x.shape[3]

                    pbar.update(1)

            data_means = data_sums / float(pix_count)
            with tqdm(
                total=int(len(dataset) / batch_size),
                desc='Calculating SSEs',
                colour=sse_color,
            ) as pbar:
                for x, y in data_loader:
                    sse += (
                        (x - data_means.unsqueeze(0)[..., None, None]).pow(2)
                    ).sum(dim=(0, 2, 3))

                    pbar.update(1)

            data_stds = torch.sqrt(sse / pix_count)

        else:
            data_loader = DataLoader(
                dataset,
                batch_size=batch_size,
                num_workers=num_workers,
                shuffle=False,
                collate_fn=collate_fn,
            )

            if centering == 'median':
                stat_var = Variance(method='median')
                stat_q = Quantile(r=1024 * 6)
                tmp_cache_path = Path.home().absolute() / '.cultionet'
                tmp_cache_path.mkdir(parents=True, exist_ok=True)
                var_data_cache = tmp_cache_path / '_var.npz'
                q_data_cache = tmp_cache_path / '_q.npz'
                crop_counts = torch.zeros(
                    class_info['max_crop_class'] + 1
                ).long()
                edge_counts = torch.zeros(2).long()
                with cache_load_enabled(True):
                    with tqdm(
                        total=int(len(dataset) / batch_size),
                        desc='Calculating dataset statistics',
                    ) as pbar:
                        for batch in tally_stats(
                            stats=(stat_var, stat_q),
                            loader=data_loader,
                            caches=(var_data_cache, q_data_cache),
                        ):
                            # Stack samples
                            x = rearrange(batch.x, 'b c t h w -> (b t h w) c')

                            # Update the stats
                            stat_var.add(x)
                            stat_q.add(x)

                            # Update counts
                            crop_counts[0] += (
                                (batch.y == 0)
                                | (batch.y == class_info['edge_class'])
                            ).sum()
                            for i in range(1, class_info['edge_class']):
                                crop_counts[i] += (batch.y == i).sum()

                            edge_counts[0] += (
                                batch.y != class_info['edge_class']
                            ).sum()
                            edge_counts[1] += (
                                batch.y == class_info['edge_class']
                            ).sum()

                            pbar.update(1)

                data_stds = stat_var.std()
                data_means = stat_q.median()

                var_data_cache.unlink()
                q_data_cache.unlink()
                tmp_cache_path.rmdir()

            else:

                def get_info(
                    x: torch.Tensor, y: torch.Tensor
                ) -> T.Tuple[torch.Tensor, int, torch.Tensor, torch.Tensor]:
                    crop_counts = torch.zeros(class_info['max_crop_class'] + 1)
                    edge_counts = torch.zeros(2)
                    crop_counts[0] = (
                        (y == 0) | (y == class_info['edge_class'])
                    ).sum()
                    for i in range(1, class_info['edge_class']):
                        crop_counts[i] = (y == i).sum()
                    edge_counts[0] = (y != class_info['edge_class']).sum()
                    edge_counts[1] = (y == class_info['edge_class']).sum()

                    return x.sum(dim=0), x.shape[0], crop_counts, edge_counts

                with parallel_backend(
                    backend='loky',
                    n_jobs=processes,
                    inner_max_num_threads=threads_per_worker,
                ):
                    with TqdmParallel(
                        tqdm_kwargs={
                            'total': int(len(dataset) / batch_size),
                            'desc': 'Calculating means',
                            'colour': mean_color,
                        }
                    ) as pool:
                        results = pool(
                            delayed(get_info)(batch.x, batch.y)
                            for batch in data_loader
                        )
                data_sums, pix_count, crop_counts, edge_counts = list(
                    map(list, zip(*results))
                )

                data_sums = torch.stack(data_sums).sum(dim=0)
                pix_count = torch.tensor(pix_count).sum()
                crop_counts = torch.stack(crop_counts).sum(dim=0)
                edge_counts = torch.stack(edge_counts).sum(dim=0)
                data_means = data_sums / float(pix_count)

                def get_sse(
                    x_mu: torch.Tensor, x: torch.Tensor
                ) -> torch.Tensor:
                    return ((x - x_mu).pow(2)).sum(dim=0)

                sse_partial = partial(get_sse, add_dim(data_means))

                with parallel_backend(
                    backend='loky',
                    n_jobs=processes,
                    inner_max_num_threads=threads_per_worker,
                ):
                    with TqdmParallel(
                        tqdm_kwargs={
                            'total': int(len(dataset) / batch_size),
                            'desc': 'Calculating SSEs',
                            'colour': sse_color,
                        }
                    ) as pool:
                        sses = pool(
                            delayed(sse_partial)(batch.x)
                            for batch in data_loader
                        )

                sses = torch.stack(sses).sum(dim=0)
                data_stds = torch.sqrt(sses / float(pix_count))
                data_maxs = torch.zeros_like(data_means)

        return cls(
            dataset_mean=rearrange(data_means, 'c -> 1 c 1 1 1'),
            dataset_std=rearrange(data_stds, 'c -> 1 c 1 1 1'),
            dataset_crop_counts=crop_counts,
            dataset_edge_counts=edge_counts,
            num_channels=len(data_means),
        )
