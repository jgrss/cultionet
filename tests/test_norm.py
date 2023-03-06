from pathlib import Path

from cultionet.data.datasets import zscores, EdgeDataset
from cultionet.utils.normalize import get_norm_values
# from cultionet.data.modules import EdgeDataModule
# from cultionet.utils.stats import (
#     tally_stats,
#     cache_load_enabled,
#     load_cached_state,
#     Mean,
#     Quantile,
#     Variance
# )

import torch
from torch_geometric.data import Data
import pytest


PROJECT_PATH = Path(__file__).parent.absolute()
CLASS_INFO = {
    'max_crop_class': 1,
    'edge_class': 2
}


def create_small_chips(b: torch.Tensor, rc_slice: tuple) -> Data:
    """Method used to create new data

    Example:
        >>> import joblib
        >>>
        >>> batch = joblib.load('...')
        >>> create_small_chips(
        >>>     batch,
        >>>     rc_slice=(slice(0, None), slice(45, 55), slice(45, 55))
        >>> )
        >>>
        >>> # Create small data chips in the test dir
        >>> out_path = Path('test_dir')
        >>> for fn in Path('train/processed').glob('*.pt'):
        >>>     batch = joblib.load(fn)
        >>>     small_batch = create_create_small_chipstest_data(
        >>>         batch,
        >>>         (slice(0, None), slice(45, 55), slice(45, 55))
        >>>     )
        >>>     joblib.dump(small_batch, out_path / fn.name)
    """
    exclusion = ('x', 'height', 'width')
    # Reshape to (C x H x W)
    x = b.x.t().reshape(b.ntime*b.nbands, b.height, b.width)
    # Take a subset
    x = x[rc_slice]
    # Reshape back to (S x D)
    height = rc_slice[1].stop - rc_slice[1].start
    width = rc_slice[2].stop - rc_slice[2].start
    x = x.permute(1, 2, 0).reshape(height*width, b.ntime*b.nbands)

    return Data(
        x=x,
        height=height,
        width=width,
        **{k: getattr(b, k) for k in b.keys if k not in exclusion}
    )


@pytest.fixture(scope='session')
def train_dataset() -> EdgeDataset:
    train_path = PROJECT_PATH / 'data' / 'train' / 'small_chips'

    ds = EdgeDataset(
        train_path,
        processes=1,
        threads_per_worker=1,
        random_seed=100
    )

    return ds


@pytest.fixture(scope='session')
def serial_ref_data(train_dataset: EdgeDataset) -> torch.Tensor:
    ref_data = torch.cat([batch.x for batch in train_dataset], dim=0)

    return ref_data


@pytest.fixture(scope='session')
def serial_norm_data(train_dataset: EdgeDataset) -> Data:
    norm_values = get_norm_values(
        dataset=train_dataset,
        batch_size=1,
        class_info=CLASS_INFO,
        num_workers=1,
        processes=1,
        threads_per_worker=1,
        mean_color='#3edf2b',
        sse_color='#dfb92b'
    )

    return norm_values


def test_cumnorm_serial(
    serial_ref_data: torch.Tensor,
    serial_norm_data: Data
):
    assert torch.allclose(serial_norm_data.mean, serial_ref_data.mean(dim=0), rtol=1e-4), \
        'The mean values do not match the expected values.'
    assert torch.allclose(serial_norm_data.std, serial_ref_data.std(dim=0, unbiased=False), rtol=1e-4), \
        'The mean values do not match the expected values.'


def test_cumnorm_concurrent(train_dataset: EdgeDataset, serial_ref_data: torch.Tensor):
    norm_values = get_norm_values(
        dataset=train_dataset,
        batch_size=1,
        class_info=CLASS_INFO,
        num_workers=1,
        processes=4,
        threads_per_worker=2,
        mean_color='#df4a2b',
        sse_color='#2ba0df'
    )

    assert torch.allclose(norm_values.mean, serial_ref_data.mean(dim=0), rtol=1e-4), \
        'The mean values do not match the expected values.'
    assert torch.allclose(norm_values.std, serial_ref_data.std(dim=0, unbiased=False), rtol=1e-4), \
        'The mean values do not match the expected values.'


def test_transform_data(train_dataset: EdgeDataset, serial_norm_data: Data):
    ref_batch = train_dataset[0]
    batch = zscores(
        batch=ref_batch,
        data_means=serial_norm_data.mean,
        data_stds=serial_norm_data.std,
    )

    # z = (x - μ) / σ
    ref_zscores = (ref_batch.x - serial_norm_data.mean) / serial_norm_data.std

    assert torch.allclose(batch.x, ref_zscores), 'The z-scores do not match the expected values.'


# NOTE: this module is not currently used, but we will
# keep the test here in case of future use
# def test_norm():
#     train_path = PROJECT_PATH / 'data' / 'train' / 'small_chips'
#     mean_data_cache = PROJECT_PATH / 'data' / 'train' / 'small_chips' / 'data_means.npz'
#     var_data_cache = PROJECT_PATH / 'data' / 'train' / 'small_chips' / 'data_vars.npz'
#     var_median_data_cache = PROJECT_PATH / 'data' / 'train' / 'small_chips' / 'data_vars_median.npz'
#     q_data_cache = PROJECT_PATH / 'data' / 'train' / 'small_chips' / 'data_quantiles.npz'
#     ref_q_data_cache = PROJECT_PATH / 'data' / 'train' / 'small_chips' / 'ref_data_quantiles.npz'
#     ref_var_data_cache = PROJECT_PATH / 'data' / 'train' / 'small_chips' / 'ref_data_vars.npz'
#     ref_var_median_data_cache = PROJECT_PATH / 'data' / 'train' / 'small_chips' / 'ref_data_vars_median.npz'

#     ds = EdgeDataset(
#         train_path,
#         processes=1,
#         threads_per_worker=1,
#         random_seed=100
#     )
#     # TODO: test this
#     # norm_values = get_norm_values(
#     #     dataset=ds,
#     #     batch_size=1,
#     #     class_info=CLASS_INFO,
#     #     num_workers=4,
#     #     centering='median'
#     # )

#     data_module = EdgeDataModule(
#         train_ds=ds,
#         batch_size=1,
#         num_workers=0,
#         shuffle=False
#     )

#     ref_data = []
#     stat_mean = Mean()
#     stat_var = Variance()
#     stat_var_median = Variance(method='median')
#     stat_q = Quantile()
#     with cache_load_enabled(False):
#         for batch in tally_stats(
#             stats=(stat_mean, stat_var, stat_var_median, stat_q),
#             loader=data_module.train_dataloader(),
#             caches=(mean_data_cache, var_data_cache, var_median_data_cache, q_data_cache)
#         ):
#             ref_data.append(batch.x)
#             stat_mean.add(batch.x)
#             stat_q.add(batch.x)
#             stat_var.add(batch.x)
#             stat_var_median.add(batch.x)
#     ref_data = torch.cat(ref_data, dim=0)
#     mean = stat_mean.mean()
#     std = stat_var.std()
#     std_median = stat_var_median.std()
#     median = stat_q.median()

#     ref_stat_var = Variance()
#     cached_state = load_cached_state(ref_var_data_cache)
#     ref_stat_var.load_state_dict(cached_state)
#     ref_std = ref_stat_var.std()

#     ref_stat_var_median = Variance(method='median')
#     cached_state = load_cached_state(ref_var_median_data_cache)
#     ref_stat_var_median.load_state_dict(cached_state)
#     ref_std_median = ref_stat_var_median.std()

#     ref_stat_q = Quantile()
#     cached_state = load_cached_state(ref_q_data_cache)
#     ref_stat_q.load_state_dict(cached_state)
#     ref_median = ref_stat_q.median()

#     assert torch.allclose(ref_data.mean(dim=0), mean, rtol=1e-4), \
#         'The data means do not match the expected values.'
#     assert torch.allclose(std, ref_std, rtol=1e-4), \
#         'The data standard deviations do not match the cached values.'
#     assert torch.allclose(std_median, ref_std_median, rtol=1e-4), \
#         'The data median standard deviations do not match the cached values.'
#     assert torch.allclose(median, ref_median, rtol=1e-4), \
#         'The data medians do not match the cached values.'
