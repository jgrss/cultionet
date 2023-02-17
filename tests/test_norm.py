from pathlib import Path

from cultionet.data.datasets import EdgeDataset
from cultionet.data.modules import EdgeDataModule
from cultionet.utils.stats import (
    tally_stats,
    cache_load_enabled,
    load_cached_state,
    Mean,
    Quantile,
    Variance
)

import torch
from torch_geometric.data import Data


PROJECT_PATH = Path(__file__).parent.absolute()


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


def test_norm():
    train_path = PROJECT_PATH / 'data' / 'train' / 'small_chips'
    mean_data_cache = PROJECT_PATH / 'data' / 'train' / 'small_chips' / 'data_means.npz'
    var_data_cache = PROJECT_PATH / 'data' / 'train' / 'small_chips' / 'data_vars.npz'
    var_median_data_cache = PROJECT_PATH / 'data' / 'train' / 'small_chips' / 'data_vars_median.npz'
    q_data_cache = PROJECT_PATH / 'data' / 'train' / 'small_chips' / 'data_quantiles.npz'
    ref_q_data_cache = PROJECT_PATH / 'data' / 'train' / 'small_chips' / 'ref_data_quantiles.npz'
    ref_var_data_cache = PROJECT_PATH / 'data' / 'train' / 'small_chips' / 'ref_data_vars.npz'
    ref_var_median_data_cache = PROJECT_PATH / 'data' / 'train' / 'small_chips' / 'ref_data_vars_median.npz'

    ds = EdgeDataset(
        train_path,
        processes=1,
        threads_per_worker=1,
        random_seed=100
    )
    data_module = EdgeDataModule(
        train_ds=ds,
        batch_size=1,
        num_workers=0,
        shuffle=False
    )

    ref_data = []
    stat_mean = Mean()
    stat_var = Variance()
    stat_var_median = Variance(method='median')
    stat_q = Quantile()
    with cache_load_enabled(False):
        for batch in tally_stats(
            stats=(stat_mean, stat_var, stat_var_median, stat_q),
            loader=data_module.train_dataloader(),
            caches=(mean_data_cache, var_data_cache, var_median_data_cache, q_data_cache)
        ):
            ref_data.append(batch.x)
            stat_mean.add(batch.x)
            stat_q.add(batch.x)
            stat_var.add(batch.x)
            stat_var_median.add(batch.x)
    ref_data = torch.cat(ref_data, dim=0)
    mean = stat_mean.mean()
    std = stat_var.std()
    std_median = stat_var_median.std()
    median = stat_q.median()

    ref_stat_var = Variance()
    cached_state = load_cached_state(ref_var_data_cache)
    ref_stat_var.load_state_dict(cached_state)
    ref_std = ref_stat_var.std()

    ref_stat_var_median = Variance(method='median')
    cached_state = load_cached_state(ref_var_median_data_cache)
    ref_stat_var_median.load_state_dict(cached_state)
    ref_std_median = ref_stat_var_median.std()

    ref_stat_q = Quantile()
    cached_state = load_cached_state(ref_q_data_cache)
    ref_stat_q.load_state_dict(cached_state)
    ref_median = ref_stat_q.median()

    assert torch.allclose(ref_data.mean(dim=0), mean, rtol=1e-4), \
        'The data means do not match the expected values.'
    assert torch.allclose(std, ref_std, rtol=1e-4), \
        'The data standard deviations do not match the cached values.'
    assert torch.allclose(std_median, ref_std_median, rtol=1e-4), \
        'The data median standard deviations do not match the cached values.'
    assert torch.allclose(median, ref_median, rtol=1e-4), \
        'The data medians do not match the cached values.'
