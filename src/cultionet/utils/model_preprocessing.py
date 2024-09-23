import typing as T
from pathlib import Path

import attr
import pandas as pd
from geowombat.core.util import sort_images_by_date
from joblib import Parallel
from tqdm.auto import tqdm


class ParallelProgress(Parallel):
    """A tqdm progress bar for joblib Parallel tasks.

    Reference:
        https://stackoverflow.com/questions/37804279/how-can-we-use-tqdm-in-a-parallel-execution-with-joblib
    """

    def __init__(self, tqdm_kwargs: dict, **joblib_kwargs):
        self.tqdm_kwargs = tqdm_kwargs
        super().__init__(**joblib_kwargs)

    def __call__(self, *args, **kwargs):
        with tqdm(**self.tqdm_kwargs) as self._pbar:
            return Parallel.__call__(self, *args, **kwargs)

    def print_progress(self):
        self._pbar.n = self.n_completed_tasks
        self._pbar.refresh()


@attr.s
class VegetationIndices(object):
    image_vis: T.List[str] = attr.ib(
        default=None, validator=attr.validators.instance_of(list)
    )

    @property
    def n_vis(self):
        return len(self.image_vis)


@attr.s
class TrainInputs(object):
    regions: T.List[str] = attr.ib(
        default=None, validator=attr.validators.instance_of(list)
    )
    years: T.List[int] = attr.ib(
        default=None, validator=attr.validators.instance_of(list)
    )

    def __attrs_post_init__(self):
        region_list = self.regions
        self.regions_lists: T.List[T.List[str]] = [region_list]
        self.year_lists: T.List[T.List[int]] = [self.years]


def get_time_series_list(
    feature_path: Path,
    date_format: str = '%Y%j',
    start_date: T.Optional[pd.Timestamp] = None,
    end_date: T.Optional[pd.Timestamp] = None,
    end_year: T.Optional[T.Union[int, str]] = None,
    start_mmdd: T.Optional[str] = None,
    end_mmdd: T.Optional[str] = None,
    num_months: T.Optional[int] = None,
) -> T.List[str]:
    """Gets a list of time series paths."""
    # Get the requested time slice
    image_dict = sort_images_by_date(
        feature_path,
        '*.tif',
        date_pos=0,
        date_start=0,
        date_end=7 if date_format == '%Y%j' else 8,
        date_format=date_format,
    )

    # Create a DataFrame with paths and dates
    df = pd.DataFrame(
        data=list(image_dict.keys()),
        columns=['name'],
        index=list(image_dict.values()),
    )

    if (start_date is not None) and (end_date is not None):
        start_date_stamp = start_date
        end_date_stamp = end_date
    else:
        end_date_stamp = pd.Timestamp(
            f"{end_year}-{end_mmdd}"
        ) + pd.DateOffset(days=1)
        start_year = (end_date_stamp - pd.DateOffset(months=num_months)).year
        start_date_stamp = pd.Timestamp(f"{start_year}-{start_mmdd}")

    image_df = df.loc[start_date_stamp:end_date_stamp]

    if num_months is not None:
        assert (
            num_months <= len(image_df.index) <= num_months + 1
        ), "The image list is not the correct length."

    # Slice the requested time series from the dataFrame
    ts_list = image_df.name.values.tolist()

    return ts_list
