import typing as T
from pathlib import Path

from geowombat.core.util import sort_images_by_date

import pandas as pd
import attr
from tqdm.auto import tqdm
from joblib import Parallel


class TqdmParallel(Parallel):
    """A tqdm progress bar for joblib Parallel tasks

    Reference:
        https://stackoverflow.com/questions/37804279/how-can-we-use-tqdm-in-a-parallel-execution-with-joblib
    """
    def __init__(self, tqdm_kwargs: dict):
        self.tqdm_kwargs = tqdm_kwargs
        super().__init__()

    def __call__(self, *args, **kwargs):
        with tqdm(**self.tqdm_kwargs) as self._pbar:
            return Parallel.__call__(self, *args, **kwargs)

    def print_progress(self):
        self._pbar.n = self.n_completed_tasks
        self._pbar.refresh()


@attr.s
class VegetationIndices(object):
    image_vis: T.List[str] = attr.ib(default=None, validator=attr.validators.instance_of(list))

    @property
    def n_vis(self):
        return len(self.image_vis)


@attr.s
class TrainInputs(object):
    regions: T.List[str] = attr.ib(default=None, validator=attr.validators.instance_of(list))
    years: T.List[int] = attr.ib(default=None, validator=attr.validators.instance_of(list))
    lc_path: T.Optional[str] = attr.ib(
        default=None, validator=attr.validators.optional(attr.validators.instance_of(str))
    )

    def __attrs_post_init__(self):
        region_list = self.regions
        self.regions_lists: T.List[T.List[str]] = [region_list]
        self.year_lists: T.List[T.List[int]] = [self.years]
        self.lc_paths_lists: T.List[str] = [self.lc_path]


def get_time_series_list(
    feature_path: Path,
    start_year: int,
    start_date: str,
    end_date: str
) -> T.List[str]:
    """Gets a list of time series paths
    """
    # Get the requested time slice
    image_dict = sort_images_by_date(
        feature_path,
        '*.tif',
        date_pos=0,
        date_start=0,
        date_end=7,
        date_format='%Y%j'
    )
    # Create a DataFrame with paths and dates
    df = pd.DataFrame(
        data=list(image_dict.keys()),
        columns=['name'],
        index=list(image_dict.values())
    )
    # Slice the requested time series from the dataFrame
    ts_list = df.loc[
        f'{start_year}-{start_date}':f'{start_year+1}-{end_date}'
    ].name.values.tolist()

    return ts_list
