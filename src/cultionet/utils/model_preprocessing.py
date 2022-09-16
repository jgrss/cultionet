import typing as T
from pathlib import Path

from geowombat.core.util import sort_images_by_date

import pandas as pd
import attr


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


def get_time_series_list(vi_path: Path, image_year: int, start_date: str, end_date: str) -> T.List[str]:
    """Gets a list of time series paths
    """
    # Get the requested time slice
    image_dict = sort_images_by_date(
        vi_path, "*.tif", date_pos=0, date_start=0, date_end=7, date_format='%Y%j'
    )
    # Create a DataFrame with paths and dates
    df = pd.DataFrame(data=list(image_dict.keys()), columns=['name'], index=list(image_dict.values()))
    # Slice the requested time series from the dataFrame
    ts_list = df.loc[f'{image_year-1}-{start_date}':f'{image_year}-{end_date}'].name.values.tolist()

    return ts_list
