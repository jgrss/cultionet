#!/usr/bin/env python

import argparse
import logging
from pathlib import Path

from cultionet.utils.project_paths import setup_paths

import geowombat as gw
import numpy as np
import geopandas as gpd
from shapely.geometry import Polygon

logger = logging.getLogger(__name__)


class TemporalStats(gw.TimeModule):
    def __init__(self):
        super(TemporalStats, self).__init__()
        self.count = 3
        self.dtype = 'uint16'

    @staticmethod
    def nan_to_num(array) -> np.ndarray:
        return np.nan_to_num(array, nan=0.0, posinf=0.0, neginf=0.0)

    def calculate(self, array):
        array_mean = self.nan_to_num(array.mean(axis=0).squeeze())
        array_max = self.nan_to_num(array.max(axis=0).squeeze())
        array_cv = (self.nan_to_num(array.std(axis=0).squeeze() / array_mean) * 10000.0).clip(0, 10000)

        return np.stack((array_mean, array_max, array_cv))


def create_train_files(args):
    ppaths = setup_paths(args.project_path)

    region_ids = args.regions.split('-')
    region_ids = list(map(int, region_ids))
    if len(region_ids) > 1:
        region_ids = list(range(region_ids[0], region_ids[1]+1))

    for region in region_ids:
        for var in args.image_vars:
            image_path = Path(args.project_path) / f'{region:06d}' / 'brdf_ts' / 'ms' / var
            image_list = list(image_path.glob('*.tif'))
            image_year = int(image_list[0].name[:4]) + 1

            # File names
            grids = ppaths.edge_training_path / f'{region:06d}_grid_{image_year}.gpkg'
            edges = ppaths.edge_training_path / f'{region:06d}_poly_{image_year}.gpkg'
            view_bands_path = ppaths.edge_training_path.parent / 'view_images'
            view_bands_path.mkdir(parents=True, exist_ok=True)
            view_bands = view_bands_path / f'{region:06d}_view_{image_year}.tif'

            if not view_bands.is_file():
                with gw.series(image_list, transfer_lib='numpy') as src:
                    src.apply(
                        TemporalStats(),
                        bands=-1,
                        processes=False,
                        num_workers=4,
                        outfile=str(view_bands)
                    )

            if not grids.is_file():
                with gw.open(image_list[0], chunks=512) as src:
                    grid_df = src.gw.geodataframe
                grid_df['grid'] = 0
                left, bottom, right, top = grid_df.total_bounds.tolist()
                geom = Polygon([(left, top),
                                (left, top),
                                (left, top),
                                (left, top),
                                (left, top)])
                edge_df = gpd.GeoDataFrame(
                    data=[0], columns=['class'], geometry=[geom], crs=grid_df.crs
                )
                edge_df.to_file(edges, driver='GPKG')
                grid_df.to_file(grids, driver='GPKG')


def main():
    parser = argparse.ArgumentParser(description='Creates edge and grid files for training',
                                     formatter_class=argparse.RawTextHelpFormatter)

    parser.add_argument(
        '-v', '--image-vars', dest='image_vars', help='The time series variables', default=None, nargs='+'
    )
    parser.add_argument('-p', '--project-path', dest='project_path', help='The NNet project path', default=None)
    parser.add_argument('--regions', dest='regions', help='The region ids (e.g., 1-10)', default=None)

    args = parser.parse_args()

    create_train_files(args)


if __name__ == '__main__':
    main()
