#!/usr/bin/env python

import argparse
import logging
from pathlib import Path

from cultionet.utils.project_paths import setup_paths

import geowombat as gw
import numpy as np
import geopandas as gpd
from shapely.geometry import box
from pyproj import CRS
import pyproj

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

    # File names
    ppaths.edge_training_path.mkdir(exist_ok=True, parents=True)
    grids = ppaths.edge_training_path / f'{args.region:06d}_grid_{args.year}.gpkg'
    mask = ppaths.edge_training_path / f'{args.region:06d}_poly_{args.year}.gpkg'

    # if not view_bands.is_file():
    #     with gw.series(image_list, transfer_lib='numpy') as src:
    #         src.apply(
    #             TemporalStats(),
    #             bands=-1,
    #             processes=False,
    #             num_workers=4,
    #             outfile=str(view_bands)
    #         )

    lat, lon = args.lat_lon
    crs = CRS.from_user_input(args.crs)
    x, y = pyproj.Proj(crs)(lon, lat)

    left = x - args.grid_size
    right = x + args.grid_size
    bottom = y - args.grid_size
    top = y + args.grid_size

    left, top = pyproj.Proj(crs)(left, top, inverse=True)
    right, bottom = pyproj.Proj(crs)(right, bottom, inverse=True)

    geometry = box(left, bottom, right, top)
    grid_df = gpd.GeoDataFrame(
        data=[0],
        columns=['class_value'],
        geometry=[geometry],
        crs='epsg:4326'
    )
    poly_df = grid_df.copy()
    grid_df.to_file(grids, driver='GPKG')
    poly_df.to_file(mask, driver='GPKG')


def main():
    parser = argparse.ArgumentParser(
        description='Creates training vector files',
        formatter_class=argparse.RawTextHelpFormatter
    )

    parser.add_argument(
        '-p', '--project-path', dest='project_path', help='The project path', default=None
    )
    parser.add_argument(
        '--region', dest='region', help='The region id', default=None, type=int
    )
    parser.add_argument(
        '--year', dest='year', help='The year', default=None, type=int
    )
    parser.add_argument(
        '--crs', dest='crs', help='The CRS', default='epsg:8858'
    )
    parser.add_argument(
        '--grid-size', dest='grid_size', help='The grid size (meters)', default=1_000, type=int
    )
    parser.add_argument(
        '--lat-lon', dest='lat_lon', help='The lat,lon coordinates', default=None, type=float, nargs='+'
    )

    args = parser.parse_args()

    create_train_files(args)


if __name__ == '__main__':
    main()
