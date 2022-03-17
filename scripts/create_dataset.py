#!/usr/bin/env python

import typing as T
import argparse
import logging
from pathlib import Path

from cultionet.data.create import create_dataset
from cultionet.utils import model_preprocessing
from cultionet.utils.project_paths import setup_paths

import geopandas as gpd
import yaml

logger = logging.getLogger(__name__)

DEFAULT_AUGMENTATIONS = ['none', 'fliplr', 'flipud', 'flipfb',
                         'rot90', 'rot180', 'rot270',
                         'ts-warp', 'ts-noise', 'ts-drift']

with open('config.yml', 'r') as pf:
    CONFIG = yaml.load(pf, Loader=yaml.FullLoader)


def cycle_data(year_lists: list,
               regions_lists: list,
               project_path_lists: list,
               lc_paths_lists: list,
               ref_res_lists: list):
    for years, regions, project_path, lc_path, ref_res in zip(year_lists,
                                                              regions_lists,
                                                              project_path_lists,
                                                              lc_paths_lists,
                                                              ref_res_lists):
        for region in regions:
            for image_year in years:
                yield region, image_year, project_path, lc_path, ref_res


def get_centroid_coords(df: gpd.GeoDataFrame, dst_crs: T.Optional[str] = None) -> T.Tuple[float, float]:
    """Gets the lon/lat or x/y coordinates of a centroid
    """
    centroid = df.to_crs(dst_crs).centroid

    return float(centroid.x), float(centroid.y)


def persist_dataset(args):
    project_path_lists = [args.project_path]
    ref_res_lists = [args.ref_res]

    inputs = model_preprocessing.TrainInputs(
        regions=CONFIG['regions'],
        years=CONFIG['years'],
        lc_path=CONFIG['lc_path']
    )

    for region, image_year, project_path, lc_path, ref_res in cycle_data(
            inputs.year_lists,
            inputs.regions_lists,
            project_path_lists,
            inputs.lc_paths_lists,
            ref_res_lists
    ):
        ppaths = setup_paths(project_path, append_ts=True if args.append_ts == 'y' else False)

        try:
            tmp = int(region)
            region = f'{tmp:06d}'
        except:
            pass

        # Read the training data
        grids = ppaths.edge_training_path / f'{region}_grid_{image_year}.gpkg'
        edges = ppaths.edge_training_path / f'{region}_edges_{image_year}.gpkg'
        if not grids.is_file():
            logger.warning(f'{grids} does not exist.')
            continue
        if not edges.is_file():
            edges = ppaths.edge_training_path / f'{region}_poly_{image_year}.gpkg'
        df_grids = gpd.read_file(grids)
        df_edges = gpd.read_file(edges)

        image_list = []
        for image_vi in model_preprocessing.VegetationIndices(image_vis=CONFIG['image_vis']).image_vis:
            # Set the full path to the images
            if str(ppaths.image_path).endswith('time_series_vars'):
                vi_path = ppaths.image_path / region / image_vi
            else:
                vi_path = ppaths.image_path / region / 'brdf_ts' / 'ms' / image_vi

            if not vi_path.is_dir():
                logger.warning(f'{str(vi_path)} does not exist')
                continue

            # Get the centroid coordinates of the grid
            lon, lat = get_centroid_coords(df_grids.centroid, dst_crs='epsg:4326')

            if lat > 0:
                start_date = '01-01'
                end_date = '01-01'
            else:
                start_date = '07-01'
                end_date = '07-01'

            # Get the requested time slice
            ts_list = model_preprocessing.get_time_series_list(
                vi_path, image_year, start_date, end_date
            )

            if len(ts_list) <= 1:
                continue

            image_list += ts_list

        if image_list:
            if lc_path is None:
                lc_image = None
            else:
                if (Path(lc_path) / f'{image_year-1}_30m_cdls.tif').is_file():
                    lc_image = str(Path(lc_path) / f'{image_year-1}_30m_cdls.tif')
                else:
                    if not (Path(lc_path) / f'{image_year-1}_30m_cdls.img').is_file():
                        continue
                    lc_image = str(Path(lc_path) / f'{image_year-1}_30m_cdls.img')

            create_dataset(
                image_list,
                df_grids,
                df_edges,
                group_id=f'{region}_{image_year}',
                process_path=ppaths.process_path,
                transforms=args.transforms,
                ref_res=ref_res,
                resampling=args.resampling,
                num_workers=args.num_workers,
                lc_path=lc_image,
                n_ts=args.n_ts,
                data_type=args.data_type
            )


def main():

    parser = argparse.ArgumentParser(description='Creates a dataset',
                                     formatter_class=argparse.RawTextHelpFormatter,
                                     epilog="########\n"
                                            "Examples\n"
                                            "########\n\n"
                                            "# Create training data for boundaries\n"
                                            "python create_dataset.py boundaries \ \n"
                                            "    --project-path /projects/data \n\n"
                                            "# Create training data for crop-type\n"
                                            "python create_dataset.py crop-type \ \n"
                                            "    --project-path /projects/data \n\n")

    subparsers = parser.add_subparsers(dest='data_type')

    for step in ['boundaries', 'crop-type']:
        subparser = subparsers.add_parser(step)

        subparser.add_argument('-p', '--project-path', dest='project_path', help='The project path', default=None)
        subparser.add_argument(
            '-n', '--num-workers', dest='num_workers',
            help='The number of CPUs for data creation (default: %(default)s)',
            default=4, type=int
        )
        subparser.add_argument(
            '-t', '--transforms', dest='transforms', help='Augmentation transforms (default: %(default)s)',
            default=DEFAULT_AUGMENTATIONS, choices=DEFAULT_AUGMENTATIONS, nargs='+'
        )
        subparser.add_argument(
            '--n-ts', dest='n_ts', help='The number of temporal augmentations (default: %(default)s)',
            default=6, type=int
        )
        subparser.add_argument(
            '-r', '--res', dest='ref_res', help='The cell resolution (default: %(default)s)', default=10.0, type=float
        )
        subparser.add_argument(
            '-rm', '--resampling', dest='resampling', help='The resampling method (default: %(default)s)',
            default='nearest'
        )
        subparser.add_argument(
            '--append-ts',
            dest='append_ts',
            help='Whether to append time_series_vars to the image path (default: %(default)s)',
            default='y',
            choices=['y', 'n']
        )

    args = parser.parse_args()

    persist_dataset(args)


if __name__ == '__main__':
    main()
