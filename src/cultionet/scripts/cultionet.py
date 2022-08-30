#!/usr/bin/env python

import argparse
import typing as T
import logging
from pathlib import Path
from datetime import datetime

import cultionet
from cultionet.data.datasets import EdgeDataset
from cultionet.utils.project_paths import setup_paths
from cultionet.utils.normalize import get_norm_values
from cultionet.data.create import create_dataset
from cultionet.utils import model_preprocessing
from cultionet.data.utils import create_network_data, NetworkDataset

import torch
import geopandas as gpd
import pandas as pd
import yaml


logger = logging.getLogger(__name__)

geo_id_data = pd.read_csv('~/geo_id_grid_list.csv')
geo_id_list = geo_id_data['geo_id_grid'].unique().tolist() 

DEFAULT_AUGMENTATIONS = ['none', 'fliplr', 'flipud', 'flipfb',
                         'rot90', 'rot180', 'rot270',
                         'ts-warp', 'ts-noise', 'ts-drift']


def open_config(config_file: T.Union[str, Path, bytes]) -> dict:
    with open(config_file, 'r') as pf:
        config = yaml.load(pf, Loader=yaml.FullLoader)

    return config


def get_centroid_coords_from_image(vi_path: Path, dst_crs: T.Optional[str] = None) -> T.Tuple[float, float]:
    """Gets the lon/lat or x/y coordinates of a centroid
    """
    import geowombat as gw

    with gw.open(list(vi_path.glob('*.tif'))[0]) as src:
        df = src.gw.geodataframe
    centroid = df.to_crs(dst_crs).centroid

    return float(centroid.x), float(centroid.y)


def get_image_list(ppaths, region, config):
    image_list = []
    for image_vi in model_preprocessing.VegetationIndices(image_vis=config['image_vis']).image_vis:
        # Set the full path to the images
        if str(ppaths.image_path).endswith('time_series_vars'):
            vi_path = ppaths.image_path / region / image_vi
        else:
            vi_path = ppaths.image_path / region / 'brdf_ts' / 'ms' / image_vi

        if not vi_path.is_dir():
            logger.warning(f'{str(vi_path)} does not exist')
            continue

        # Get the centroid coordinates of the grid
        lon, lat = get_centroid_coords_from_image(vi_path, dst_crs='epsg:4326')
        # Get the image year
        file = list(vi_path.glob('*.tif'))[0]
        image_year = int(datetime.strptime(file.stem, '%Y%j').strftime('%Y')) + 1

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

    return image_list


def predict_image(args):
    import geowombat as gw
    from geowombat.core.windows import get_window_offsets
    import numpy as np
    import rasterio as rio
    from rasterio.windows import Window
    import torch
    import yaml
    from tqdm.auto import tqdm

    logging.getLogger('pytorch_lightning').setLevel(logging.WARNING)

    config = open_config(args.config_file)

    # This is a helper function to manage paths
    ppaths = setup_paths(args.project_path, append_ts=True if args.append_ts == 'y' else False)

    # Load the z-score norm values
    data_values = torch.load(ppaths.norm_file)

    try:
        tmp = int(args.grid_id)
        region = f'{tmp:06d}'
    except:
        region = args.grid_id

    # Get the image list
    image_list = get_image_list(ppaths, region, config)

    with gw.open(
            image_list,
            stack_dim='band',
            band_names=list(range(1, len(image_list)+1))
    ) as src_ts:
        time_series = ((src_ts * args.gain + args.offset)
                       .astype('float64')
                       .clip(0, 1))
        # Get the image dimensions
        nvars = model_preprocessing.VegetationIndices(image_vis=config['image_vis']).n_vis
        nfeas, height, width = time_series.shape
        ntime = int(nfeas / nvars)
        # TODO: chunk size and padding
        windows = get_window_offsets(
            height,
            width,
            args.window_size,
            args.window_size,
            padding=(
                args.padding, args.padding, args.padding, args.padding
            )
        )

        profile = {
            'crs': src_ts.crs,
            'transform': src_ts.gw.transform,
            'height': height,
            'width': width,
            'count': 4,
            'dtype': 'uint16',
            'blockxsize': 64 if 64 < width else width,
            'blockysize': 64 if 64 < height else height,
            'driver': 'GTiff'
        }

        # Create the output file
        with rio.open(args.out_path, mode='w', **profile) as dst:
            pass

        for w, w_pad in tqdm(windows, total=len(windows)):
            slc = (
                slice(0, None),
                slice(w_pad.row_off, w_pad.row_off+w_pad.height),
                slice(w_pad.col_off, w_pad.col_off+w_pad.width)
            )
            # Create the data for the chunk
            data = create_network_data(time_series[slc].data.compute(num_workers=8), ntime)
            # Create the temporary dataset
            net_ds = NetworkDataset(data, ppaths.predict_path, data_values)

            # Apply inference on the chunk
            stack, lit_model = cultionet.predict(
                predict_ds=net_ds.ds,
                ckpt_file=ppaths.ckpt_file,
                filters=args.filters,
                device=args.device,
                w=w,
                w_pad=w_pad
            )
            # Write the prediction stack to file
            with rio.open(args.out_path, mode='r+') as dst:
                dst.write(
                    (stack*10000.0).clip(0, 10000).astype('uint16'),
                    indexes=[1, 2, 3, 4],
                    window=w
                )

            # Remove the temporary dataset
            net_ds.unlink()


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
    config = open_config(args.config_file)
    project_path_lists = [args.project_path]
    ref_res_lists = [args.ref_res]

    inputs = model_preprocessing.TrainInputs(
        regions=geo_id_list,
        years=config['years'],
        lc_path=config['lc_path']
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
        for image_vi in model_preprocessing.VegetationIndices(image_vis=config['image_vis']).image_vis:
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

            # TODO: allow user to specify start/end dates
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
                grid_size=args.grid_size,
                lc_path=lc_image,
                n_ts=args.n_ts,
                data_type='boundaries'
            )


def train_model(args):
    # This is a helper function to manage paths
    ppaths = setup_paths(args.project_path)

    # Check dimensions
    ds = EdgeDataset(ppaths.train_path)
    ds.check_dims()
    # Get the normalization means and std. deviations on the train data
    cultionet.model.seed_everything(args.random_seed)
    train_ds, val_ds = ds.split_train_val(val_frac=args.val_frac)
    # Calculate the values needed to transform to z-scores, using
    # the training data
    data_values = get_norm_values(dataset=train_ds, batch_size=args.batch_size*4)
    torch.save(data_values, str(ppaths.norm_file))

    # Create the train data object again, this time passing
    # the means and standard deviation tensors
    ds = EdgeDataset(
        ppaths.train_path,
        data_means=data_values.mean,
        data_stds=data_values.std
    )

    # Fit the model
    cultionet.fit(
        dataset=ds,
        ckpt_file=ppaths.ckpt_file,
        val_frac=args.val_frac,
        batch_size=args.batch_size,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        filters=args.filters,
        random_seed=args.random_seed,
        reset_model=args.reset_model,
        auto_lr_find=args.auto_lr_find,
        device=args.device,
        gradient_clip_val=args.gradient_clip_val,
        early_stopping_patience=args.patience,
        stochastic_weight_avg=args.stochastic_weight_avg
    )


def main():
    parser = argparse.ArgumentParser(
        description='Cultionet models',
        formatter_class=argparse.RawTextHelpFormatter,
        epilog="########\n"
               "Examples\n"
               "########\n\n"
               "# Create training data\n"
               "cultionet create --project-path /projects/data \n\n"
               "# Train a model\n"
               "cultionet train --project-path /projects/data \n\n"
               "# Apply inference over an image\n"
               "cultionet predict --project-path /projects/data -o estimates.tif \n\n"
    )

    subparsers = parser.add_subparsers(dest='process')
    available_processes = ['create', 'train', 'predict', 'version']
    for process in available_processes:
        subparser = subparsers.add_parser(process)

        if process == 'version':
            continue

        subparser.add_argument('-p', '--project-path', dest='project_path', help='The project path', default=None)

        if process == 'create':
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
                '-r', '--res', dest='ref_res', help='The cell resolution (default: %(default)s)', default=10.0,
                type=float
            )
            subparser.add_argument(
                '-rm', '--resampling', dest='resampling', help='The resampling method (default: %(default)s)',
                default='nearest'
            )
            subparser.add_argument(
                '-gs', '--grid-size', dest='grid_size',
                help='The grid size (*If not given, grid size is taken from the the grid vector. If given, grid size '
                     'is taken from the upper left coordinate of the grid vector.) (default: %(default)s)',
                default=None, nargs='+', type=int
            )
        elif process == 'train':
            subparser.add_argument(
                '--val-frac', dest='val_frac', help='The validation fraction (default: %(default)s)',
                default=0.2, type=float
            )
            subparser.add_argument(
                '--random-seed', dest='random_seed', help='The random seed (default: %(default)s)',
                default=42, type=int
            )
            subparser.add_argument(
                '--batch-size', dest='batch_size', help='The batch size (default: %(default)s)',
                default=4, type=int
            )
            subparser.add_argument(
                '--epochs', dest='epochs', help='The number of training epochs (default: %(default)s)',
                default=30, type=int
            )
            subparser.add_argument(
                '--learning-rate', dest='learning_rate', help='The learning rate (default: %(default)s)',
                default=0.001, type=float
            )
            subparser.add_argument(
                '--reset-model', dest='reset_model', help='Whether to reset the model (default: %(default)s)',
                action='store_true'
            )
            subparser.add_argument(
                '--lr-find', dest='auto_lr_find', help='Whether to tune the learning rate (default: %(default)s)',
                action='store_true'
            )
            subparser.add_argument(
                '--gradient-clip-val', dest='gradient_clip_val', help='The gradient clip value (default: %(default)s)',
                default=0.1, type=float
            )
            subparser.add_argument(
                '--patience', dest='patience', help='The early stopping patience (default: %(default)s)',
                default=7, type=int
            )
            subparser.add_argument(
                '--apply-swa', dest='stochastic_weight_avg',
                help='Whether to apply stochastic weight averaging (default: %(default)s)',
                action='store_true'
            )
        elif process == 'predict':
            subparser.add_argument('-o', '--out-path', dest='out_path', help='The output path', default=None)
            subparser.add_argument('-g', '--grid-id', dest='grid_id', help='The grid id to process', default=None)
            subparser.add_argument(
                '-w', '--window-size', dest='window_size', help='The window size (default: %(default)s)',
                default=256, type=int
            )
            subparser.add_argument(
                '--padding', dest='padding', help='The window size (default: %(default)s)',
                default=5, type=int
            )
            subparser.add_argument(
                '--gain', dest='gain', help='The image gain (default: %(default)s)', default=0.0001, type=float
            )
            subparser.add_argument(
                '--offset', dest='offset', help='The image offset (default: %(default)s)', default=0.0, type=float
            )
        if process in ['create', 'predict']:
            subparser.add_argument(
                '--append-ts',
                dest='append_ts',
                help='Whether to append time_series_vars to the image path (default: %(default)s)',
                default='y',
                choices=['y', 'n']
            )
            subparser.add_argument(
                '--config-file',
                dest='config_file',
                help='The configuration YAML file (default: %(default)s)',
                default=(Path('.') / 'config.yml').resolve()
            )
        if process in ['train', 'predict']:
            subparser.add_argument(
                '--filters', dest='filters', help='The number of base filters (default: %(default)s)', default=32,
                type=int
            )
            subparser.add_argument(
                '--device', dest='device', help='The device to train on (default: %(default)s)',
                default='gpu', choices=['cpu', 'gpu']
            )

    args = parser.parse_args()

    if args.process == 'version':
        print(cultionet.__version__)
        return

    if args.process == 'create':
        persist_dataset(args)
    elif args.process == 'train':
        train_model(args)
    elif args.process == 'predict':
        predict_image(args)


if __name__ == '__main__':
    main()
