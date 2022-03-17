#!/usr/bin/env python

import argparse
import typing as T
import logging
from pathlib import Path
from datetime import datetime

import cultionet
from cultionet.data.utils import create_network_data, NetworkDataset
from cultionet.utils import model_preprocessing
from cultionet.utils.project_paths import setup_paths

import geowombat as gw
from geowombat.core.windows import get_window_offsets
import numpy as np
import rasterio as rio
from rasterio.windows import Window
import geopandas as gpd
import torch
import yaml
from tqdm.auto import tqdm


logger = logging.getLogger(__name__)
logging.getLogger('pytorch_lightning').setLevel(logging.WARNING)


with open('config.yml', 'r') as pf:
    CONFIG = yaml.load(pf, Loader=yaml.FullLoader)


def get_centroid_coords(vi_path: Path, dst_crs: T.Optional[str] = None) -> T.Tuple[float, float]:
    """Gets the lon/lat or x/y coordinates of a centroid
    """
    with gw.open(list(vi_path.glob('*.tif'))[0]) as src:
        df = src.gw.geodataframe
    centroid = df.to_crs(dst_crs).centroid

    return float(centroid.x), float(centroid.y)


def get_image_list(ppaths, region):
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
        lon, lat = get_centroid_coords(vi_path, dst_crs='epsg:4326')
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
    image_list = get_image_list(ppaths, region)

    with gw.open(
            image_list,
            stack_dim='band',
            band_names=list(range(1, len(image_list)+1))
    ) as src_ts:
        time_series = ((src_ts * args.gain + args.offset)
                       .astype('float64')
                       .clip(0, 1))
        # Get the image dimensions
        nvars = model_preprocessing.VegetationIndices(image_vis=CONFIG['image_vis']).n_vis
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


def main():
    parser = argparse.ArgumentParser(description='Applies inference to an image',
                                     formatter_class=argparse.RawTextHelpFormatter,
                                     epilog="########\n"
                                            "Examples\n"
                                            "########\n\n"
                                            "python predict_image.py --project-path /projects/data -o file.tif \n\n")

    parser.add_argument('-p', '--project-path', dest='project_path', help='The project path', default=None)
    parser.add_argument('-o', '--out-path', dest='out_path', help='The output path', default=None)
    parser.add_argument('-g', '--grid-id', dest='grid_id', help='The grid id to process', default=None)
    parser.add_argument(
        '-w', '--window-size', dest='window_size', help='The window size (default: %(default)s)',
        default=256, type=int
    )
    parser.add_argument(
        '--padding', dest='padding', help='The window size (default: %(default)s)',
        default=5, type=int
    )
    parser.add_argument(
        '--gain', dest='gain', help='The image gain (default: %(default)s)', default=0.0001, type=float
    )
    parser.add_argument(
        '--offset', dest='offset', help='The image offset (default: %(default)s)', default=0.0, type=float
    )
    parser.add_argument(
        '--filters', dest='filters', help='The number of base filters (default: %(default)s)', default=32, type=int
    )
    parser.add_argument(
        '--device', dest='device', help='The device to train on (default: %(default)s)',
        default='gpu', choices=['cpu', 'gpu']
    )
    parser.add_argument(
        '--append-ts',
        dest='append_ts',
        help='Whether to append time_series_vars to the image path (default: %(default)s)',
        default='y',
        choices=['y', 'n']
    )

    args = parser.parse_args()

    predict_image(args)


if __name__ == '__main__':
    main()
