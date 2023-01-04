#!/usr/bin/env python

from abc import abstractmethod
import argparse
import typing as T
import logging
from pathlib import Path
from datetime import datetime
import asyncio
import filelock
import builtins
import json
import ast

import cultionet
from cultionet.data.const import SCALE_FACTOR
from cultionet.data.datasets import EdgeDataset
from cultionet.utils.project_paths import (
    setup_paths, ProjectPaths
)
from cultionet.errors import TensorShapeError
from cultionet.utils.normalize import get_norm_values
from cultionet.data.create import create_dataset, create_predict_dataset
from cultionet.data.utils import (
    get_image_list_dims, create_network_data
)
from cultionet.utils import model_preprocessing
from cultionet.utils.logging import set_color_logger

import geowombat as gw
from geowombat.core.windows import get_window_offsets
import geopandas as gpd
import pandas as pd
import yaml
import rasterio as rio
from rasterio.windows import Window
import torch
import xarray as xr
import ray
from ray.actor import ActorHandle
from tqdm import tqdm
from tqdm.dask import TqdmCallback
from pytorch_lightning import seed_everything


logger = set_color_logger(__name__)


def open_config(config_file: T.Union[str, Path, bytes]) -> dict:
    with open(config_file, 'r') as pf:
        config = yaml.safe_load(pf)

    return config


def get_centroid_coords_from_image(
    vi_path: Path,
    dst_crs: T.Optional[str] = None
) -> T.Tuple[float, float]:
    """Gets the lon/lat or x/y coordinates of a centroid
    """
    import geowombat as gw

    with gw.open(list(vi_path.glob('*.tif'))[0]) as src:
        df = src.gw.geodataframe
    centroid = df.to_crs(dst_crs).centroid

    return float(centroid.x), float(centroid.y)


def get_start_end_dates(
    feature_path: Path,
    start_year: int,
    start_date: str,
    end_date: str,
    date_format: str = '%Y%j',
    lat: T.Optional[float] = None
) -> T.Tuple[str, str]:
    """Gets the start and end dates from user args or from the filenames

    Returns:
        str (mm-dd), str (mm-dd)
    """
    # Get the first file for the start year
    filename = list(feature_path.glob(f"{start_year}*.tif"))[0]
    # Get the date from the file name
    file_dt = datetime.strptime(filename.stem, date_format)

    if start_date is not None:
        start_date = start_date
    else:
        start_date = file_dt.strftime('%m-%d')
    if end_date is not None:
        end_date = end_date
    else:
        end_date = file_dt.strftime('%m-%d')

    month = int(start_date.split('-')[0])

    if lat is not None:
        if lat > 0:
            # Expected time series start in northern hemisphere winter
            if 2 < month < 11:
                logger.warning(
                    f"The time series start date is {start_date} but the time series is in the Northern hemisphere."
                )
        else:
            # Expected time series start in northern southern winter
            if (month < 5) or (month > 9):
                logger.warning(
                    f"The time series start date is {start_date} but the time series is in the Southern hemisphere."
                )

    return start_date, end_date


def get_image_list(
    ppaths: ProjectPaths,
    region: str,
    predict_year: int,
    start_date: str,
    end_date: str,
    config: dict,
    date_format: str,
    skip_index: int
):
    """Gets a list of the time series images
    """
    image_list = []
    for image_vi in model_preprocessing.VegetationIndices(
        image_vis=config['image_vis']
    ).image_vis:
        # Set the full path to the images
        if str(ppaths.image_path).endswith('time_series_vars'):
            vi_path = ppaths.image_path / region / image_vi
        else:
            vi_path = ppaths.image_path / region / 'brdf_ts' / 'ms' / image_vi

        if not vi_path.is_dir():
            logger.warning(f'{str(vi_path)} does not exist')
            continue

        # Get the centroid coordinates of the grid
        lat = get_centroid_coords_from_image(vi_path, dst_crs='epsg:4326')[1]
        # Get the start and end dates
        start_date, end_date = get_start_end_dates(
            vi_path,
            start_year=predict_year-1,
            start_date=start_date,
            end_date=end_date,
            date_format=date_format,
            lat=lat
        )
        # Get the requested time slice
        ts_list = model_preprocessing.get_time_series_list(
            vi_path, config['predict_year']-1, start_date, end_date, date_format=date_format
        )
        if len(ts_list) <= 1:
            continue

        if skip_index > 0:
            image_list += ts_list[::skip_index]
        else:
            image_list += ts_list

    return image_list


@ray.remote
class ProgressBarActor:
    """
    Reference:
        https://docs.ray.io/en/releases-1.11.1/ray-core/examples/progress_bar.html
    """
    counter: int
    delta: int
    event: asyncio.Event

    def __init__(self) -> None:
        self.counter = 0
        self.delta = 0
        self.event = asyncio.Event()

    def update(self, num_items_completed: int) -> None:
        """Updates the ProgressBar with the incremental
        number of items that were just completed.
        """
        self.counter += num_items_completed
        self.delta += num_items_completed
        self.event.set()

    async def wait_for_update(self) -> T.Tuple[int, int]:
        """Blocking call.

        Waits until somebody calls `update`, then returns a tuple of
        the number of updates since the last call to
        `wait_for_update`, and the total number of completed items.
        """
        await self.event.wait()
        self.event.clear()
        saved_delta = self.delta
        self.delta = 0

        return saved_delta, self.counter

    def get_counter(self) -> int:
        """Returns the total number of complete items.
        """
        return self.counter


class ProgressBar:
    """
    Reference:
        https://docs.ray.io/en/releases-1.11.1/ray-core/examples/progress_bar.html
    """
    progress_actor: ActorHandle
    total: int
    desc: str
    position: int
    leave: bool
    pbar: tqdm

    def __init__(
        self,
        total: int,
        desc: str = "",
        position: int = 0,
        leave: bool = True
    ):
        # Ray actors don't seem to play nice with mypy, generating
        # a spurious warning for the following line,
        # which we need to suppress. The code is fine.
        self.progress_actor = ProgressBarActor.remote()  # type: ignore
        self.total = total
        self.desc = desc
        self.position = position
        self.leave = leave

    @property
    def actor(self) -> ActorHandle:
        """Returns a reference to the remote `ProgressBarActor`.

        When you complete tasks, call `update` on the actor.
        """
        return self.progress_actor

    def print_until_done(self) -> None:
        """Blocking call.

        Do this after starting a series of remote Ray tasks, to which you've
        passed the actor handle. Each of them calls `update` on the actor.
        When the progress meter reaches 100%, this method returns.
        """
        pbar = tqdm(
            desc=self.desc,
            position=self.position,
            total=self.total,
            leave=self.leave
        )
        while True:
            delta, counter = ray.get(self.actor.wait_for_update.remote())
            pbar.update(delta)
            if counter >= self.total:
                pbar.close()
                return


class BlockWriter(object):
    def _build_slice(self, window: Window) -> tuple:
        return (
            slice(0, None),
            slice(window.row_off, window.row_off+window.height),
            slice(window.col_off, window.col_off+window.width)
        )

    def predict_write_block(self, w: Window, w_pad: Window):
        slc = self._build_slice(w_pad)
        # Create the data for the chunk
        data = create_network_data(
            self.ts[slc].gw.compute(num_workers=1),
            ntime=self.ntime,
            nbands=self.nbands
        )
        # Apply inference on the chunk
        stack = cultionet.predict(
            lit_model=self.lit_model,
            data=data,
            written=None, #self.dst.read(self.bands[-1], window=w_pad),
            data_values=self.data_values,
            w=w,
            w_pad=w_pad,
            device=self.device,
            include_maskrcnn=self.include_maskrcnn
        )
        # Write the prediction stack to file
        with filelock.FileLock('./dst.lock'):
            self.dst.write(
                stack,
                indexes=range(1, self.dst.profile['count']+1),
                window=w
            )


class WriterModule(BlockWriter):
    def __init__(
        self,
        out_path: T.Union[str, Path],
        mode: str,
        profile: dict,
        ntime: int,
        nbands: int,
        filters: int,
        num_classes: int,
        ts: xr.DataArray,
        data_values: torch.Tensor,
        ppaths: ProjectPaths,
        device: str,
        scale_factor: float,
        include_maskrcnn: bool
    ) -> None:
        self.out_path = out_path
        # Create the output file
        if mode == 'w':
            with rio.open(self.out_path, mode=mode, **profile):
                pass

        self.dst = rio.open(self.out_path, mode='r+')

        self.ntime = ntime
        self.nbands = nbands
        self.ts = ts
        self.data_values = data_values
        self.ppaths = ppaths
        self.device = device
        self.scale_factor = scale_factor
        self.include_maskrcnn = include_maskrcnn
        # self.bands = [1, 2, 3] #+ list(range(4, 4+num_classes-1))
        # if self.include_maskrcnn:
        #     self.bands.append(self.bands[-1] + 1)

        self.lit_model = cultionet.load_model(
            ckpt_file=self.ppaths.ckpt_file,
            model_file=self.ppaths.ckpt_file.parent / 'cultionet.pt',
            num_features=ntime*nbands,
            num_time_features=ntime,
            filters=filters,
            num_classes=num_classes,
            device=self.device,
            enable_progress_bar=False
        )[1]

    def close_open(self):
        self.close()
        self.dst = rio.open(self.out_path, mode='r+')

    def close(self):
        self.dst.close()

    @abstractmethod
    def write(
        self,
        w: Window,
        w_pad: Window,
        pba: T.Optional[T.Union[ActorHandle, int]] = None
    ):
        raise NotImplementedError


@ray.remote
class RemoteWriter(WriterModule):
    """A concurrent writer with Ray
    """
    def __init__(
        self,
        out_path: T.Union[str, Path],
        mode: str,
        profile: dict,
        ntime: int,
        nbands: int,
        filters: int,
        num_classes: int,
        ts: xr.DataArray,
        data_values: torch.Tensor,
        ppaths: ProjectPaths,
        device: str,
        scale_factor: float,
        include_maskrcnn: bool
    ) -> None:
        super().__init__(
            out_path=out_path,
            mode=mode,
            profile=profile,
            ntime=ntime,
            nbands=nbands,
            filters=filters,
            num_classes=num_classes,
            ts=ts,
            data_values=data_values,
            ppaths=ppaths,
            device=device,
            scale_factor=scale_factor,
            include_maskrcnn=include_maskrcnn
        )

    def write(
        self,
        w: Window,
        w_pad: Window,
        pba: ActorHandle = None
    ):
        self.predict_write_block(w, w_pad)
        if pba is not None:
            pba.update.remote(1)


class SerialWriter(WriterModule):
    """A serial writer
    """
    def __init__(
        self,
        out_path: T.Union[str, Path],
        mode: str,
        profile: dict,
        ntime: int,
        nbands: int,
        filters: int,
        num_classes: int,
        ts: xr.DataArray,
        data_values: torch.Tensor,
        ppaths: ProjectPaths,
        device: str,
        scale_factor: float,
        include_maskrcnn: bool
    ) -> None:
        super().__init__(
            out_path=out_path,
            mode=mode,
            profile=profile,
            ntime=ntime,
            nbands=nbands,
            filters=filters,
            num_classes=num_classes,
            ts=ts,
            data_values=data_values,
            ppaths=ppaths,
            device=device,
            scale_factor=scale_factor,
            include_maskrcnn=include_maskrcnn
        )

    def write(
        self,
        w: Window,
        w_pad: Window,
        pba: int = None
    ):
        self.predict_write_block(w, w_pad)
        self.close_open()
        if pba is not None:
            pba.update(1)


def predict_image(args):
    logging.getLogger('pytorch_lightning').setLevel(logging.WARNING)

    config = open_config(args.config_file)

    # This is a helper function to manage paths
    ppaths = setup_paths(
        args.project_path, append_ts=True if args.append_ts == 'y' else False
    )
    # Load the z-score norm values
    data_values = torch.load(ppaths.norm_file)
    with open(ppaths.classes_info_path, mode='r') as f:
        class_info = json.load(f)

    num_classes = args.num_classes if args.num_classes is not None else class_info['max_crop_class'] + 1
    edge_class = args.edge_class if args.edge_class is not None else class_info['edge_class']

    if args.data_path is not None:
        ds = EdgeDataset(
            ppaths.predict_path,
            data_means=data_values.mean,
            data_stds=data_values.std,
            pattern=f'data_{args.region}_{args.predict_year}*.pt'
        )
        ckpt_file = ppaths.ckpt_path / 'last.ckpt'
        temperature_scales_file = ckpt_file.parent / 'temperature' / 'temperature.scales'
        edge_temperature = None
        crop_temperature = None
        if temperature_scales_file.is_file():
            with open(temperature_scales_file, mode='r') as f:
                temperature_scales = json.load(f)
            crop_temperature = torch.tensor([temperature_scales['crop']])
            if 'edge' in temperature_scales:
                edge_temperature = torch.tensor([temperature_scales['edge']])
            if torch.cuda.is_available():
                crop_temperature = crop_temperature.to('cuda')
                if 'edge' in temperature_scales:
                    edge_temperature = edge_temperature.to('cuda')

        cultionet.predict_lightning(
            reference_image=args.reference_image,
            out_path=args.out_path,
            ckpt=ckpt_file,
            dataset=ds,
            batch_size=args.batch_size,
            load_batch_workers=args.load_batch_workers,
            device=args.device,
            precision=args.precision,
            num_classes=num_classes,
            ref_res=ds[0].res,
            resampling=ds[0].resampling,
            compression=args.compression,
            crop_temperature=crop_temperature,
            edge_temperature=edge_temperature
        )

        if args.delete_dataset:
            ds.cleanup()
    else:
        try:
            tmp = int(args.grid_id)
            region = f'{tmp:06d}'
        except ValueError:
            region = args.grid_id

        # Get the image list
        image_list = get_image_list(
            ppaths,
            region=region,
            predict_year=args.predict_year,
            start_date=args.start_date,
            end_date=args.end_date,
            config=config,
            date_format=args.date_format,
            skip_index=args.skip_index
        )

        with gw.open(
            image_list,
            stack_dim='band',
            band_names=list(range(1, len(image_list)+1))
        ) as src_ts:
            time_series = (
                (src_ts * args.gain + args.offset)
                .astype('float64')
                .clip(0, 1)
            )
            if args.preload_data:
                with TqdmCallback(desc='Loading data'):
                    time_series.load(num_workers=args.processes)
            # Get the image dimensions
            nvars = model_preprocessing.VegetationIndices(image_vis=config['image_vis']).n_vis
            nfeas, height, width = time_series.shape
            ntime = int(nfeas / nvars)
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
                # Orientation (+1) + distance (+1) + edge (+1) + crop (+1) crop types (+N)
                # `num_classes` includes background
                'count': 3 + num_classes - 1,
                'dtype': 'uint16',
                'blockxsize': 64 if 64 < width else width,
                'blockysize': 64 if 64 < height else height,
                'driver': 'GTiff',
                'sharing': False,
                'compress': args.compression
            }
            profile['tiled'] = True if max(profile['blockxsize'], profile['blockysize']) >= 16 else False

            # Get the time and band count
            ntime, nbands = get_image_list_dims(
                image_list,
                time_series
            )

            if args.processes == 1:
                serial_writer = SerialWriter(
                    out_path=args.out_path,
                    mode=args.mode,
                    profile=profile,
                    ntime=ntime,
                    nbands=nbands,
                    filters=args.filters,
                    num_classes=num_classes,
                    ts=time_series,
                    data_values=data_values,
                    ppaths=ppaths,
                    device=args.device,
                    scale_factor=SCALE_FACTOR,
                    include_maskrcnn=args.include_maskrcnn
                )
                try:
                    with tqdm(total=len(windows), desc='Predicting windows', position=0) as pbar:
                        results = [
                            serial_writer.write(w, w_pad, pba=pbar) for w, w_pad in windows
                        ]
                    serial_writer.close()
                except Exception as e:
                    serial_writer.close()
                    logger.exception(f"The predictions failed because {e}.")
            else:
                if ray.is_initialized():
                    logger.warning('The Ray cluster is already running.')
                else:
                    if args.device == 'gpu':
                        # TODO: support multiple GPUs through CLI
                        try:
                            ray.init(num_cpus=args.processes, num_gpus=1)
                        except KeyError as e:
                            logger.exception(f"Ray could not be instantiated with a GPU because {e}.")
                    else:
                        ray.init(num_cpus=args.processes)
                assert ray.is_initialized(), 'The Ray cluster is not running.'
                # Setup the remote ray writer
                remote_writer = RemoteWriter.options(
                    max_concurrency=args.processes
                ).remote(
                    out_path=args.out_path,
                    mode=args.mode,
                    profile=profile,
                    ntime=ntime,
                    nbands=nbands,
                    filters=args.filters,
                    num_classes=num_classes,
                    ts=ray.put(time_series),
                    data_values=data_values,
                    ppaths=ppaths,
                    device=args.device,
                    scale_factor=SCALE_FACTOR,
                    include_maskrcnn=args.include_maskrcnn
                )
                actor_chunksize = args.processes * 8
                try:
                    with tqdm(total=len(windows), desc='Predicting windows', position=0) as pbar:
                        for wchunk in range(0, len(windows)+actor_chunksize, actor_chunksize):
                            chunk_windows = windows[wchunk:wchunk+actor_chunksize]
                            pbar.set_description(
                                f"Windows {wchunk:,d}--{wchunk+len(chunk_windows):,d}"
                            )
                            # pb = ProgressBar(
                            #     total=len(chunk_windows),
                            #     desc=f'Chunks {wchunk}-{wchunk+len(chunk_windows)}',
                            #     position=1,
                            #     leave=False
                            # )
                            # tqdm_actor = pb.actor
                            # Write each window concurrently
                            results = [
                                remote_writer.write.remote(w, w_pad)
                                for w, w_pad in chunk_windows
                            ]
                            # Initiate the processing
                            # pb.print_until_done()
                            ray.get(results)
                            # Close the file
                            ray.get(remote_writer.close_open.remote())
                            pbar.update(len(chunk_windows))
                    ray.get(remote_writer.close.remote())
                    ray.shutdown()
                except Exception as e:
                    ray.get(remote_writer.close.remote())
                    ray.shutdown()
                    logger.exception(f"The predictions failed because {e}.")


def cycle_data(
    year_lists: list,
    regions_lists: list,
    project_path_lists: list,
    ref_res_lists: list
):
    for years, regions, project_path, ref_res in zip(
        year_lists,
        regions_lists,
        project_path_lists,
        ref_res_lists
    ):
        for region in regions:
            for image_year in years:
                yield region, image_year, project_path, ref_res


def get_centroid_coords(df: gpd.GeoDataFrame, dst_crs: T.Optional[str] = None) -> T.Tuple[float, float]:
    """Gets the lon/lat or x/y coordinates of a centroid
    """
    centroid = df.to_crs(dst_crs).centroid

    return float(centroid.x), float(centroid.y)


def create_datasets(args):
    config = open_config(args.config_file)
    project_path_lists = [args.project_path]
    ref_res_lists = [args.ref_res]

    if hasattr(args, 'max_crop_class'):
        assert isinstance(args.max_crop_class, int), \
            'The maximum crop class value must be given.'

        region_as_list = config['regions'] is not None
        region_as_file = config['region_id_file'] is not None

        assert (
            region_as_list or region_as_file
        ), "Only submit region as a list or as a given file"

    if hasattr(args, 'time_series_path') and (args.time_series_path is not None):
        inputs = model_preprocessing.TrainInputs(
            regions=[Path(args.time_series_path).name],
            years=[args.predict_year]
        )
    else:
        if region_as_file:
            file_path = config['region_id_file']
            if not Path(file_path).is_file():
                raise IOError('The id file does not exist')
            id_data = pd.read_csv(file_path)
            assert "id" in id_data.columns, f"id column not found in {file_path}."
            regions = id_data['id'].unique().tolist()
        else:
            regions = list(range(config['regions'][0], config['regions'][1]+1))

        inputs = model_preprocessing.TrainInputs(
            regions=regions,
            years=config['years']
        )

    for region, end_year, project_path, ref_res in cycle_data(
        inputs.year_lists,
        inputs.regions_lists,
        project_path_lists,
        ref_res_lists
    ):
        ppaths = setup_paths(
            project_path,
            append_ts=True if args.append_ts == 'y' else False
        )

        try:
            tmp = int(region)
            region = f'{tmp:06d}'
        except ValueError:
            pass

        if args.destination == 'predict':
            df_grids = None
            df_edges = None
        else:
            # Read the training data
            grids = ppaths.edge_training_path / f'{region}_grid_{end_year}.gpkg'
            edges = ppaths.edge_training_path / f'{region}_edges_{end_year}.gpkg'
            if not grids.is_file():
                logger.warning(f'{grids} does not exist.')
                continue

            df_grids = gpd.read_file(grids)

            if not edges.is_file():
                edges = ppaths.edge_training_path / f'{region}_poly_{end_year}.gpkg'
            if not edges.is_file():
                # No training polygons
                df_edges = gpd.GeoDataFrame(data=[], geometry=[], crs=df_grids.crs)
            else:
                df_edges = gpd.read_file(edges)

        image_list = []
        for image_vi in model_preprocessing.VegetationIndices(
            image_vis=config['image_vis']
        ).image_vis:
            # Set the full path to the images
            vi_path = ppaths.image_path / args.feature_pattern.format(
                region=region,
                image_vi=image_vi
            )

            if not vi_path.is_dir():
                logger.warning(f'{str(vi_path)} does not exist')
                continue

            # Get the centroid coordinates of the grid
            lat = None
            if args.destination != 'predict':
                lat = get_centroid_coords(df_grids.centroid, dst_crs='epsg:4326')[1]
            # Get the start and end dates
            start_date, end_date = get_start_end_dates(
                vi_path,
                start_year=end_year-1,
                start_date=args.start_date,
                end_date=args.end_date,
                date_format=args.date_format,
                lat=lat,
            )
            # Get the requested time slice
            ts_list = model_preprocessing.get_time_series_list(
                vi_path, end_year-1, start_date, end_date, date_format=args.date_format
            )
            if len(ts_list) <= 1:
                continue

            if args.skip_index > 0:
                ts_list = ts_list[::args.skip_index]
            image_list += ts_list

        if args.destination != 'predict':
            class_info = {
                'max_crop_class': args.max_crop_class,
                'edge_class': args.max_crop_class + 1
            }
            with open(ppaths.classes_info_path, mode='w') as f:
                f.write(json.dumps(class_info))

        if image_list:
            if args.destination == 'predict':
                create_predict_dataset(
                    image_list=image_list,
                    region=region,
                    year=end_year,
                    process_path=ppaths.get_process_path(args.destination),
                    gain=args.gain,
                    offset=args.offset,
                    ref_res=ref_res,
                    resampling=args.resampling,
                    window_size=args.window_size,
                    padding=args.padding,
                    num_workers=args.num_workers
                )
            else:
                create_dataset(
                    image_list=image_list,
                    df_grids=df_grids,
                    df_edges=df_edges,
                    max_crop_class=args.max_crop_class,
                    group_id=f'{region}_{end_year}',
                    process_path=ppaths.get_process_path(args.destination),
                    transforms=args.transforms,
                    gain=args.gain,
                    offset=args.offset,
                    ref_res=ref_res,
                    resampling=args.resampling,
                    num_workers=args.num_workers,
                    grid_size=args.grid_size,
                    n_ts=args.n_ts,
                    instance_seg=args.instance_seg,
                    zero_padding=args.zero_padding,
                    crop_column=args.crop_column,
                    keep_crop_classes=args.keep_crop_classes,
                    replace_dict=args.replace_dict
                )


def train_maskrcnn(args):
    seed_everything(args.random_seed, workers=True)

    # This is a helper function to manage paths
    ppaths = setup_paths(args.project_path, ckpt_name='maskrcnn.ckpt')

    if (
        (args.expected_dim is not None)
        or not ppaths.norm_file.is_file()
        or (ppaths.norm_file.is_file() and args.recalc_zscores)
    ):
        ds = EdgeDataset(
            ppaths.train_path,
            processes=args.processes,
            threads_per_worker=args.threads,
            random_seed=args.random_seed
        )
    # Check dimensions
    if args.expected_dim is not None:
        try:
            ds.check_dims(
                args.expected_dim,
                args.delete_mismatches,
                args.dim_color
            )
        except TensorShapeError as e:
            raise ValueError(e)
    # Get the normalization means and std. deviations on the train data
    # Calculate the values needed to transform to z-scores, using
    # the training data
    if ppaths.norm_file.is_file():
        if args.recalc_zscores:
            ppaths.norm_file.unlink()
    if not ppaths.norm_file.is_file():
        train_ds = ds.split_train_val(val_frac=args.val_frac)[0]
        data_values = get_norm_values(
            dataset=train_ds,
            batch_size=args.batch_size,
            mean_color=args.mean_color,
            sse_color=args.sse_color
        )
        torch.save(data_values, str(ppaths.norm_file))
    else:
        data_values = torch.load(str(ppaths.norm_file))

    # Create the train data object again, this time passing
    # the means and standard deviation tensors
    ds = EdgeDataset(
        ppaths.train_path,
        data_means=data_values.mean,
        data_stds=data_values.std,
        random_seed=args.random_seed
    )
    # Check for a test dataset
    test_ds = None
    if list((ppaths.test_process_path).glob('*.pt')):
        test_ds = EdgeDataset(
            ppaths.test_path,
            data_means=data_values.mean,
            data_stds=data_values.std,
            random_seed=args.random_seed
        )
        if args.expected_dim is not None:
            try:
                test_ds.check_dims(
                    args.expected_dim,
                    args.delete_mismatches,
                    args.dim_color
                )
            except TensorShapeError as e:
                raise ValueError(e)

    # Fit the model
    cultionet.fit_maskrcnn(
        dataset=ds,
        ckpt_file=ppaths.ckpt_file,
        test_dataset=test_ds,
        val_frac=args.val_frac,
        batch_size=args.batch_size,
        epochs=args.epochs,
        accumulate_grad_batches=args.accumulate_grad_batches,
        learning_rate=args.learning_rate,
        filters=args.filters,
        num_classes=args.num_classes,
        reset_model=args.reset_model,
        auto_lr_find=args.auto_lr_find,
        device=args.device,
        gradient_clip_val=args.gradient_clip_val,
        early_stopping_patience=args.patience,
        weight_decay = args.weight_decay,
        precision=args.precision,
        stochastic_weight_averaging=args.stochastic_weight_averaging,
        stochastic_weight_averaging_lr=args.stochastic_weight_averaging_lr,
        stochastic_weight_averaging_start=args.stochastic_weight_averaging_start,
        model_pruning=args.model_pruning,
        resize_height=args.resize_height,
        resize_width=args.resize_width,
        min_image_size=args.min_image_size,
        max_image_size=args.max_image_size,
        trainable_backbone_layers=args.trainable_backbone_layers
    )


def spatial_kfoldcv(args):
    ppaths = setup_paths(args.project_path)

    with open(ppaths.classes_info_path, mode='r') as f:
        class_info = json.load(f)

    ds = EdgeDataset(
        ppaths.train_path,
        processes=args.processes,
        threads_per_worker=args.threads,
        random_seed=args.random_seed
    )
    # Read or create the spatial partitions (folds)
    ds.get_spatial_partitions(
        spatial_partitions=args.spatial_partitions,
        splits=args.splits
    )
    for k, (partition_name, train_ds, test_ds) in enumerate(
        ds.spatial_kfoldcv_iter(args.partition_column)
    ):
        logger.info(f"Fold {k} of {len(ds.spatial_partitions.index)}, partition {partition_name} ...")
        # Normalize the partition
        temp_ds = train_ds.split_train_val(val_frac=args.val_frac)[0]
        data_values = get_norm_values(
            dataset=temp_ds,
            class_info=class_info,
            batch_size=args.batch_size,
            mean_color=args.mean_color,
            sse_color=args.sse_color
        )
        train_ds.data_means = data_values.mean
        train_ds.data_stds = data_values.std
        test_ds.data_means = data_values.mean
        test_ds.data_stds = data_values.std

        # Get balanced class weights
        # Reference: https://github.com/scikit-learn/scikit-learn/blob/f3f51f9b6/sklearn/utils/class_weight.py#L10
        recip_freq = data_values.crop_counts[1:].sum() / ((len(data_values.crop_counts)-1) * data_values.crop_counts[1:])
        class_weights = recip_freq[torch.arange(0, len(data_values.crop_counts)-1)]
        class_weights = torch.tensor([0] + list(class_weights), dtype=torch.float)

        # Fit the model
        cultionet.fit(
            dataset=train_ds,
            ckpt_file=ppaths.ckpt_file,
            test_dataset=test_ds,
            val_frac=args.val_frac,
            batch_size=args.batch_size,
            load_batch_workers=args.load_batch_workers,
            epochs=args.epochs,
            accumulate_grad_batches=args.accumulate_grad_batches,
            learning_rate=args.learning_rate,
            filters=args.filters,
            num_classes=args.num_classes if args.num_classes is not None else class_info['max_crop_class'] + 1,
            edge_class=args.edge_class if args.edge_class is not None else class_info['edge_class'],
            class_weights=class_weights,
            reset_model=True,
            auto_lr_find=False,
            device=args.device,
            gradient_clip_val=args.gradient_clip_val,
            early_stopping_patience=args.patience,
            weight_decay=args.weight_decay,
            precision=args.precision,
            stochastic_weight_averaging=args.stochastic_weight_averaging,
            model_pruning=args.model_pruning
        )
        # Rename the test metric JSON file
        (
            ppaths.ckpt_path / 'test.metrics'
        ).rename(
            ppaths.ckpt_path / f"fold-{k}-{partition_name.replace(' ', '_')}.metrics"
        )


def train_model(args):
    seed_everything(args.random_seed, workers=True)

    # This is a helper function to manage paths
    ppaths = setup_paths(args.project_path)

    with open(ppaths.classes_info_path, mode='r') as f:
        class_info = json.load(f)

    if (
        (args.expected_dim is not None)
        or not ppaths.norm_file.is_file()
        or (ppaths.norm_file.is_file() and args.recalc_zscores)
    ):
        ds = EdgeDataset(
            ppaths.train_path,
            processes=args.processes,
            threads_per_worker=args.threads,
            random_seed=args.random_seed
        )
    # Check dimensions
    if args.expected_dim is not None:
        try:
            ds.check_dims(
                args.expected_dim,
                args.expected_height,
                args.expected_width,
                args.delete_mismatches,
                args.dim_color
            )
        except TensorShapeError as e:
            raise ValueError(e)
        ds = EdgeDataset(
            ppaths.train_path,
            processes=args.processes,
            threads_per_worker=args.threads,
            random_seed=args.random_seed
        )
    # Get the normalization means and std. deviations on the train data
    # Calculate the values needed to transform to z-scores, using
    # the training data
    if ppaths.norm_file.is_file():
        if args.recalc_zscores:
            ppaths.norm_file.unlink()
    if not ppaths.norm_file.is_file():
        if args.spatial_partitions is not None:
            train_ds = ds.split_train_val_by_partition(
                spatial_partitions=args.spatial_partitions,
                partition_column=args.partition_column,
                val_frac=args.val_frac,
                partition_name=args.partition_name
            )[0]
        else:
            train_ds = ds.split_train_val(val_frac=args.val_frac)[0]
        data_values = get_norm_values(
            dataset=train_ds,
            class_info=class_info,
            batch_size=args.batch_size,
            mean_color=args.mean_color,
            sse_color=args.sse_color
        )
        torch.save(data_values, str(ppaths.norm_file))
    else:
        data_values = torch.load(str(ppaths.norm_file))

    # Create the train data object again, this time passing
    # the means and standard deviation tensors
    ds = EdgeDataset(
        ppaths.train_path,
        data_means=data_values.mean,
        data_stds=data_values.std,
        crop_counts=data_values.crop_counts,
        edge_counts=data_values.edge_counts,
        random_seed=args.random_seed
    )

    # Check for a test dataset
    test_ds = None
    if list((ppaths.test_process_path).glob('*.pt')):
        test_ds = EdgeDataset(
            ppaths.test_path,
            data_means=data_values.mean,
            data_stds=data_values.std,
            crop_counts=data_values.crop_counts,
            edge_counts=data_values.edge_counts,
            random_seed=args.random_seed
        )
        if args.expected_dim is not None:
            try:
                test_ds.check_dims(
                    args.expected_dim,
                    args.delete_mismatches,
                    args.dim_color
                )
            except TensorShapeError as e:
                raise ValueError(e)
            test_ds = EdgeDataset(
                ppaths.test_path,
                data_means=data_values.mean,
                data_stds=data_values.std,
                crop_counts=data_values.crop_counts,
                edge_counts=data_values.edge_counts,
                random_seed=args.random_seed
            )

    # Get balanced class weights
    # Reference: https://github.com/scikit-learn/scikit-learn/blob/f3f51f9b6/sklearn/utils/class_weight.py#L10
    def get_class_weights(counts: torch.Tensor) -> torch.Tensor:
        recip_freq = counts.sum() / (len(counts) * counts)
        weights = recip_freq[torch.arange(0, len(counts))]

        if torch.cuda.is_available():
            return weights.to('cuda')
        else:
            return weights

    class_weights = get_class_weights(data_values.crop_counts)
    edge_weights = get_class_weights(data_values.edge_counts)

    # Fit the model
    cultionet.fit(
        dataset=ds,
        ckpt_file=ppaths.ckpt_file,
        test_dataset=test_ds,
        val_frac=args.val_frac,
        spatial_partitions=args.spatial_partitions,
        partition_name=args.partition_name,
        partition_column=args.partition_column,
        batch_size=args.batch_size,
        epochs=args.epochs,
        accumulate_grad_batches=args.accumulate_grad_batches,
        learning_rate=args.learning_rate,
        filters=args.filters,
        num_classes=args.num_classes if args.num_classes is not None else class_info['max_crop_class'] + 1,
        edge_class=args.edge_class if args.edge_class is not None else class_info['edge_class'],
        class_weights=class_weights,
        edge_weights=edge_weights,
        reset_model=args.reset_model,
        auto_lr_find=args.auto_lr_find,
        device=args.device,
        profiler=args.profiler,
        gradient_clip_val=args.gradient_clip_val,
        early_stopping_patience=args.patience,
        weight_decay=args.weight_decay,
        precision=args.precision,
        stochastic_weight_averaging=args.stochastic_weight_averaging,
        stochastic_weight_averaging_lr=args.stochastic_weight_averaging_lr,
        stochastic_weight_averaging_start=args.stochastic_weight_averaging_start,
        model_pruning=args.model_pruning
    )


def main():
    args_config = open_config((Path(__file__).parent / 'args.yml').absolute())

    parser = argparse.ArgumentParser(
        description='Cultionet models',
        formatter_class=argparse.RawTextHelpFormatter,
        epilog=args_config['epilog']
    )

    subparsers = parser.add_subparsers(dest='process')
    available_processes = [
        'create', 'create-predict', 'skfoldcv', 'train', 'maskrcnn', 'predict', 'version'
    ]
    for process in available_processes:
        subparser = subparsers.add_parser(process)

        if process == 'version':
            continue

        subparser.add_argument(
            '-p',
            '--project-path',
            dest='project_path',
            help='The project path (the directory that contains the grid ids)'
        )

        process_dict = args_config[process.replace('-', '_')]
        if process in ('skfoldcv', 'maskrcnn'):
            process_dict.update(args_config['train'])
        if process in ('train', 'maskrcnn', 'predict', 'skfoldcv'):
            process_dict.update(args_config['train_predict'])
            process_dict.update(args_config['shared_partitions'])
        if process in ('create', 'create-predict'):
            process_dict.update(args_config['shared_create'])
        if process in ('create', 'create-predict', 'predict'):
            process_dict.update(args_config['shared_image'])
        process_dict.update(args_config['dates'])
        for process_key, process_values in process_dict.items():
            if 'kwargs' in process_values:
                kwargs = process_values['kwargs']
                for key, value in kwargs.items():
                    if isinstance(value, str) and value.startswith('&'):
                        kwargs[key] = getattr(builtins, value.replace('&', ''))

            else:
                process_values['kwargs'] = {}
            key_args = ()
            if len(process_values['short']) > 0:
                key_args += (f"-{process_values['short']}",)
            if len(process_values['long']) > 0:
                key_args += (f"--{process_values['long']}",)
            subparser.add_argument(
                *key_args,
                dest=process_key,
                help=f"{process_values['help']} (default: %(default)s)",
                **process_values['kwargs']
            )

        if process in ('create', 'create-predict', 'predict'):
            subparser.add_argument(
                '--config-file',
                dest='config_file',
                help='The configuration YAML file (default: %(default)s)',
                default=(Path(__file__).parent / 'config.yml').absolute()
            )

    args = parser.parse_args()
    if args.process == 'create-predict':
        setattr(args, 'destination', 'predict')

    if args.process == 'version':
        print(cultionet.__version__)
        return

    if hasattr(args, 'replace_dict'):
        if args.replace_dict is not None:
            setattr(args, 'replace_dict', ast.literal_eval(args.replace_dict))

    if args.process in ('create', 'create-predict'):
        create_datasets(args)
    elif args.process == 'skfoldcv':
        spatial_kfoldcv(args)
    elif args.process == 'train':
        train_model(args)
    elif args.process == 'maskrcnn':
        train_maskrcnn(args)
    elif args.process == 'predict':
        predict_image(args)


if __name__ == '__main__':
    main()
