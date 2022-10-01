#!/usr/bin/env python

import argparse
import typing as T
import logging
from pathlib import Path
from datetime import datetime
import asyncio
import filelock

import cultionet
from cultionet.data.datasets import EdgeDataset
from cultionet.utils.project_paths import setup_paths, ProjectPaths
from cultionet.utils.normalize import get_norm_values
from cultionet.data.create import create_dataset
from cultionet.data.utils import get_image_list_dims
from cultionet.utils import model_preprocessing
from cultionet.data.utils import create_network_data, NetworkDataset
from cultionet.models.lightning import CultioLitModel

import torch
import geopandas as gpd
import pandas as pd
import yaml
from rasterio.windows import Window
import ray
from ray.actor import ActorHandle
from tqdm import tqdm
import xarray as xr


logger = logging.getLogger(__name__)

DEFAULT_AUGMENTATIONS = ['none', 'fliplr', 'flipud', 'flipfb',
                         'rot90', 'rot180', 'rot270',
                         'ts-warp', 'ts-noise', 'ts-drift']

SCALE_FACTOR = 10_000.0


def open_config(config_file: T.Union[str, Path, bytes]) -> dict:
    with open(config_file, 'r') as pf:
        config = yaml.load(pf, Loader=yaml.FullLoader)

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


def get_image_list(
    ppaths: ProjectPaths,
    region: str,
    config: dict
):
    """Gets a list of the time series images
    """
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

        # Get the image year
        filename = list(vi_path.glob(f"{config['predict_year']}*.tif"))[0]
        file_dt = datetime.strptime(filename.stem, '%Y%j')

        if config['start_date'] is not None:
            start_date = config['start_date']
        else:
            start_date = file_dt.strftime('%m-%d')
        if config['end_date'] is not None:
            end_date = config['end_date']
        else:
            end_date = file_dt.strftime('%m-%d')
        image_year = file_dt.year + 1

        # Get the centroid coordinates of the grid
        lat = get_centroid_coords_from_image(vi_path, dst_crs='epsg:4326')[1]
        if lat > 0:
            # Expected time series start in northern hemisphere winter
            if (file_dt.month > 2) or (file_dt.month < 11):
                logger.warning(
                    f"The time series start date is {file_dt.strftime('%Y-%m-%d')} but the time series is in the Northern hemisphere."
                )
        else:
            # Expected time series start in northern southern winter
            if (file_dt.month < 5) or (file_dt.month > 9):
                logger.warning(
                    f"The time series start date is {file_dt.strftime('%Y-%m-%d')} but the time series is in the Southern hemisphere."
                )

        # Get the requested time slice
        ts_list = model_preprocessing.get_time_series_list(
            vi_path, image_year, start_date, end_date
        )

        if len(ts_list) <= 1:
            continue

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


def predict_image(args):
    import geowombat as gw
    from geowombat.core.windows import get_window_offsets
    import rasterio as rio
    import torch
    from tqdm.dask import TqdmCallback

    logging.getLogger('pytorch_lightning').setLevel(logging.WARNING)

    config = open_config(args.config_file)

    # This is a helper function to manage paths
    ppaths = setup_paths(
        args.project_path, append_ts=True if args.append_ts == 'y' else False
    )
    # Load the z-score norm values
    data_values = torch.load(ppaths.norm_file)

    try:
        tmp = int(args.grid_id)
        region = f'{tmp:06d}'
    except ValueError:
        region = args.grid_id

    # Get the image list
    image_list = get_image_list(ppaths, region, config)

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
            'driver': 'GTiff',
            'sharing': True
        }
        profile['tiled'] = True if max(profile['blockxsize'], profile['blockysize']) >= 16 else False

        # Get the time and band count
        ntime, nbands = get_image_list_dims(image_list, time_series)

        @ray.remote
        class Writer(object):
            def __init__(
                self,
                window: Window,
                out_path: T.Union[str, Path],
                mode: str,
                profile: dict,
                ntime: int,
                nbands: int,
                ts: xr.DataArray,
                data_values: torch.Tensor,
                ppaths: ProjectPaths,
                filters: int,
                device: str,
                scale_factor: float
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
                self.filters = filters
                self.device = device
                self.scale_factor = scale_factor

                slc = self._build_slice(window)
                data = create_network_data(
                    self.ts[slc].data.compute(num_workers=1),
                    ntime=self.ntime,
                    nbands=self.nbands
                )
                self.lit_model = cultionet.load_model(
                    num_features=data.x.shape[1],
                    num_time_features=data.ntime,
                    ckpt_file=self.ppaths.ckpt_file,
                    filters=self.filters,
                    device=self.device,
                    enable_progress_bar=False
                )[1]

            def _build_slice(self, window: Window) -> tuple:
                return (
                    slice(0, None),
                    slice(window.row_off, window.row_off+window.height),
                    slice(window.col_off, window.col_off+window.width)
                )

            def close_open(self):
                self.close()
                self.dst = rio.open(self.out_path, mode='r+')

            def close(self):
                self.dst.close()

            def write_predictions(
                self,
                w: Window,
                w_pad: Window,
                pba: ActorHandle = None
            ):
                slc = self._build_slice(w_pad)
                # Create the data for the chunk
                data = create_network_data(
                    self.ts[slc].data.compute(num_workers=1),
                    ntime=self.ntime,
                    nbands=self.nbands
                )
                # Apply inference on the chunk
                stack = cultionet.predict(
                    lit_model=self.lit_model,
                    data=data,
                    data_values=self.data_values,
                    w=w,
                    w_pad=w_pad
                )
                # Write the prediction stack to file
                with filelock.FileLock('./dst.lock'):
                    self.dst.write(
                        (
                            (stack*self.scale_factor)
                            .clip(0, self.scale_factor)
                            .astype('uint16')
                        ),
                        indexes=[1, 2, 3, 4],
                        window=w
                    )
                pba.update.remote(1)

        if ray.is_initialized():
            logger.warning('The Ray cluster is already running.')
        else:
            ray.init(num_cpus=args.processes)
        assert ray.is_initialized(), 'The Ray cluster is not running.'
        # Setup the remote ray writer
        remote_writer = Writer.options(
            max_concurrency=args.processes
        ).remote(
            window=windows[0][1],
            out_path=args.out_path,
            mode=args.mode,
            profile=profile,
            ntime=ntime,
            nbands=nbands,
            ts=ray.put(time_series),
            data_values=data_values,
            ppaths=ppaths,
            filters=args.filters,
            device=args.device,
            scale_factor=SCALE_FACTOR
        )
        try:
            with tqdm(total=32, desc='Predicting windows', position=0) as pbar:
                for wchunk in range(0, 32+args.processes, args.processes):
                    chunk_windows = windows[wchunk:wchunk+args.processes]
                    pb = ProgressBar(
                        total=len(chunk_windows),
                        desc=f'Chunks {wchunk}-{wchunk+len(chunk_windows)}',
                        position=1,
                        leave=False
                    )
                    tqdm_actor = pb.actor
                    # Write each window concurrently
                    results = [
                        remote_writer.write_predictions.remote(w, w_pad, pba=tqdm_actor)
                        for w, w_pad in chunk_windows
                    ]
                    # Initiate the processing
                    pb.print_until_done()
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
    lc_paths_lists: list,
    ref_res_lists: list
):
    for years, regions, project_path, lc_path, ref_res in zip(
        year_lists,
        regions_lists,
        project_path_lists,
        lc_paths_lists,
        ref_res_lists
    ):
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

    region_as_list = config['regions'] is not None
    region_as_file = config["region_id_file"] is not None

    assert (
        region_as_list or region_as_file
    ), "Only submit region as a list or as a given file"


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
        except ValueError:
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
        accumulate_grad_batches=args.accumulate_grad_batches,
        learning_rate=args.learning_rate,
        filters=args.filters,
        random_seed=args.random_seed,
        reset_model=args.reset_model,
        auto_lr_find=args.auto_lr_find,
        device=args.device,
        gradient_clip_val=args.gradient_clip_val,
        early_stopping_patience=args.patience,
        stochastic_weight_avg=args.stochastic_weight_avg,
        weight_decay = args.weight_decay
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
            subparser.add_argument(
                '--weight-decay', dest='weight_decay',
                help='Sets the weight decay for Adam optimizer\'s regularization (default: %(default)s)',
                default=1e-5, type=float
            )
            subparser.add_argument(
                '-agb', '--accumulate-grad-batches', dest='accumulate_grad_batches',
                help='Sets the number of batches to apply gradients after (default: %(default)s)',
                default=1, type=int
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
            subparser.add_argument(
                '--processes', dest='processes', help='The number of concurrent processes (default: %(default)s)',
                default=1, type=int
            )
            subparser.add_argument(
                '--mode', dest='mode', help='The file open() mode (default: %(default)s)',
                default='w', choices=['w', 'r+']
            )
            subparser.add_argument(
                '--preload-data',
                dest='preload_data',
                help='Whether to preload the time series data into memory (default: %(default)s)',
                action='store_true'
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
