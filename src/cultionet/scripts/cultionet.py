#!/usr/bin/env python

import argparse
import builtins
import json
import logging
import typing as T
from collections import namedtuple
from datetime import datetime
from functools import partial
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import torch
import yaml
from fiona.errors import DriverError
from geowombat.core import sort_images_by_date
from joblib import delayed, parallel_config
from lightning import seed_everything
from rich.markdown import Markdown
from rich_argparse import RichHelpFormatter
from shapely.errors import GEOSException
from shapely.geometry import box

import cultionet
from cultionet.data.create import create_predict_dataset, create_train_batch
from cultionet.data.datasets import EdgeDataset
from cultionet.data.utils import split_multipolygons
from cultionet.enums import CLISteps, DataColumns, ModelNames
from cultionet.errors import TensorShapeError
from cultionet.model import CultionetParams
from cultionet.utils import model_preprocessing
from cultionet.utils.logging import set_color_logger
from cultionet.utils.model_preprocessing import ParallelProgress
from cultionet.utils.normalize import NormValues
from cultionet.utils.project_paths import ProjectPaths, setup_paths

logger = set_color_logger(__name__)


def open_config(config_file: T.Union[str, Path, bytes]) -> dict:
    with open(config_file, "r") as pf:
        config = yaml.safe_load(pf)

    return config


def get_centroid_coords_from_image(
    vi_path: Path, dst_crs: T.Optional[str] = None
) -> T.Tuple[float, float]:
    """Gets the lon/lat or x/y coordinates of a centroid."""
    import geowombat as gw

    with gw.open(list(vi_path.glob("*.tif"))[0]) as src:
        df = src.gw.geodataframe
    centroid = df.to_crs(dst_crs).centroid

    return float(centroid.x), float(centroid.y)


def get_start_end_dates(
    feature_path: Path,
    end_year: T.Union[int, str],
    start_mmdd: str,
    end_mmdd: str,
    num_months: int,
    date_format: str = "%Y%j",
    lat: T.Optional[float] = None,
) -> T.Tuple[str, str]:
    """Gets the start and end dates from user args or from the filenames.

    Returns
    =======
    str (mm-dd), str (mm-dd)
    """

    image_dict = sort_images_by_date(
        feature_path,
        '*.tif',
        date_pos=0,
        date_start=0,
        date_end=8,
        date_format=date_format,
    )
    image_df = pd.DataFrame(
        data=list(image_dict.keys()),
        columns=['filename'],
        index=list(image_dict.values()),
    )

    end_date_stamp = pd.Timestamp(f"{end_year}-{end_mmdd}")
    start_year = (end_date_stamp - pd.DateOffset(months=num_months - 1)).year
    start_date_stamp = pd.Timestamp(f"{start_year}-{start_mmdd}")
    image_df = image_df.loc[start_date_stamp:end_date_stamp]

    return image_df.index[0].strftime("%Y-%m-%d"), image_df.index[-1].strftime(
        "%Y-%m-%d"
    )


def get_image_list(
    ppaths: ProjectPaths,
    region: str,
    predict_year: int,
    start_date: str,
    end_date: str,
    config: dict,
    date_format: str,
    skip_index: int,
):
    """Gets a list of the time series images."""
    image_list = []
    for image_vi in model_preprocessing.VegetationIndices(
        image_vis=config["image_vis"]
    ).image_vis:
        # Set the full path to the images
        if str(ppaths.image_path).endswith("time_series_vars"):
            vi_path = ppaths.image_path / region / image_vi
        else:
            vi_path = ppaths.image_path / region / "brdf_ts" / "ms" / image_vi

        if not vi_path.is_dir():
            logger.warning(f"{str(vi_path)} does not exist")
            continue

        # Get the centroid coordinates of the grid
        lat = get_centroid_coords_from_image(vi_path, dst_crs="epsg:4326")[1]
        # Get the start and end dates
        start_date, end_date = get_start_end_dates(
            vi_path,
            start_year=predict_year - 1,
            start_date=start_date,
            end_date=end_date,
            date_format=date_format,
            lat=lat,
        )
        # Get the requested time slice
        ts_list = model_preprocessing.get_time_series_list(
            vi_path,
            config["predict_year"] - 1,
            start_date,
            end_date,
            date_format=date_format,
        )
        if len(ts_list) <= 1:
            continue

        if skip_index > 0:
            image_list += ts_list[::skip_index]
        else:
            image_list += ts_list

    return image_list


def predict_image(args):
    logging.getLogger("pytorch_lightning").setLevel(logging.WARNING)

    # This is a helper function to manage paths
    ppaths = setup_paths(
        args.project_path, append_ts=True if args.append_ts == "y" else False
    )

    # Load the z-score norm values
    norm_values = NormValues.from_file(ppaths.norm_file)

    ds = EdgeDataset(
        root=ppaths.predict_path,
        log_transform=args.log_transform,
        norm_values=norm_values,
        pattern=f"{args.region}_{args.start_date.replace('-', '')}_{args.end_date.replace('-', '')}*.pt",
    )

    cultionet.predict_lightning(
        reference_image=args.reference_image,
        out_path=args.out_path,
        ckpt=ppaths.ckpt_path / ModelNames.CKPT_NAME,
        dataset=ds,
        device=args.device,
        devices=args.devices,
        strategy=args.strategy,
        batch_size=args.batch_size,
        load_batch_workers=args.load_batch_workers,
        precision=args.precision,
        resampling=ds[0].resampling[0]
        if hasattr(ds[0], "resampling")
        else "nearest",
        compression=args.compression,
        is_transfer_model=args.process == CLISteps.PREDICT_TRANSFER,
    )

    if args.delete_dataset:
        ds.cleanup()


def create_one_id(
    args: namedtuple,
    config: dict,
    ppaths: ProjectPaths,
    region_df: gpd.GeoDataFrame,
    polygon_df: gpd.GeoDataFrame,
    processed_path: Path,
    bbox_offsets: T.Optional[T.List[T.Tuple[int, int]]] = None,
) -> None:
    """Creates a single dataset.

    Parameters
    ==========
    args
        An ``argparse`` ``namedtuple`` of CLI arguments.
    config
        The configuration.
    ppaths
        The project path object.
    region_df
        The region grid ``geopandas.GeoDataFrame``.
    polygon_df
        The region polygon ``geopandas.GeoDataFrame``.
    processed_path
        The time series path.
    bbox_offsets
        Bounding box (x, y) offsets as [(x, y)]. E.g., shifts of
        [(-1000, 0), (0, 1000)] would shift the grid left by 1,000 meters and
        then right by 1,000 meters.

        Note that the ``polygon_df`` should support the shifts outside of the grid.
    """

    row_id = processed_path.name

    bbox_offset_list = [(0, 0)]
    if bbox_offsets is not None:
        bbox_offset_list.extend(bbox_offsets)

    for grid_offset in bbox_offset_list:

        if args.destination != "predict":
            # Get the grid
            row_region_df = region_df.query(
                f"{DataColumns.GEOID} == '{row_id}'"
            )

            if row_region_df.empty:
                return

            left, bottom, right, top = row_region_df.total_bounds

            if grid_offset != (0, 0):
                # Create a new, shifted grid
                row_region_df = gpd.GeoDataFrame(
                    geometry=[
                        box(
                            left + grid_offset[1],
                            bottom + grid_offset[0],
                            right + grid_offset[1],
                            top + grid_offset[0],
                        ),
                    ],
                    crs=row_region_df.crs,
                )
                left, bottom, right, top = row_region_df.total_bounds

            # Clip the polygons to the current grid
            # NOTE: .cx gets all intersecting polygons and reduces the problem size for clip()
            polygon_df_intersection = polygon_df.cx[left:right, bottom:top]

            # Clip the polygons to the grid edges
            try:
                row_polygon_df = gpd.clip(
                    polygon_df_intersection,
                    row_region_df,
                )
            except GEOSException:
                try:
                    # Try clipping with any MultiPolygon split
                    row_polygon_df = gpd.clip(
                        split_multipolygons(polygon_df_intersection),
                        row_region_df,
                    )
                except GEOSException:
                    try:
                        # Try clipping with a ghost buffer
                        row_polygon_df = gpd.clip(
                            split_multipolygons(
                                polygon_df_intersection
                            ).assign(
                                geometry=polygon_df_intersection.geometry.buffer(
                                    0
                                )
                            ),
                            row_region_df,
                        )
                    except GEOSException:
                        logger.warning(
                            f"Could not create a dataset file for {row_id}."
                        )
                        return

            # Check for multi-polygons
            row_polygon_df = split_multipolygons(row_polygon_df)
            # Rather than check for a None CRS, just set it
            row_polygon_df = row_polygon_df.set_crs(
                polygon_df_intersection.crs, allow_override=True
            )

            end_year = int(row_region_df[DataColumns.YEAR])

        if args.add_year > 0:
            end_year += args.add_year

        image_list = []
        for image_vi in config["image_vis"]:
            # Set the full path to the images
            vi_path = ppaths.image_path.resolve().joinpath(
                args.feature_pattern.format(region=row_id, image_vi=image_vi)
            )

            if not vi_path.exists():
                logger.warning(
                    f"The {image_vi} path is missing for {str(vi_path)}."
                )
                return

            # Get the requested time slice
            ts_list = model_preprocessing.get_time_series_list(
                vi_path,
                date_format=args.date_format,
                start_date=pd.to_datetime(args.start_date)
                if args.destination == "predict"
                else None,
                end_date=pd.to_datetime(args.end_date)
                if args.destination == "predict"
                else None,
                end_year=end_year if args.destination != "predict" else None,
                start_mmdd=config["start_mmdd"],
                end_mmdd=config["end_mmdd"],
                num_months=config["num_months"],
            )

            if args.skip_index > 0:
                ts_list = ts_list[:: args.skip_index]

            image_list += ts_list

        if image_list:
            if args.destination == "predict":
                create_predict_dataset(
                    image_list=image_list,
                    region=row_id,
                    process_path=ppaths.get_process_path(args.destination),
                    date_format=args.date_format,
                    gain=args.gain,
                    offset=args.offset,
                    ref_res=args.ref_res,
                    resampling=args.resampling,
                    window_size=args.window_size,
                    padding=args.padding,
                    num_workers=args.num_workers,
                )
            else:
                class_info = {
                    "max_crop_class": args.max_crop_class,
                    "edge_class": args.max_crop_class + 1,
                }
                with open(ppaths.classes_info_path, mode="w") as f:
                    f.write(json.dumps(class_info))

                create_train_batch(
                    image_list=image_list,
                    df_grid=row_region_df,
                    df_polygons=row_polygon_df,
                    max_crop_class=args.max_crop_class,
                    region=row_id,
                    process_path=ppaths.get_process_path(args.destination),
                    date_format=args.date_format,
                    gain=args.gain,
                    offset=args.offset,
                    ref_res=args.ref_res,
                    resampling=args.resampling,
                    grid_size=args.grid_size,
                    crop_column=args.crop_column,
                    keep_crop_classes=args.keep_crop_classes,
                    replace_dict=args.replace_dict,
                    nonag_is_unknown=args.nonag_is_unknown,
                    all_touched=args.all_touched,
                )


def read_training(
    filename: T.Union[list, tuple, str, Path], columns: list
) -> gpd.GeoDataFrame:
    if isinstance(filename, (list, tuple)):
        try:
            df = pd.concat(
                [
                    gpd.read_file(
                        fn,
                        columns=columns,
                        engine="pyogrio",
                    )
                    for fn in filename
                ]
            ).reset_index(drop=True)

        except DriverError:
            raise IOError("The id file does not exist")

    else:
        filename = Path(filename)
        if not filename.exists():
            raise IOError("The id file does not exist")

        df = gpd.read_file(filename)

    return df


def create_dataset(args):
    """Creates a train or predict dataset."""

    config = open_config(args.config_file)

    ppaths: ProjectPaths = setup_paths(
        args.project_path,
        append_ts=True if args.append_ts == "y" else False,
    )

    if hasattr(args, "max_crop_class"):
        assert isinstance(
            args.max_crop_class, int
        ), "The maximum crop class value must be given."

    region_df = None
    polygon_df = None
    if args.destination == "train":
        region_id_file = config.get("region_id_file")
        polygon_file = config.get("polygon_file")

        if region_id_file is None:
            raise NameError("A region file or file list must be given.")

        if polygon_file is None:
            raise NameError("A polygon file or file list must be given.")

        # Read the training grids
        region_df = read_training(
            region_id_file,
            columns=[DataColumns.GEOID, DataColumns.YEAR, "geometry"],
        )

        # Read the training polygons
        polygon_df = read_training(
            polygon_file,
            columns=[args.crop_column, "geometry"],
        )
        polygon_df[args.crop_column]
        polygon_df = polygon_df.astype({args.crop_column: int})

        assert (
            region_df.crs == polygon_df.crs
        ), "The region id CRS does not match the polygon CRS."

        assert (
            DataColumns.GEOID in region_df.columns
        ), "The geo_id column was not found in the grid region file."

        assert (
            DataColumns.YEAR in region_df.columns
        ), "The year column was not found in the grid region file."

        if 0 in polygon_df[args.crop_column].unique():
            raise ValueError("The field crop values should not have zeros.")

    # Get processed ids
    if hasattr(args, 'time_series_path') and (
        args.time_series_path is not None
    ):
        processed_ids = [Path(args.time_series_path)]
    else:
        if 'data_pattern' in config:
            processed_ids = list(
                ppaths.image_path.resolve().glob(config['data_pattern'])
            )
        else:
            processed_ids = list(ppaths.image_path.resolve().glob('*'))

    if args.destination == "train":
        # Filter ids to those that have been processed
        processed_mask = np.isin(
            np.array([fn.name for fn in processed_ids]),
            region_df[DataColumns.GEOID].values,
        )
        processed_ids = np.array(processed_ids)[processed_mask]

    partial_create_one_id = partial(
        create_one_id,
        args=args,
        config=config,
        ppaths=ppaths,
        region_df=region_df,
        polygon_df=polygon_df,
        bbox_offsets=args.bbox_offsets
        if args.destination == "train"
        else None,
    )

    if args.destination == "predict":
        partial_create_one_id(processed_path=processed_ids[0])
    else:
        with parallel_config(
            backend="loky",
            n_jobs=args.num_workers,
        ):
            with ParallelProgress(
                tqdm_params={
                    "total": len(processed_ids),
                    "desc": f"Creating {args.destination} files",
                    "colour": "green",
                    "ascii": "\u2015\u25E4\u25E5\u25E2\u25E3\u25AA",
                },
            ) as parallel_pool:
                parallel_pool(
                    delayed(partial_create_one_id)(
                        processed_path=processed_path
                    )
                    for processed_path in processed_ids
                )


def spatial_kfoldcv(args):
    ppaths = setup_paths(args.project_path)

    with open(ppaths.classes_info_path, mode="r") as f:
        class_info = json.load(f)

    ds = EdgeDataset(
        root=ppaths.train_path,
        log_transform=args.log_transform,
        processes=args.processes,
        random_seed=args.random_seed,
    )
    # Read or create the spatial partitions (folds)
    ds.get_spatial_partitions(
        spatial_partitions=args.spatial_partitions, splits=args.splits
    )
    for k, (partition_name, train_ds, test_ds) in enumerate(
        ds.spatial_kfoldcv_iter(args.partition_column)
    ):
        logger.info(
            f"Fold {k} of {len(ds.spatial_partitions.index)}, partition {partition_name} ..."
        )
        # Normalize the partition
        temp_ds = train_ds.split_train_val(val_frac=args.val_frac)[0]
        norm_values = NormValues.from_dataset(
            dataset=temp_ds,
            class_info=class_info,
            batch_size=args.batch_size,
        )
        train_ds.norm_values = norm_values
        test_ds.norm_values = norm_values

        # Get balanced class weights
        # Reference: https://github.com/scikit-learn/scikit-learn/blob/f3f51f9b6/sklearn/utils/class_weight.py#L10
        # recip_freq = data_values.crop_counts[1:].sum() / ((len(data_values.crop_counts)-1) * data_values.crop_counts[1:])
        # class_weights = recip_freq[torch.arange(0, len(data_values.crop_counts)-1)]
        # class_weights = torch.tensor([0] + list(class_weights), dtype=torch.float)
        if torch.cuda.is_available():
            class_counts = norm_values.crop_counts.to("cuda")
        else:
            class_counts = norm_values.crop_counts

        # Fit the model
        cultionet.fit(
            dataset=train_ds,
            ckpt_file=ppaths.ckpt_file,
            test_dataset=test_ds,
            val_frac=args.val_frac,
            batch_size=args.batch_size,
            load_batch_workers=args.load_batch_workers,
            epochs=args.epochs,
            save_top_k=args.save_top_k,
            accumulate_grad_batches=args.accumulate_grad_batches,
            optimizer=args.optimizer,
            learning_rate=args.learning_rate,
            hidden_channels=args.hidden_channels,
            num_classes=args.num_classes
            if args.num_classes is not None
            else class_info["max_crop_class"] + 1,
            edge_class=args.edge_class
            if args.edge_class is not None
            else class_info["edge_class"],
            class_counts=class_counts,
            reset_model=True,
            auto_lr_find=False,
            device=args.device,
            devices=args.devices,
            gradient_clip_val=args.gradient_clip_val,
            gradient_clip_algorithm=args.gradient_clip_algorithm,
            early_stopping_patience=args.patience,
            weight_decay=args.weight_decay,
            precision=args.precision,
            stochastic_weight_averaging=args.stochastic_weight_averaging,
            model_pruning=args.model_pruning,
        )
        # Rename the test metric JSON file
        (ppaths.ckpt_path / "test.metrics").rename(
            ppaths.ckpt_path
            / f"fold-{k}-{partition_name.replace(' ', '_')}.metrics"
        )


def train_model(args):
    seed_everything(args.random_seed, workers=True)

    # This is a helper function to manage paths
    ppaths = setup_paths(args.project_path)

    with open(ppaths.classes_info_path, mode="r") as f:
        class_info = json.load(f)

    if (
        (args.expected_time is not None)
        or not ppaths.norm_file.is_file()
        or (ppaths.norm_file.is_file() and args.recalc_zscores)
    ):
        ds = EdgeDataset(
            root=ppaths.train_path,
            log_transform=args.log_transform,
            pattern=args.data_pattern,
            processes=args.processes,
            random_seed=args.random_seed,
        )

    # Check dimensions
    if args.expected_time is not None:
        try:
            ds.check_dims(
                args.expected_time,
                args.expected_height,
                args.expected_width,
                args.delete_mismatches,
                args.dim_color,
            )
        except TensorShapeError as e:
            raise ValueError(e)

        ds = EdgeDataset(
            root=ppaths.train_path,
            log_transform=args.log_transform,
            pattern=args.data_pattern,
            processes=args.processes,
            random_seed=args.random_seed,
        )

    # Get the normalization means and std. deviations on the train data
    # Calculate the values needed to transform to z-scores, using
    # the training data
    if ppaths.norm_file.exists():
        if args.recalc_zscores:
            ppaths.norm_file.unlink()

    if not ppaths.norm_file.exists():
        if ds.grid_gpkg_path.exists():
            ds.grid_gpkg_path.unlink()

        if args.spatial_partitions is not None:
            train_ds = ds.split_train_val(
                val_frac=args.val_frac,
                spatial_overlap_allowed=False,
                spatial_balance=True,
            )[0]
        else:
            train_ds = ds.split_train_val(val_frac=args.val_frac)[0]

        # Get means and standard deviations from the training dataset
        norm_values: NormValues = NormValues.from_dataset(
            dataset=train_ds,
            class_info=class_info,
            batch_size=args.batch_size * 4,
            num_workers=args.load_batch_workers,
        )

        norm_values.to_file(ppaths.norm_file)
    else:
        norm_values = NormValues.from_file(ppaths.norm_file)

    # Create the train data object again, this time passing
    # the means and standard deviation tensors
    ds = EdgeDataset(
        root=ppaths.train_path,
        log_transform=args.log_transform,
        pattern=args.data_pattern,
        norm_values=norm_values,
        augment_prob=args.augment_prob,
        random_seed=args.random_seed,
    )

    # Check for a test dataset
    test_ds = None
    if list((ppaths.test_process_path).glob("*.pt")):
        test_ds = EdgeDataset(
            root=ppaths.test_path,
            log_transform=args.log_transform,
            norm_values=norm_values,
            random_seed=args.random_seed,
        )
        if args.expected_time is not None:
            try:
                test_ds.check_dims(
                    args.expected_time, args.delete_mismatches, args.dim_color
                )
            except TensorShapeError as e:
                raise ValueError(e)

            test_ds = EdgeDataset(
                root=ppaths.test_path,
                log_transform=args.log_transform,
                norm_values=norm_values,
                random_seed=args.random_seed,
            )

    # Combine edge counts with crop counts
    class_counts = torch.zeros(len(norm_values.dataset_crop_counts) + 1)
    class_counts[1:-1] = norm_values.dataset_crop_counts[1:]
    class_counts[-1] = norm_values.dataset_edge_counts[1]
    class_counts[0] = (
        norm_values.dataset_edge_counts[0]
        - norm_values.dataset_crop_counts[1:].sum()
    )

    if torch.cuda.is_available():
        class_counts = class_counts.to(device="cuda")

    cultionet_params = CultionetParams(
        ckpt_file=ppaths.ckpt_file,
        model_name="cultionet_transfer"
        if args.process == CLISteps.TRAIN_TRANSFER
        else "cultionet",
        dataset=ds,
        test_dataset=test_ds,
        val_frac=args.val_frac,
        spatial_partitions=args.spatial_partitions,
        batch_size=args.batch_size,
        load_batch_workers=args.load_batch_workers,
        edge_class=args.edge_class
        if args.edge_class is not None
        else class_info["edge_class"],
        class_counts=class_counts,
        hidden_channels=args.hidden_channels,
        model_type=args.model_type,
        activation_type=args.activation_type,
        dropout=args.dropout,
        dilations=args.dilations,
        res_block_type=args.res_block_type,
        attention_weights=args.attention_weights,
        optimizer=args.optimizer,
        loss_name=args.loss_name,
        learning_rate=args.learning_rate,
        lr_scheduler=args.lr_scheduler,
        steplr_step_size=args.steplr_step_size,
        weight_decay=args.weight_decay,
        pool_by_max=args.pool_by_max,
        batchnorm_first=args.batchnorm_first,
        scale_pos_weight=args.scale_pos_weight,
        save_batch_val_metrics=args.save_batch_val_metrics,
        epochs=args.epochs,
        accumulate_grad_batches=args.accumulate_grad_batches,
        gradient_clip_val=args.gradient_clip_val,
        gradient_clip_algorithm=args.gradient_clip_algorithm,
        precision=args.precision,
        device=args.device,
        devices=args.devices,
        reset_model=args.reset_model,
        auto_lr_find=args.auto_lr_find,
        stochastic_weight_averaging=args.stochastic_weight_averaging,
        stochastic_weight_averaging_lr=args.stochastic_weight_averaging_lr,
        stochastic_weight_averaging_start=args.stochastic_weight_averaging_start,
        skip_train=args.skip_train,
        finetune=args.finetune,
        strategy=args.strategy,
        profiler=args.profiler,
    )

    # Fit the model
    if args.process == CLISteps.TRAIN_TRANSFER:
        cultionet.fit_transfer(cultionet_params)
    else:
        cultionet.fit(cultionet_params)


def main():
    args_config = open_config((Path(__file__).parent / "args.yml").absolute())

    RichHelpFormatter.styles["argparse.groups"] = "#ACFCD6"
    RichHelpFormatter.styles["argparse.args"] = "#FCADED"
    RichHelpFormatter.styles["argparse.prog"] = "#AA9439"
    RichHelpFormatter.styles["argparse.help"] = "#cacaca"

    description = "# Cultionet: deep learning network for agricultural field boundary detection"

    epilog = """
# Examples
---

## Create training data
```commandline
cultionet create --project-path /projects/data -gs 100 100 -r 10.0 --max-crop-class 1 --crop-column crop_col --num-workers 8 --config-file config.yml
```

## View training help
```commandline
cultionet train --help
```

## Train a model
```commandline
cultionet train -p . --val-frac 0.1 --epochs 100 --processes 8 --load-batch-workers 8 --batch-size 4 --accumulate-grad-batches 4 --deep-sup
```

## Apply inference over an image
```commandline
cultionet predict --project-path /projects/data -o estimates.tif --region imageid --ref-image time_series_vars/imageid/brdf_ts/ms/evi2/20200101.tif --batch-size 4 --load-batch-workers 8 --start-date 2020-01-01 --end-date 2021-01-01 --config-file config.yml
```
    """

    parser = argparse.ArgumentParser(
        description=Markdown(description, style="argparse.text"),
        formatter_class=RichHelpFormatter,
        epilog=Markdown(epilog, style="argparse.text"),
    )

    subparsers = parser.add_subparsers(dest="process")
    available_processes = [
        CLISteps.CREATE,
        CLISteps.CREATE_PREDICT,
        CLISteps.SKFOLDCV,
        CLISteps.TRAIN,
        CLISteps.PREDICT,
        CLISteps.TRAIN_TRANSFER,
        CLISteps.PREDICT_TRANSFER,
        CLISteps.VERSION,
    ]
    for process in available_processes:
        subparser = subparsers.add_parser(
            process, formatter_class=parser.formatter_class
        )

        if process == CLISteps.VERSION:
            continue

        subparser.add_argument(
            "-p",
            "--project-path",
            dest="project_path",
            help="The project path (the directory that contains the grid ids)",
        )

        process_dict = args_config[process.replace("-", "_")]
        # Processes that use train args in addition to 'train'
        if process in (CLISteps.SKFOLDCV, CLISteps.TRAIN_TRANSFER):
            process_dict.update(args_config["train"])
        # Processes that use the predict args in addition to 'predict'
        if process in (CLISteps.PREDICT_TRANSFER,):
            process_dict.update(args_config["predict"])
        # Processes that use args shared between train and predict
        if process in (
            CLISteps.TRAIN,
            CLISteps.TRAIN_TRANSFER,
            CLISteps.PREDICT,
            CLISteps.PREDICT_TRANSFER,
            CLISteps.SKFOLDCV,
        ):
            process_dict.update(args_config["train_predict"])
            process_dict.update(args_config["shared_partitions"])
        if process in (CLISteps.CREATE, CLISteps.CREATE_PREDICT):
            process_dict.update(args_config["shared_create"])
        if process in (
            CLISteps.CREATE,
            CLISteps.CREATE_PREDICT,
            CLISteps.PREDICT,
            CLISteps.PREDICT_TRANSFER,
        ):
            process_dict.update(args_config["shared_image"])
        process_dict.update(args_config["dates"])
        for process_key, process_values in process_dict.items():
            if "kwargs" in process_values:
                kwargs = process_values["kwargs"]
                for key, value in kwargs.items():
                    if isinstance(value, str) and value.startswith("&"):
                        kwargs[key] = getattr(builtins, value.replace("&", ""))

            else:
                process_values["kwargs"] = {}
            key_args = ()
            if len(process_values["short"]) > 0:
                key_args += (f"-{process_values['short']}",)
            if len(process_values["long"]) > 0:
                key_args += (f"--{process_values['long']}",)
            subparser.add_argument(
                *key_args,
                dest=process_key,
                help=f"{process_values['help']} (default: %(default)s)",
                **process_values["kwargs"],
            )

        # if process in (
        #     CLISteps.CREATE,
        #     CLISteps.CREATE_PREDICT,
        #     CLISteps.PREDICT,
        #     CLISteps.PREDICT_TRANSFER,
        # ):
        subparser.add_argument(
            "--config-file",
            dest="config_file",
            help="The configuration YAML file (default: %(default)s)",
            default=(Path(__file__).parent / "config.yml").absolute(),
        )

    args = parser.parse_args()

    if hasattr(args, "config_file") and (args.config_file is not None):
        args.config_file = str(args.config_file)

    if args.process == CLISteps.CREATE_PREDICT:
        setattr(args, "destination", "predict")

    if args.process == CLISteps.VERSION:
        print(cultionet.__version__)
        return

    if hasattr(args, "replace_dict"):
        if args.replace_dict is not None:
            replace_dict = dict(
                list(
                    map(
                        lambda x: list(map(int, x.split(":"))),
                        args.replace_dict.split(" "),
                    )
                )
            )
            setattr(args, "replace_dict", replace_dict)

    # config = open_config(args.config_file)
    # for k, v in config["train"].get("trainer").items():
    #     setattr(args, k, v)
    # for k, v in config["train"].get("model").items():
    #     setattr(args, k, v)

    project_path = Path(args.project_path) / "ckpt"
    project_path.mkdir(parents=True, exist_ok=True)
    command_path = Path(args.project_path) / "commands"
    command_path.mkdir(parents=True, exist_ok=True)
    now = datetime.now()

    with open(
        command_path
        / f"{args.process}_command_{now.strftime('%Y%m%d-%H%M')}.json",
        mode="w",
    ) as f:
        f.write(json.dumps(vars(args), indent=4))

    if args.process in (
        CLISteps.CREATE,
        CLISteps.CREATE_PREDICT,
    ):
        create_dataset(args)
    elif args.process == CLISteps.SKFOLDCV:
        spatial_kfoldcv(args)
    elif args.process in (
        CLISteps.TRAIN,
        CLISteps.TRAIN_TRANSFER,
    ):
        train_model(args)
    elif args.process in (
        CLISteps.PREDICT,
        CLISteps.PREDICT_TRANSFER,
    ):
        predict_image(args)


if __name__ == "__main__":
    main()
