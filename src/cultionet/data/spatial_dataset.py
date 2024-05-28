from pathlib import Path
from typing import Optional, Tuple

import geopandas as gpd
import pygrts
from joblib import delayed, parallel_backend
from shapely.geometry import box
from torch.utils.data import Dataset

from ..utils.model_preprocessing import TqdmParallel


def get_box_id(data_id: str, *bounds) -> tuple:
    return data_id, box(*list(map(float, bounds))).centroid


class SpatialDataset(Dataset):
    dataset_df = None

    @property
    def grid_gpkg_path(self) -> Path:
        return self.root / "dataset_grids.gpkg"

    def create_spatial_index(self, id_column: str, n_jobs: int):
        """Creates the spatial index."""

        if self.grid_gpkg_path.exists():
            self.dataset_df = gpd.read_file(self.grid_gpkg_path)
        else:
            self.dataset_df = self.to_frame(id_column=id_column, n_jobs=n_jobs)
            self.dataset_df.to_file(self.grid_gpkg_path, driver="GPKG")

    def to_frame(self, id_column: str, n_jobs: int) -> gpd.GeoDataFrame:
        """Converts the Dataset to a GeoDataFrame."""

        with parallel_backend(backend="loky", n_jobs=n_jobs):
            with TqdmParallel(
                tqdm_kwargs={
                    "total": len(self),
                    "desc": "Building GeoDataFrame",
                    "ascii": "\u2015\u25E4\u25E5\u25E2\u25E3\u25AA",
                    "colour": "green",
                }
            ) as pool:
                results = pool(
                    delayed(get_box_id)(
                        data.batch_id,
                        data.left,
                        data.bottom,
                        data.right,
                        data.top,
                    )
                    for data in self
                )

        ids, geometry = list(map(list, zip(*results)))
        df = gpd.GeoDataFrame(
            data=ids,
            columns=[id_column],
            geometry=geometry,
            crs="epsg:4326",
        )

        return df

    def spatial_splits(
        self,
        val_frac: float,
        id_column: str,
        spatial_balance: bool = True,
        crs: str = "EPSG:8857",
        random_state: Optional[int] = None,
    ) -> Tuple[Dataset, Dataset]:
        """Takes spatially-balanced splits of the dataset."""

        if spatial_balance:
            # Separate train and validation by spatial location

            # Setup a quad-tree using the GRTS method
            # (see https://github.com/jgrss/pygrts for details)
            qt = pygrts.QuadTree(
                self.dataset_df.to_crs(crs),
                force_square=False,
            )

            # Recursively split the quad-tree until each grid has
            # only one sample.
            qt.split_recursive(max_samples=1)

            n_val = int(val_frac * len(self.dataset_df.index))
            # `qt.sample` random samples from the quad-tree in a
            # spatially balanced manner. Thus, `df_val_sample` is
            # a GeoDataFrame with `n_val` sites spatially balanced.
            df_val_sample = qt.sample(n=n_val, random_state=random_state)

            # Since we only took one sample from each coordinate,
            # we need to find all of the .pt files that share
            # coordinates with the sampled sites.
            val_mask = self.dataset_df[id_column].isin(
                df_val_sample[id_column]
            )
        else:
            # Randomly sample a percentage for validation
            df_val_ids = self.dataset_df.sample(
                frac=val_frac, random_state=random_state
            ).to_frame(name=id_column)

            # Get all ids for validation samples
            val_mask = self.dataset_df[id_column].isin(df_val_ids[id_column])

        # Get train/val indices
        val_idx = self.dataset_df.loc[val_mask].index.values
        train_idx = self.dataset_df.loc[~val_mask].index.values

        # Slice the dataset
        train_ds = self[train_idx]
        val_ds = self[val_idx]

        return train_ds, val_ds
