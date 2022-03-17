import typing as T
from pathlib import Path

import geopandas as gpd
import rasterio as rio
from shapely.geometry import Polygon


def bounds_to_frame(
    left: float, bottom: float, right: float, top: float, crs: T.Optional[str] = 'epsg:4326'
) -> gpd.GeoDataFrame:
    """Converts a bounding box to a GeoDataFrame
    """
    geom = Polygon([(left, bottom), (left, top), (right, top), (right, bottom), (left, bottom)])
    df = gpd.GeoDataFrame(data=[0], geometry=[geom], crs=crs)

    return df


def warp_by_image(
    df: gpd.GeoDataFrame, image_path: T.Union[str, Path]
) -> T.Tuple[gpd.GeoDataFrame, str]:
    """Warps a GeoDataFrame CRS by a reference image
    """
    with rio.open(image_path) as src:
        df = df.to_crs(src.crs.to_epsg())
    ref_crs = f'epsg:{df.crs.to_epsg()}'

    return df, ref_crs
