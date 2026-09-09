"""Rasterize OSM building polygons onto the shared analysis grid, producing the
ground-truth building mask that both the classifier and the evaluation metrics
are checked against.
"""
from __future__ import annotations

import geopandas as gpd
import numpy as np
from rasterio.features import rasterize

from src.preprocessing.grid import CRS, grid_shape_transform


def buildings_to_mask(buildings_gdf: gpd.GeoDataFrame) -> np.ndarray:
    height, width, transform, bounds = grid_shape_transform()
    gdf_utm = buildings_gdf.to_crs(CRS)
    shapes = [(geom, 1) for geom in gdf_utm.geometry if geom is not None and not geom.is_empty]
    mask = rasterize(shapes, out_shape=(height, width), transform=transform, fill=0, dtype="uint8")
    return mask.astype(bool)
