"""Read the Sentinel-2 red/nir/blue windows for the AOI directly off the remote
COGs and resample them onto the shared analysis grid, then compute NDVI.
"""
from __future__ import annotations

import numpy as np
import rasterio
from rasterio.warp import Resampling, reproject

from src.preprocessing.grid import CRS, grid_shape_transform


def _read_band_on_grid(href: str) -> np.ndarray:
    height, width, transform, _ = grid_shape_transform()
    dst = np.zeros((height, width), dtype=np.float32)
    with rasterio.open(href) as src:
        reproject(
            source=rasterio.band(src, 1),
            destination=dst,
            src_transform=src.transform,
            src_crs=src.crs,
            dst_transform=transform,
            dst_crs=CRS,
            resampling=Resampling.bilinear,
        )
    return dst


def build_ndvi(red_href: str, nir_href: str) -> np.ndarray:
    red = _read_band_on_grid(red_href).astype(np.float32)
    nir = _read_band_on_grid(nir_href).astype(np.float32)
    denom = nir + red
    ndvi = np.where(denom == 0, 0.0, (nir - red) / denom)
    return ndvi
