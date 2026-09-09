"""Shared analysis grid definition. Everything (nDSM, NDVI, OSM mask, predictions)
gets resampled onto this one grid so per-pixel feature stacks line up exactly.
"""
from __future__ import annotations

import numpy as np
from pyproj import Transformer
from rasterio.transform import from_origin

CRS = "EPSG:32618"  # UTM zone 18N - matches the native CRS of both the LiDAR tile and Sentinel-2 tile 18SUJ
RESOLUTION_M = 2.0

# Analysis AOI in WGS84 (lon/lat), a ~700m x 650m sub-window of the LiDAR tile centered
# on a dense Capitol Hill rowhouse block in Washington, DC - chosen for building density
# and OSM completeness, comfortably inside the LAZ tile's coverage to avoid edge effects.
AOI_WGS84 = (-77.005, 38.882, -76.998, 38.888)  # (minx, miny, maxx, maxy)


def aoi_bounds_utm() -> tuple[float, float, float, float]:
    tf = Transformer.from_crs("EPSG:4326", CRS, always_xy=True)
    x0, y0 = tf.transform(AOI_WGS84[0], AOI_WGS84[1])
    x1, y1 = tf.transform(AOI_WGS84[2], AOI_WGS84[3])
    return min(x0, x1), min(y0, y1), max(x0, x1), max(y0, y1)


def grid_shape_transform():
    minx, miny, maxx, maxy = aoi_bounds_utm()
    width = int(np.ceil((maxx - minx) / RESOLUTION_M))
    height = int(np.ceil((maxy - miny) / RESOLUTION_M))
    transform = from_origin(minx, maxy, RESOLUTION_M, RESOLUTION_M)
    return height, width, transform, (minx, miny, maxx, maxy)
