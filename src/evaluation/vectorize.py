"""Vectorize a raster building mask into polygons, for polygon-level evaluation."""
from __future__ import annotations

import numpy as np
from rasterio.features import shapes
from shapely.geometry import shape
from shapely.validation import make_valid


def mask_to_polygons(mask: np.ndarray, transform, min_area_m2: float = 8.0) -> list:
    polys = []
    for geom, _value in shapes(mask.astype(np.uint8), mask=mask.astype(bool), transform=transform):
        g = shape(geom)
        if not g.is_valid:
            g = make_valid(g)
        if g.area >= min_area_m2:
            polys.append(g)
    return polys
