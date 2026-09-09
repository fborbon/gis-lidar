"""Build a Digital Terrain Model (DTM, ground surface), Digital Surface Model
(DSM, first-return / top-of-canopy-and-roofs surface) and normalized DSM
(nDSM = DSM - DTM, i.e. height of buildings/vegetation above bare earth) from a
raw USGS 3DEP LAZ point cloud, gridded onto the shared analysis grid.

This is the classic LiDAR building/vegetation extraction technique: nDSM alone
already separates "stuff sitting on the ground" from "the ground", which is why
LiDAR is such a strong complement to optical imagery for GIS building extraction.
"""
from __future__ import annotations

import laspy
import numpy as np
from pyproj import CRS as PyprojCRS
from pyproj import Transformer
from scipy.interpolate import griddata
from scipy.ndimage import grey_dilation, grey_erosion

from src.preprocessing.grid import CRS as TARGET_CRS
from src.preprocessing.grid import grid_shape_transform


def _bin_points(x, y, z, height, width, transform, agg="max"):
    """Rasterize scattered points into a (height, width) grid by binning into
    cells and aggregating (max for DSM top surface, min for DTM ground surface).
    Empty cells are left as NaN and filled by the caller.
    """
    inv = ~transform
    cols, rows = inv * (x, y)
    cols = np.floor(cols).astype(int)
    rows = np.floor(rows).astype(int)
    valid = (cols >= 0) & (cols < width) & (rows >= 0) & (rows < height)
    cols, rows, z = cols[valid], rows[valid], z[valid]

    grid = np.full((height, width), np.nan, dtype=np.float64)
    flat_idx = rows * width + cols
    order = np.argsort(flat_idx)
    flat_idx, z = flat_idx[order], z[order]

    unique_idx, start = np.unique(flat_idx, return_index=True)
    if agg == "max":
        agg_vals = np.maximum.reduceat(z, start)
    elif agg == "min":
        agg_vals = np.minimum.reduceat(z, start)
    else:
        raise ValueError(agg)
    grid.ravel()[unique_idx] = agg_vals
    return grid


def _fill_nan(grid: np.ndarray) -> np.ndarray:
    """Fill empty raster cells (no LiDAR return landed in them) via nearest-neighbor
    interpolation from the surrounding filled cells - standard practice for sparse
    point-cloud rasterization at fine grid resolution.
    """
    mask = ~np.isnan(grid)
    if mask.all():
        return grid
    if not mask.any():
        return np.zeros_like(grid)
    yy, xx = np.mgrid[0 : grid.shape[0], 0 : grid.shape[1]]
    filled = griddata((yy[mask], xx[mask]), grid[mask], (yy, xx), method="nearest")
    return filled


def load_points_in_aoi(laz_path: str, aoi_bounds_utm: tuple[float, float, float, float]):
    """Read the LAZ file, reproject to the target UTM CRS if needed, and clip to the AOI."""
    with laspy.open(laz_path) as f:
        las = f.read()

    src_crs = None
    try:
        src_crs = las.header.parse_crs()
    except Exception:
        src_crs = None

    x, y, z = np.asarray(las.x), np.asarray(las.y), np.asarray(las.z)
    classification = np.asarray(las.classification)

    if src_crs is not None and PyprojCRS(src_crs) != PyprojCRS(TARGET_CRS):
        transformer = Transformer.from_crs(src_crs, TARGET_CRS, always_xy=True)
        x, y = transformer.transform(x, y)

    minx, miny, maxx, maxy = aoi_bounds_utm
    keep = (x >= minx) & (x <= maxx) & (y >= miny) & (y <= maxy)
    return x[keep], y[keep], z[keep], classification[keep], src_crs


def build_dtm_dsm(laz_path: str):
    height, width, transform, bounds = grid_shape_transform()
    x, y, z, classification, src_crs = load_points_in_aoi(laz_path, bounds)
    if len(x) == 0:
        raise RuntimeError("No LiDAR points fell inside the AOI - check CRS/bounds alignment")

    ground = classification == 2
    if ground.sum() < 50:
        raise RuntimeError(f"Only {ground.sum()} ground-classified points in AOI - not enough for a DTM")

    dtm = _bin_points(x[ground], y[ground], z[ground], height, width, transform, agg="min")
    dtm = _fill_nan(dtm)
    # Light closing to smooth single-cell LiDAR ground noise without erasing real relief.
    dtm = grey_erosion(grey_dilation(dtm, size=2), size=2)

    # ASPRS class 18 = "high noise" - drop it outright. This tile's vendor also
    # dumped a large share of points into class 17 ("bridge deck" per the ASPRS
    # spec) despite the AOI having no bridges, which only makes sense as a loose
    # or non-standard reuse of that class code by the original contractor; rather
    # than trust the label, a plain elevation sanity clip (-10m to 150m, well
    # above anything in this part of DC) catches the genuine outliers in both
    # class 17 and the unclassified points without discarding real rooftops.
    not_noise = classification != 18
    sane_z = (z > -10) & (z < 150)
    dsm_keep = not_noise & sane_z
    dsm = _bin_points(x[dsm_keep], y[dsm_keep], z[dsm_keep], height, width, transform, agg="max")
    dsm = _fill_nan(dsm)

    ndsm = np.clip(dsm - dtm, 0, None)

    stats = {
        "n_points_total": len(x),
        "n_points_ground": int(ground.sum()),
        "source_crs": str(src_crs) if src_crs else "unknown (assumed already " + TARGET_CRS + ")",
        "dtm_min_m": float(np.nanmin(dtm)),
        "dtm_max_m": float(np.nanmax(dtm)),
        "ndsm_max_m": float(np.nanmax(ndsm)),
    }
    return dtm, dsm, ndsm, transform, stats
