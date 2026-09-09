"""Turn the aligned raster stack (nDSM, DTM, NDVI) into a per-pixel feature table."""
from __future__ import annotations

import numpy as np
from scipy.ndimage import generic_filter, sobel


def terrain_slope(dtm: np.ndarray, cell_size_m: float) -> np.ndarray:
    """Slope in degrees from a DTM, via a simple Sobel gradient (standard GIS
    terrain-analysis approach, equivalent in spirit to a Horn 1981 slope operator).
    """
    dzdx = sobel(dtm, axis=1) / (8 * cell_size_m)
    dzdy = sobel(dtm, axis=0) / (8 * cell_size_m)
    slope_rad = np.arctan(np.sqrt(dzdx**2 + dzdy**2))
    return np.degrees(slope_rad)


def local_texture(ndsm: np.ndarray, window: int = 3) -> np.ndarray:
    """Local standard deviation of nDSM in a small window - a cheap texture feature.
    Building edges/roof structure produce higher local variance than flat ground
    or uniform tree canopy.
    """
    return generic_filter(ndsm, np.std, size=window)


def build_feature_stack(ndsm: np.ndarray, dtm: np.ndarray, ndvi: np.ndarray, cell_size_m: float) -> dict[str, np.ndarray]:
    return {
        "ndsm": ndsm,
        "ndvi": ndvi,
        "slope": terrain_slope(dtm, cell_size_m),
        "texture": local_texture(ndsm),
    }


def stack_to_table(features: dict[str, np.ndarray]) -> tuple[np.ndarray, list[str]]:
    names = list(features.keys())
    arrs = [features[n].ravel() for n in names]
    X = np.stack(arrs, axis=1)
    return X, names
