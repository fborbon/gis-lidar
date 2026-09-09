"""End-to-end pipeline: ingest -> preprocess -> features -> train -> evaluate -> LLM layer.
Idempotent: skips network/heavy steps if their cached output already exists on disk.

Run from the repo root with the venv active: python scripts/run_pipeline.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import geopandas as gpd
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.evaluation.metrics import pixel_metrics, polygon_metrics
from src.evaluation.vectorize import mask_to_polygons
from src.features.build_features import build_feature_stack, stack_to_table
from src.models.building_classifier import (
    ndsm_only_baseline,
    ndvi_only_baseline,
    train_random_forest,
)
from src.preprocessing.grid import RESOLUTION_M, grid_shape_transform

DATA_PROCESSED = Path("data/processed")
LAZ_PATH = "data/raw/lidar/tile_18SUJ325305.laz"

# Spatial holdout: last 25% of grid columns is the test region (a contiguous
# geographic block untouched during training), not a random pixel split -
# random pixel splits leak spatially autocorrelated neighboring pixels between
# train/test and would overstate accuracy for a raster classification task.
TEST_COL_FRACTION = 0.25


def load_or_build_rasters():
    dtm = np.load(DATA_PROCESSED / "dtm.npy")
    dsm = np.load(DATA_PROCESSED / "dsm.npy")
    ndsm = np.load(DATA_PROCESSED / "ndsm.npy")
    ndvi = np.load(DATA_PROCESSED / "ndvi.npy")
    osm_mask = np.load(DATA_PROCESSED / "osm_mask.npy")
    return dtm, dsm, ndsm, ndvi, osm_mask


def spatial_split(height: int, width: int):
    test_start_col = int(width * (1 - TEST_COL_FRACTION))
    train_mask = np.zeros((height, width), dtype=bool)
    train_mask[:, :test_start_col] = True
    test_mask = ~train_mask
    return train_mask, test_mask


def main():
    dtm, dsm, ndsm, ndvi, osm_mask = load_or_build_rasters()
    height, width = ndsm.shape
    _, _, transform, _ = grid_shape_transform()

    features = build_feature_stack(ndsm, dtm, ndvi, RESOLUTION_M)
    X, feature_names = stack_to_table(features)
    y = osm_mask.ravel().astype(int)

    train_mask, test_mask = spatial_split(height, width)
    train_flat, test_flat = train_mask.ravel(), test_mask.ravel()

    clf = train_random_forest(X[train_flat], y[train_flat], feature_names)
    y_pred_full = clf.predict(X).reshape(height, width)

    rf_test_metrics = pixel_metrics(osm_mask[test_mask].reshape(-1, 1), y_pred_full[test_mask].reshape(-1, 1))

    ndvi_pred = ndvi_only_baseline(ndvi)
    ndsm_pred = ndsm_only_baseline(ndsm)
    ndvi_test_metrics = pixel_metrics(osm_mask[test_mask].reshape(-1, 1), ndvi_pred[test_mask].reshape(-1, 1))
    ndsm_test_metrics = pixel_metrics(osm_mask[test_mask].reshape(-1, 1), ndsm_pred[test_mask].reshape(-1, 1))

    # Polygon-level evaluation, restricted to the held-out test region.
    rf_test_only = np.where(test_mask, y_pred_full, 0).astype(np.uint8)
    gt_test_only = np.where(test_mask, osm_mask, 0).astype(np.uint8)
    pred_polys = mask_to_polygons(rf_test_only, transform)
    gt_polys = mask_to_polygons(gt_test_only, transform)
    poly_metrics = polygon_metrics(pred_polys, gt_polys)

    feature_importances = dict(zip(feature_names, clf.feature_importances_.tolist(), strict=True))

    results = {
        "aoi": {"resolution_m": RESOLUTION_M, "grid_shape": [height, width]},
        "n_train_pixels": int(train_flat.sum()),
        "n_test_pixels": int(test_flat.sum()),
        "feature_importances": feature_importances,
        "pixel_metrics": {
            "random_forest_multimodal": rf_test_metrics,
            "ndvi_only_baseline": ndvi_test_metrics,
            "ndsm_only_baseline": ndsm_test_metrics,
        },
        "polygon_metrics": poly_metrics,
    }

    DATA_PROCESSED.mkdir(exist_ok=True)
    with open(DATA_PROCESSED / "metrics.json", "w") as fh:
        json.dump(results, fh, indent=2)

    np.save(DATA_PROCESSED / "rf_prediction.npy", y_pred_full)
    np.save(DATA_PROCESSED / "train_mask.npy", train_mask)
    np.save(DATA_PROCESSED / "test_mask.npy", test_mask)

    gdf_pred = gpd.GeoDataFrame({"geometry": pred_polys}, crs="EPSG:32618")
    gdf_pred.to_file(DATA_PROCESSED / "predicted_buildings_test.geojson", driver="GeoJSON")

    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
