import numpy as np

from src.features.build_features import (
    build_feature_stack,
    stack_to_table,
    terrain_slope,
)
from src.models.building_classifier import ndsm_only_baseline, ndvi_only_baseline


def test_terrain_slope_flat_surface_is_zero():
    dtm = np.full((10, 10), 50.0)
    slope = terrain_slope(dtm, cell_size_m=2.0)
    assert np.allclose(slope, 0.0, atol=1e-6)


def test_terrain_slope_ramp_is_positive():
    dtm = np.tile(np.arange(10, dtype=float), (10, 1))  # 1 unit rise per column
    slope = terrain_slope(dtm, cell_size_m=1.0)
    assert slope[5, 5] > 0


def test_build_feature_stack_shapes_match():
    ndsm = np.random.rand(8, 8) * 10
    dtm = np.random.rand(8, 8) * 50
    ndvi = np.random.rand(8, 8) * 2 - 1
    feats = build_feature_stack(ndsm, dtm, ndvi, cell_size_m=2.0)
    assert set(feats.keys()) == {"ndsm", "ndvi", "slope", "texture"}
    for arr in feats.values():
        assert arr.shape == (8, 8)


def test_stack_to_table_flattens_correctly():
    feats = {"a": np.array([[1, 2], [3, 4]]), "b": np.array([[5, 6], [7, 8]])}
    X, names = stack_to_table(feats)
    assert names == ["a", "b"]
    assert X.shape == (4, 2)
    assert list(X[:, 0]) == [1, 2, 3, 4]


def test_ndvi_only_baseline_thresholds_correctly():
    ndvi = np.array([[-0.1, 0.5], [0.1, 0.3]])
    pred = ndvi_only_baseline(ndvi, threshold=0.2)
    assert pred.tolist() == [[1, 0], [1, 0]]


def test_ndsm_only_baseline_thresholds_correctly():
    ndsm = np.array([[0.0, 5.0], [3.0, 1.0]])
    pred = ndsm_only_baseline(ndsm, threshold_m=2.5)
    assert pred.tolist() == [[0, 1], [1, 0]]
