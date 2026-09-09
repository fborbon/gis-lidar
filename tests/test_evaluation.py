import numpy as np
from shapely.geometry import box

from src.evaluation.metrics import pixel_metrics, polygon_metrics


def test_pixel_metrics_perfect_match():
    y = np.array([[1, 0], [0, 1]])
    m = pixel_metrics(y, y)
    assert m["precision"] == 1.0
    assert m["recall"] == 1.0
    assert m["f1"] == 1.0
    assert m["iou"] == 1.0


def test_pixel_metrics_no_overlap():
    y_true = np.array([[1, 0], [0, 0]])
    y_pred = np.array([[0, 0], [0, 1]])
    m = pixel_metrics(y_true, y_pred)
    assert m["iou"] == 0.0
    assert m["tp"] == 0


def test_polygon_metrics_exact_match():
    gt = [box(0, 0, 10, 10)]
    pred = [box(0, 0, 10, 10)]
    m = polygon_metrics(pred, gt)
    assert m["detection_rate"] == 1.0
    assert abs(m["mean_matched_iou"] - 1.0) < 1e-9
    assert m["mean_area_error_pct"] == 0.0
    assert m["mean_centroid_disp_m"] == 0.0


def test_polygon_metrics_missed_building():
    gt = [box(0, 0, 10, 10), box(100, 100, 110, 110)]
    pred = [box(0, 0, 10, 10)]
    m = polygon_metrics(pred, gt)
    assert m["n_gt"] == 2
    assert m["n_matched"] == 1
    assert m["detection_rate"] == 0.5


def test_polygon_metrics_no_ground_truth():
    m = polygon_metrics([box(0, 0, 5, 5)], [])
    assert m["n_gt"] == 0
    assert m["detection_rate"] == 0.0
