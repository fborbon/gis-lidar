"""Pixel-level and polygon-level (GIS-aware) evaluation metrics.

Pixel accuracy alone is a weak metric for geospatial ML: a model can nail pixel
accuracy while still missing every small building or splitting one roof into
several detections. The polygon-level metrics here (building-level detection
rate, matched-polygon IoU, area error, centroid displacement) are what a GIS
reviewer actually cares about, and are the differentiator this project is built
to demonstrate.
"""
from __future__ import annotations

import numpy as np
from shapely.strtree import STRtree


def pixel_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    y_true = y_true.astype(bool)
    y_pred = y_pred.astype(bool)
    tp = int(np.sum(y_true & y_pred))
    fp = int(np.sum(~y_true & y_pred))
    fn = int(np.sum(y_true & ~y_pred))
    tn = int(np.sum(~y_true & ~y_pred))
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    iou = tp / (tp + fp + fn) if (tp + fp + fn) else 0.0
    accuracy = (tp + tn) / (tp + fp + fn + tn) if (tp + fp + fn + tn) else 0.0
    return {"precision": precision, "recall": recall, "f1": f1, "iou": iou, "accuracy": accuracy, "tp": tp, "fp": fp, "fn": fn, "tn": tn}


def polygon_metrics(pred_polygons: list, gt_polygons: list, match_iou_threshold: float = 0.3) -> dict:
    """Match predicted building polygons to ground-truth OSM polygons by best IoU
    and report building-level detection rate, mean matched IoU, mean area error
    (%), and mean centroid displacement (m).
    """
    if not gt_polygons:
        return {"n_gt": 0, "n_pred": len(pred_polygons), "detection_rate": 0.0, "mean_matched_iou": 0.0, "mean_area_error_pct": 0.0, "mean_centroid_disp_m": 0.0, "n_matched": 0}

    pred_tree = STRtree(pred_polygons) if pred_polygons else None

    matched = 0
    ious, area_errs, centroid_disps = [], [], []
    for gt in gt_polygons:
        best_iou, best_pred = 0.0, None
        if pred_tree is not None:
            candidate_idx = pred_tree.query(gt)
            for idx in np.atleast_1d(candidate_idx):
                pred = pred_polygons[int(idx)]
                inter = gt.intersection(pred).area
                union = gt.union(pred).area
                iou = inter / union if union else 0.0
                if iou > best_iou:
                    best_iou, best_pred = iou, pred
        if best_iou >= match_iou_threshold:
            matched += 1
            ious.append(best_iou)
            area_errs.append(abs(best_pred.area - gt.area) / gt.area * 100 if gt.area else 0.0)
            centroid_disps.append(gt.centroid.distance(best_pred.centroid))

    return {
        "n_gt": len(gt_polygons),
        "n_pred": len(pred_polygons),
        "n_matched": matched,
        "detection_rate": matched / len(gt_polygons),
        "mean_matched_iou": float(np.mean(ious)) if ious else 0.0,
        "mean_area_error_pct": float(np.mean(area_errs)) if area_errs else 0.0,
        "mean_centroid_disp_m": float(np.mean(centroid_disps)) if centroid_disps else 0.0,
    }
