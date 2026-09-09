"""RandomForest building/non-building classifier, plus two naive single-feature
baselines used to honestly show what multimodal fusion buys you over either
sensor alone.
"""
from __future__ import annotations

import numpy as np
from sklearn.ensemble import RandomForestClassifier


def train_random_forest(X_train: np.ndarray, y_train: np.ndarray, feature_names: list[str]) -> RandomForestClassifier:
    clf = RandomForestClassifier(
        n_estimators=300,
        max_depth=12,
        min_samples_leaf=5,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )
    clf.fit(X_train, y_train)
    return clf


def ndvi_only_baseline(ndvi: np.ndarray, threshold: float = 0.2) -> np.ndarray:
    """Naive optical-only baseline: low NDVI (not vegetation) => guess building.
    This is deliberately weak - low NDVI also covers roads, parking lots, bare
    soil, water - to make the point that spectral index alone cannot isolate
    buildings without a height cue.
    """
    return (ndvi < threshold).astype(np.uint8)


def ndsm_only_baseline(ndsm: np.ndarray, threshold_m: float = 2.5) -> np.ndarray:
    """Naive LiDAR-only baseline: anything raised more than threshold_m above
    bare earth is guessed as a building. This also catches trees, so precision
    suffers even though recall is often good.
    """
    return (ndsm > threshold_m).astype(np.uint8)
