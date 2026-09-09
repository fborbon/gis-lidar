"""Sentinel-2 L2A scene search and windowed reads from the public Earth Search STAC API.

No AWS credentials or Copernicus account needed: the Element84 Earth Search catalog
indexes the Sentinel-2 Cloud-Optimized GeoTIFFs (COGs) that AWS hosts publicly, and
rasterio can read a small AOI window directly out of a COG over HTTPS without ever
downloading the full ~110MB-per-band scene.
"""
from __future__ import annotations

import json
from pathlib import Path

import requests

STAC_URL = "https://earth-search.aws.element84.com/v1/search"


def find_best_scene(bbox: list[float], datetime_range: str, limit: int = 100) -> dict:
    """Return the lowest-cloud-cover Sentinel-2 L2A STAC item covering bbox."""
    body = {"collections": ["sentinel-2-l2a"], "bbox": bbox, "datetime": datetime_range, "limit": limit}
    r = requests.post(STAC_URL, json=body, timeout=30)
    r.raise_for_status()
    feats = r.json()["features"]
    if not feats:
        raise RuntimeError(f"No Sentinel-2 scenes found for bbox={bbox}")
    feats.sort(key=lambda f: f["properties"].get("eo:cloud_cover", 100))
    return feats[0]


def save_scene_metadata(item: dict, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as fh:
        json.dump(item, fh, indent=2)


if __name__ == "__main__":
    bbox = [-77.012, 38.8765, -76.9943, 38.8904]
    item = find_best_scene(bbox, "2024-01-01T00:00:00Z/2026-09-09T00:00:00Z")
    print(item["id"], item["properties"]["eo:cloud_cover"])
    save_scene_metadata(item, Path("data/raw/sentinel2/scene_metadata.json"))
