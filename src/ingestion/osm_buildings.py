"""Fetch OSM building footprints for an AOI via the public Overpass API and turn
the raw way/node JSON into building polygons. No API key or account needed.
"""
from __future__ import annotations

import json
from pathlib import Path

import geopandas as gpd
import requests
from shapely.geometry import Polygon

OVERPASS_URL = "https://overpass-api.de/api/interpreter"


def fetch_buildings_json(bbox_latlon: tuple[float, float, float, float], out_path: Path) -> dict:
    """bbox_latlon = (south, west, north, east), Overpass bbox order."""
    south, west, north, east = bbox_latlon
    query = f"""
[out:json][timeout:60];
(
  way["building"]({south},{west},{north},{east});
  relation["building"]({south},{west},{north},{east});
);
out body;
>;
out skel qt;
""".strip()
    headers = {"User-Agent": "gis-lidar-demo/1.0 (fborbon portfolio project)"}
    r = requests.post(OVERPASS_URL, data={"data": query}, headers=headers, timeout=90)
    r.raise_for_status()
    data = r.json()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as fh:
        json.dump(data, fh)
    return data


def osm_json_to_geodataframe(raw: dict) -> gpd.GeoDataFrame:
    """Convert raw Overpass way+node elements into a building-footprint GeoDataFrame (EPSG:4326)."""
    nodes = {e["id"]: (e["lon"], e["lat"]) for e in raw["elements"] if e["type"] == "node"}
    rows = []
    for el in raw["elements"]:
        if el["type"] != "way" or "tags" not in el or "building" not in el.get("tags", {}):
            continue
        coords = [nodes[n] for n in el["nodes"] if n in nodes]
        if len(coords) < 4 or coords[0] != coords[-1]:
            continue
        try:
            poly = Polygon(coords)
        except Exception:
            continue
        if not poly.is_valid or poly.area == 0:
            continue
        rows.append({"osm_id": el["id"], "building_type": el["tags"].get("building"), "geometry": poly})
    return gpd.GeoDataFrame(rows, geometry="geometry", crs="EPSG:4326")


if __name__ == "__main__":
    bbox = (38.8765, -77.012, 38.8904, -76.9943)  # south, west, north, east
    out_json = Path("data/raw/osm/dc_capitol_hill_buildings.json")
    if out_json.exists():
        with open(out_json) as fh:
            raw = json.load(fh)
    else:
        raw = fetch_buildings_json(bbox, out_json)
    gdf = osm_json_to_geodataframe(raw)
    print(f"{len(gdf)} building polygons")
    gdf.to_file("data/processed/osm_buildings.geojson", driver="GeoJSON")
