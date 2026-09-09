"""Render static/index.html from the processed pipeline artifacts. Run after
scripts/run_pipeline.py and the LLM benchmark/explanation scripts.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

DATA = Path("data/processed")
STATIC_OUT = Path("static/index.html")

def _load(name: str):
    with open(DATA / name) as fh:
        return json.load(fh)


metrics = _load("metrics.json")
images = _load("demo_images.json")
benchmark = _load("benchmark_answers.json")
error_expl = _load("error_explanations.json")

pm = metrics["pixel_metrics"]
rf, ndvi_b, ndsm_b = pm["random_forest_multimodal"], pm["ndvi_only_baseline"], pm["ndsm_only_baseline"]
poly = metrics["polygon_metrics"]
fi = metrics["feature_importances"]

MANUAL_GRADES = {
    9: ("correct", "Matches the actual nDSM = DSM - DTM computation used in this pipeline's own preprocessing step."),
    8: ("correct", "Standard, accurate definitions - though in practice 'DEM' is often used loosely as a synonym for DTM rather than a distinct third category, which the answer doesn't flag."),
    21: ("imprecise", "The first part is right, but 'last-returns are essential for accurately modeling building surfaces' overstates it: buildings are opaque single-return surfaces, so first return equals last return there. The first/last distinction only really matters for vegetation, not buildings - this pipeline used first-return + all-non-noise points for the DSM specifically because building rooftops don't need last-return filtering."),
}


def bar(pct: float, color: str = "var(--gold)") -> str:
    pct = max(0, min(100, pct))
    return f'<div class="bar-track"><div class="bar-fill" style="width:{pct:.1f}%;background:{color}"></div></div>'


def fmt_pct(x: float) -> str:
    return f"{x*100:.1f}%"


def question_rows() -> str:
    rows = []
    for q in benchmark["questions"]:
        grade = MANUAL_GRADES.get(q["id"])
        badge = ""
        if grade:
            level, note = grade
            cls = "grade-ok" if level == "correct" else "grade-warn"
            label = "Verified correct" if level == "correct" else "Flagged: imprecise"
            badge = f'<div class="grade {cls}"><b>{label}</b> {note}</div>'
        rows.append(f"""
        <details class="qa">
          <summary><span class="qa-topic">{q['topic']}</span>{q['question']}</summary>
          <p class="qa-answer">{q['nova_answer']}</p>
          {badge}
        </details>""")
    return "\n".join(rows)


def error_cards() -> str:
    cards = []
    labels = {"false_positive": "False positive (predicted building, none in OSM)", "false_negative": "False negative (missed a real OSM building)"}
    for c in error_expl["cases"]:
        cards.append(f"""
        <div class="error-card">
          <div class="error-label">{labels[c['error_type']]}</div>
          <div class="error-stats">nDSM {c['ndsm_m']:.1f} m &middot; NDVI {c['ndvi']:.2f} &middot; slope {c['slope_deg']:.1f}&deg;</div>
          <p>{c['explanation']}</p>
        </div>""")
    return "\n".join(cards)


html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>GeoAI Building Intelligence</title>
<meta name="description" content="LiDAR + Sentinel-2 + OpenStreetMap building extraction pipeline over Washington DC, with a Bedrock Nova GIS/remote-sensing benchmark.">
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Big+Shoulders+Display:wght@600;700&family=IBM+Plex+Sans:wght@400;500;600&family=IBM+Plex+Mono:wght@400;500;600&display=swap">
<style>
  *, *::before, *::after {{ box-sizing: border-box; margin: 0; padding: 0; }}
  :root {{
    --gold: #E1C340; --bg: #0c0c12; --card: #13131f; --card-2: #191927;
    --border: #1e1e2e; --text: #e8e8f0; --muted: #7a7a9a;
    --green: #34d399; --indigo: #818cf8; --warn: #e0ac2b;
  }}
  body {{ background: var(--bg); color: var(--text); font-family: 'Segoe UI', system-ui, sans-serif; }}
  header {{ display: flex; align-items: center; gap: 14px; padding: 20px 40px; border-bottom: 1px solid var(--border); flex-wrap: wrap; }}
  header a.back {{ color: var(--muted); text-decoration: none; font-size: .85rem; }}
  header a.back:hover {{ color: var(--gold); }}
  h1 {{ font-family: "Big Shoulders Display", sans-serif; font-size: 1.3rem; font-weight: 700; color: var(--gold); }}
  .subtitle {{ font-size: .8rem; color: var(--muted); margin-top: 2px; }}
  header .links {{ margin-left: auto; display: flex; gap: 18px; }}
  header .links a {{ color: var(--muted); text-decoration: none; font-size: .85rem; }}
  header .links a:hover {{ color: var(--gold); }}

  .page {{ max-width: 1080px; margin: 0 auto; padding: 36px 24px 80px; }}
  .eyebrow {{ font: 600 12px "IBM Plex Mono", monospace; letter-spacing: .1em; text-transform: uppercase; color: var(--gold); margin-bottom: 10px; }}
  .hero h2 {{ font-family: "Big Shoulders Display", sans-serif; font-size: 2.2rem; color: #fff; margin-bottom: 10px; }}
  .hero p {{ color: var(--muted); max-width: 68ch; line-height: 1.6; font-size: 1rem; }}
  .aoi-line {{ font: 500 13px "IBM Plex Mono", monospace; color: var(--muted); margin-top: 14px; }}

  .kpi-row {{ display: grid; grid-template-columns: repeat(4, 1fr); gap: 12px; margin: 32px 0; }}
  .kpi {{ background: var(--card); border: 1px solid var(--border); border-radius: 12px; padding: 18px; }}
  .kpi .n {{ font: 700 26px "IBM Plex Mono", monospace; color: #fff; }}
  .kpi .l {{ font-size: .78rem; color: var(--muted); margin-top: 4px; }}

  h3.section {{ font-family: "Big Shoulders Display", sans-serif; font-size: 1.5rem; color: #fff; margin: 44px 0 16px; }}
  p.lead {{ color: var(--muted); max-width: 72ch; line-height: 1.65; margin-bottom: 18px; }}

  .img-grid {{ display: grid; grid-template-columns: repeat(2, 1fr); gap: 14px; margin: 20px 0; }}
  .img-card {{ background: var(--card); border: 1px solid var(--border); border-radius: 12px; overflow: hidden; }}
  .img-card .imgwrap {{ position: relative; }}
  .img-card img {{ display: block; width: 100%; }}
  .img-card .imgwrap img.overlay {{ position: absolute; top: 0; left: 0; }}
  .img-card .cap {{ padding: 10px 14px; font: 600 12px "IBM Plex Mono", monospace; color: var(--muted); border-top: 1px solid var(--border); }}

  table.metrics-table {{ width: 100%; border-collapse: collapse; margin: 18px 0; font-size: .92rem; }}
  table.metrics-table th, table.metrics-table td {{ padding: 10px 14px; border-bottom: 1px solid var(--border); text-align: left; }}
  table.metrics-table th {{ color: var(--muted); font-weight: 600; font-size: .8rem; text-transform: uppercase; letter-spacing: .04em; }}
  table.metrics-table td.num {{ font-family: "IBM Plex Mono", monospace; }}
  table.metrics-table tr.highlight td {{ color: var(--gold); font-weight: 600; }}

  .bar-row {{ display: grid; grid-template-columns: 110px 1fr 50px; gap: 10px; align-items: center; margin-bottom: 10px; font-size: .85rem; }}
  .bar-track {{ background: var(--card-2); border-radius: 6px; height: 10px; overflow: hidden; }}
  .bar-fill {{ height: 100%; border-radius: 6px; }}

  .note {{ background: var(--card); border-left: 3px solid var(--gold); border-radius: 0 10px 10px 0; padding: 16px 20px; margin: 20px 0; font-size: .92rem; color: var(--text); line-height: 1.6; }}

  .error-card {{ background: var(--card); border: 1px solid var(--border); border-radius: 12px; padding: 16px 18px; margin-bottom: 12px; }}
  .error-label {{ font: 700 13px "IBM Plex Mono", monospace; color: var(--warn); margin-bottom: 4px; }}
  .error-stats {{ font: 500 12px "IBM Plex Mono", monospace; color: var(--muted); margin-bottom: 8px; }}
  .error-card p {{ font-size: .92rem; line-height: 1.55; }}

  details.qa {{ background: var(--card); border: 1px solid var(--border); border-radius: 10px; padding: 12px 16px; margin-bottom: 8px; }}
  details.qa summary {{ cursor: pointer; font-size: .92rem; font-weight: 500; }}
  details.qa .qa-topic {{ display: inline-block; background: rgba(225,195,64,.12); color: var(--gold); font: 600 10px "IBM Plex Mono", monospace; text-transform: uppercase; letter-spacing: .05em; padding: 2px 8px; border-radius: 10px; margin-right: 10px; }}
  .qa-answer {{ margin-top: 10px; font-size: .88rem; color: var(--text); line-height: 1.6; }}
  .grade {{ margin-top: 10px; font-size: .82rem; padding: 8px 12px; border-radius: 8px; line-height: 1.5; }}
  .grade-ok {{ background: rgba(52,211,153,.1); color: var(--green); }}
  .grade-warn {{ background: rgba(224,172,43,.1); color: var(--warn); }}
  .grade b {{ display: block; margin-bottom: 2px; }}

  .stack-grid {{ display: grid; grid-template-columns: repeat(3, 1fr); gap: 10px; margin: 18px 0; }}
  .stack-item {{ background: var(--card); border: 1px solid var(--border); border-radius: 10px; padding: 14px 16px; }}
  .stack-item b {{ display: block; font: 600 13px "IBM Plex Mono", monospace; color: var(--gold); margin-bottom: 4px; }}
  .stack-item span {{ font-size: 12.5px; color: var(--text); line-height: 1.5; }}

  footer {{ text-align: center; padding: 24px; color: var(--muted); font-size: .82rem; border-top: 1px solid var(--border); margin-top: 40px; }}
  footer span {{ color: var(--gold); }}
  footer a {{ color: var(--muted); }}

  @media (max-width: 760px) {{
    header, .page {{ padding-left: 20px; padding-right: 20px; }}
    .kpi-row {{ grid-template-columns: repeat(2, 1fr); }}
    .img-grid, .stack-grid {{ grid-template-columns: 1fr; }}
    .bar-row {{ grid-template-columns: 90px 1fr 40px; }}
  }}
</style>
</head>
<body>

<header>
  <div>
    <h1>GeoAI Building Intelligence</h1>
    <div class="subtitle">LiDAR + Sentinel-2 + OpenStreetMap, Washington DC</div>
  </div>
  <div class="links">
    <a href="https://www.forwardforecasting.eu/">Projects</a>
    <a href="https://education.forwardforecasting.eu/gis-lidar/">Field Notes</a>
    <a href="https://github.com/fborbon/gis-lidar" target="_blank" rel="noopener">GitHub</a>
  </div>
</header>

<div class="page">

  <section class="hero">
    <div class="eyebrow">Geospatial AI Demo</div>
    <h2>Multimodal building extraction, validated the way a GIS reviewer actually checks it</h2>
    <p>A real USGS 3DEP LiDAR tile and a cloud-free Sentinel-2 scene, fused into per-pixel features (canopy/building height, NDVI, terrain slope, texture), trained against real OpenStreetMap building footprints, and evaluated at both the pixel level and the polygon level, over a dense rowhouse block on Capitol Hill, Washington DC.</p>
    <div class="aoi-line">AOI: 38.882-38.888N, -77.005 to -76.998W &middot; 2m grid, {metrics['aoi']['grid_shape'][0]}&times;{metrics['aoi']['grid_shape'][1]} px &middot; spatial holdout test region (right 25% of grid)</div>
  </section>

  <div class="kpi-row">
    <div class="kpi"><div class="n">{fmt_pct(rf['f1'])}</div><div class="l">Pixel F1, multimodal RF</div></div>
    <div class="kpi"><div class="n">{fmt_pct(rf['iou'])}</div><div class="l">Pixel IoU, multimodal RF</div></div>
    <div class="kpi"><div class="n">{fmt_pct(poly['detection_rate'])}</div><div class="l">Building-level detection rate</div></div>
    <div class="kpi"><div class="n">{poly['n_gt']}</div><div class="l">OSM buildings in test region</div></div>
  </div>

  <h3 class="section">The raster stack</h3>
  <p class="lead">Every layer below is resampled onto the same 2m grid in EPSG:32618 (UTM 18N). nDSM comes from the LiDAR point cloud (DSM minus DTM); NDVI comes from Sentinel-2's native 10m red/NIR bands upsampled onto the 2m grid, which is why NDVI looks visibly blockier than the LiDAR-derived layers, an honest illustration of the spatial-resolution tradeoff between the two sensors.</p>
  <div class="img-grid">
    <div class="img-card"><img src="data:image/png;base64,{images['ndsm']}"><div class="cap">nDSM (LiDAR building/canopy height, 0-25m)</div></div>
    <div class="img-card"><img src="data:image/png;base64,{images['ndvi']}"><div class="cap">NDVI (Sentinel-2, resampled to 2m)</div></div>
    <div class="img-card">
      <div class="imgwrap"><img src="data:image/png;base64,{images['ndsm']}"><img class="overlay" src="data:image/png;base64,{images['osm_mask']}"></div>
      <div class="cap">Ground truth: OSM buildings (gold) over nDSM</div>
    </div>
    <div class="img-card">
      <div class="imgwrap"><img src="data:image/png;base64,{images['ndsm']}"><img class="overlay" src="data:image/png;base64,{images['rf_pred']}"></div>
      <div class="cap">RandomForest prediction (green) over nDSM</div>
    </div>
  </div>

  <h3 class="section">Why multimodal fusion beats either sensor alone</h3>
  <p class="lead">Two naive single-feature baselines, evaluated on the exact same held-out test region as the full model, make the case with real numbers rather than a marketing claim:</p>
  <table class="metrics-table">
    <tr><th>Model</th><th>Precision</th><th>Recall</th><th>F1</th><th>IoU</th></tr>
    <tr><td>NDVI-only (low NDVI &rarr; guess building)</td><td class="num">{fmt_pct(ndvi_b['precision'])}</td><td class="num">{fmt_pct(ndvi_b['recall'])}</td><td class="num">{fmt_pct(ndvi_b['f1'])}</td><td class="num">{fmt_pct(ndvi_b['iou'])}</td></tr>
    <tr><td>nDSM-only (height &gt; 2.5m &rarr; guess building)</td><td class="num">{fmt_pct(ndsm_b['precision'])}</td><td class="num">{fmt_pct(ndsm_b['recall'])}</td><td class="num">{fmt_pct(ndsm_b['f1'])}</td><td class="num">{fmt_pct(ndsm_b['iou'])}</td></tr>
    <tr class="highlight"><td>RandomForest, nDSM+NDVI+slope+texture</td><td class="num">{fmt_pct(rf['precision'])}</td><td class="num">{fmt_pct(rf['recall'])}</td><td class="num">{fmt_pct(rf['f1'])}</td><td class="num">{fmt_pct(rf['iou'])}</td></tr>
  </table>
  <p class="lead">NDVI-only is precise but misses most buildings (roads, parking lots, and bare soil also read as "not vegetation"). nDSM-only catches almost everything raised off the ground (99.7% recall) but drags in every tree along the way, cratering precision. Only the combined model gets both right.</p>

  <h3 class="section">Feature importance</h3>
  {''.join(f'<div class="bar-row"><span>{k}</span>{bar(v*100)}<span>{v*100:.1f}%</span></div>' for k, v in sorted(fi.items(), key=lambda kv: -kv[1]))}

  <h3 class="section">The part pixel accuracy hides: polygon-level evaluation</h3>
  <p class="lead">Pixel metrics look strong (F1 {fmt_pct(rf['f1'])}), but a GIS reviewer cares about individual buildings, not pixels. Vectorizing the prediction and matching it against individual OSM building polygons in the test region tells a more honest story:</p>
  <table class="metrics-table">
    <tr><th>Metric</th><th>Value</th></tr>
    <tr><td>Ground-truth buildings in test region</td><td class="num">{poly['n_gt']}</td></tr>
    <tr><td>Predicted building polygons</td><td class="num">{poly['n_pred']}</td></tr>
    <tr><td>Matched at IoU &ge; 0.3</td><td class="num">{poly['n_matched']}</td></tr>
    <tr class="highlight"><td>Building-level detection rate</td><td class="num">{fmt_pct(poly['detection_rate'])}</td></tr>
    <tr><td>Mean matched-polygon IoU</td><td class="num">{poly['mean_matched_iou']:.2f}</td></tr>
    <tr><td>Mean building area error</td><td class="num">{poly['mean_area_error_pct']:.1f}%</td></tr>
    <tr><td>Mean centroid displacement</td><td class="num">{poly['mean_centroid_disp_m']:.1f} m</td></tr>
  </table>
  <div class="note">The detection rate ({fmt_pct(poly['detection_rate'])}) is much lower than the pixel F1 suggests, and the reason is structural, not a model bug: Capitol Hill rowhouses share party walls, so neighboring units present one continuous rooftop in the nDSM with no height discontinuity between them. The classifier correctly flags the roofline as "building" at the pixel level, but the vectorizer then merges 3-4 attached rowhouses into a single polygon, which counts as one match against many ground-truth parcels. This is exactly the kind of instance-segmentation problem pixel accuracy hides and polygon-level GIS evaluation exposes.</div>

  <h3 class="section">Grounded error explanations (Bedrock Amazon Nova Lite)</h3>
  <p class="lead">Three real misclassifications from the held-out test region, with Nova Lite explaining each one from its actual measured nDSM/NDVI/slope values, not a generic answer:</p>
  {error_cards()}

  <h3 class="section">GIS / Remote Sensing knowledge benchmark</h3>
  <p class="lead">The freelance role this project was built for is about evaluating AI outputs on GIS/remote-sensing expertise, so the pipeline turns that around on itself: 30 domain questions put to Amazon Nova Lite, with the ones this project has direct ground truth for (DSM/DTM/nDSM, LiDAR returns) graded against that ground truth rather than taken at face value.</p>
  {question_rows()}

  <h3 class="section">Data sources and stack</h3>
  <div class="stack-grid">
    <div class="stack-item"><b>USGS 3DEP LiDAR</b><span>Public LAZ tile, no-sign-request S3/HTTPS, read with laspy. Ground-classified points to a DTM, all non-noise returns to a DSM.</span></div>
    <div class="stack-item"><b>Sentinel-2 L2A</b><span>Searched via the free Element84 STAC API, red/NIR/blue bands windowed-read straight off the public COGs with rasterio, no download.</span></div>
    <div class="stack-item"><b>OpenStreetMap</b><span>Building footprints via a single Overpass API query, used as ground truth for both training labels and evaluation.</span></div>
    <div class="stack-item"><b>scikit-learn</b><span>RandomForestClassifier on 4 engineered features, spatially held out (not a random pixel split) for evaluation.</span></div>
    <div class="stack-item"><b>rasterio / geopandas / shapely</b><span>Grid alignment, polygon vectorization, and the polygon-level GIS metrics.</span></div>
    <div class="stack-item"><b>AWS Bedrock (Amazon Nova Lite)</b><span>eu-west-1 cross-region inference profile. Benchmark Q&amp;A and grounded error explanations, ~30 short prompts, well under a cent total.</span></div>
  </div>

  <p class="lead" style="margin-top:28px;">Full pipeline, methodology, and limitations write-up: <a href="https://github.com/fborbon/gis-lidar" style="color:var(--gold)" target="_blank" rel="noopener">github.com/fborbon/gis-lidar</a>. Longer narrative: <a href="https://education.forwardforecasting.eu/gis-lidar/" style="color:var(--gold)">Field Notes write-up</a>.</p>

</div>

<footer>
  Data: USGS 3DEP (public domain), Sentinel-2 L2A (Copernicus, ESA), &copy; OpenStreetMap contributors (ODbL). Built on AWS Bedrock (Amazon Nova Lite). Built by <span>fborbon</span> &mdash; Forward Forecasting &copy; 2026
</footer>

</body>
</html>
"""

STATIC_OUT.parent.mkdir(exist_ok=True)
STATIC_OUT.write_text(html)
print(f"Wrote {STATIC_OUT} ({len(html)/1024:.1f} KB)")
