"""GIS/Remote-Sensing domain benchmark and error-explanation calls via AWS Bedrock
Amazon Nova Lite, using the EU cross-region inference profile (this account's
standing default region for new AWS/AI usage is eu-west-1; Nova is not invokable
there with a plain model id, only via the "eu." cross-region inference profile,
which is what MODEL_ID below points at).
"""
from __future__ import annotations

import json

import boto3

REGION = "eu-west-1"
MODEL_ID = "eu.amazon.nova-lite-v1:0"


def _invoke(prompt: str, max_tokens: int = 220) -> tuple[str, dict]:
    client = boto3.client("bedrock-runtime", region_name=REGION)
    body = {
        "messages": [{"role": "user", "content": [{"text": prompt}]}],
        "inferenceConfig": {"maxTokens": max_tokens, "temperature": 0.2},
    }
    resp = client.invoke_model(modelId=MODEL_ID, contentType="application/json", accept="application/json", body=json.dumps(body))
    payload = json.loads(resp["body"].read())
    text = payload["output"]["message"]["content"][0]["text"]
    usage = payload.get("usage", {})
    return text, usage


def ask_benchmark_question(question: str) -> tuple[str, dict]:
    prompt = f"Answer this GIS / remote sensing interview question in 2-4 sentences, precisely and technically:\n\n{question}"
    return _invoke(prompt)


def explain_misclassification(ndsm_m: float, ndvi: float, slope_deg: float, error_type: str) -> tuple[str, dict]:
    prompt = (
        "You are reviewing a LiDAR+Sentinel-2 building classifier's error on one pixel cluster.\n"
        f"Measured values: normalized surface height (nDSM) = {ndsm_m:.2f} m above bare earth, "
        f"NDVI = {ndvi:.2f}, terrain slope = {slope_deg:.1f} degrees.\n"
        f"The classifier produced a {error_type} at this location (ground truth from OpenStreetMap).\n"
        "In 2-3 sentences, give the most likely physical/geospatial explanation for this specific error, "
        "grounded in these exact values (e.g. height near a threshold, vegetation cover, terrain effects)."
    )
    return _invoke(prompt, max_tokens=150)
