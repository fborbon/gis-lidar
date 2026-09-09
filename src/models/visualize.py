"""Render the raster layers as small PNGs for embedding (base64) in the static demo page."""
from __future__ import annotations

import base64
import io

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def array_to_base64_png(arr: np.ndarray, cmap: str, vmin=None, vmax=None) -> str:
    fig, ax = plt.subplots(figsize=(arr.shape[1] / 100, arr.shape[0] / 100), dpi=150)
    ax.imshow(arr, cmap=cmap, vmin=vmin, vmax=vmax, origin="upper")
    ax.axis("off")
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
    buf = io.BytesIO()
    fig.savefig(buf, format="png", transparent=False)
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("ascii")


def mask_to_rgba_base64(mask: np.ndarray, color=(225, 195, 64)) -> str:
    """Render a boolean mask as a transparent-background RGBA PNG in the given color,
    for overlaying on top of another image in the browser.
    """
    h, w = mask.shape
    rgba = np.zeros((h, w, 4), dtype=np.uint8)
    rgba[..., 0] = color[0]
    rgba[..., 1] = color[1]
    rgba[..., 2] = color[2]
    rgba[..., 3] = np.where(mask, 200, 0).astype(np.uint8)
    fig, ax = plt.subplots(figsize=(w / 100, h / 100), dpi=150)
    ax.imshow(rgba, origin="upper")
    ax.axis("off")
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
    buf = io.BytesIO()
    fig.savefig(buf, format="png", transparent=True)
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("ascii")
