"""Price-tag (electronic shelf label) false-positive filter for detector output.

The SKU-110K detector occasionally boxes electronic shelf labels (white tags with
a price like "59" + a barcode) as products. The validation GT excludes price tags,
so these are false positives. This post-processing step removes them BEFORE
classification / counting.

Signature of a price tag (ALL required):
  - shape: square-to-wide  (price tags are never taller than wide; a tall box is a
                            can/bottle/package -> always a product, never a tag)
  - low colour saturation  (white/gray label, not a colourful package)
  - bright                 (white background)
  - barcode stripe texture (a horizontal sub-band with strong vertical-bar edges,
                            i.e. horizontal gradient >> vertical gradient)

Thresholds are deliberately CONSERVATIVE: it is far worse to drop a real product
(hurts recall) than to leave a cosmetic tag box. The shape gate spares tall
metallic cans (e.g. a silver Monster can whose claw logo mimics barcode stripes).
White/clear products (water, milk) are spared because they lack the stripe.
Residual tags that slip through are caught downstream by the classifier's unknown
threshold.
"""
from __future__ import annotations

import numpy as np
from PIL import Image

ASPECT_MIN = 0.85   # width/height lower bound: tall boxes (cans/bottles) are products
SAT_MAX = 0.30      # saturation upper bound (grayscale label)
BRIGHT_MIN = 110    # brightness lower bound (white background)
STRIPE_MIN = 1.7    # max over horizontal bands of (h-gradient / v-gradient)
MIN_SIDE = 10       # ignore boxes too small to assess
BANDS = 4           # the barcode localizes to one horizontal band


def _barcode_stripe_score(gray: np.ndarray) -> float:
    """Max horizontal/vertical gradient ratio over horizontal sub-bands.

    A barcode has many vertical bars -> large horizontal gradient, small vertical
    gradient -> ratio >> 1, localized to the band that contains the barcode.
    """
    height = gray.shape[0]
    best = 0.0
    for i in range(BANDS):
        band = gray[i * height // BANDS:(i + 1) * height // BANDS]
        if band.shape[0] < 3:
            continue
        h_grad = np.abs(np.diff(band, axis=1)).mean()
        v_grad = np.abs(np.diff(band, axis=0)).mean()
        best = max(best, h_grad / (v_grad + 1e-3))
    return best


def is_price_tag(crop: Image.Image) -> bool:
    """True if the crop looks like an electronic shelf label (price tag)."""
    width, height = crop.size
    if height < MIN_SIDE or width < MIN_SIDE:
        return False
    if width / height < ASPECT_MIN:   # tall box -> product (can/bottle), never a tag
        return False
    gray = np.asarray(crop.convert("L"), dtype=np.float32)
    saturation = np.asarray(crop.convert("HSV"), dtype=np.float32)[:, :, 1].mean() / 255.0
    brightness = gray.mean()
    if saturation >= SAT_MAX or brightness <= BRIGHT_MIN:
        return False
    return _barcode_stripe_score(gray) > STRIPE_MIN


def filter_price_tags(image: Image.Image, boxes_xyxy: np.ndarray) -> np.ndarray:
    """Return a boolean keep-mask (True = product, False = price tag) for boxes."""
    keep = np.ones(len(boxes_xyxy), dtype=bool)
    for idx, (x1, y1, x2, y2) in enumerate(boxes_xyxy.astype(int)):
        if is_price_tag(image.crop((x1, y1, x2, y2))):
            keep[idx] = False
    return keep
