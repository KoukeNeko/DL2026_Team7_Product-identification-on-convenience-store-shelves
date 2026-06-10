"""Detector output post-processing: containment de-dup + price-tag filter.

Greedy IoU-NMS (inside ultralytics) keeps two boxes when one is NESTED inside the
other, because their IoU is low (very different areas). That produces the
"big loose box around a can + a tight box inside it" artifact. This containment
de-dup removes the lower-confidence box of any nested pair, then the price-tag
filter removes electronic-shelf-label false positives.

Apply AFTER detection, BEFORE classification / counting:
    keep = postprocess(image, boxes_xyxy, scores)
"""
from __future__ import annotations

import numpy as np
from PIL import Image

from pricetag_filter import filter_price_tags

CONTAIN_THRESH = 0.80   # intersection / smaller-area above this => boxes are nested


def dedup_contained(boxes: np.ndarray, scores: np.ndarray, thresh: float = CONTAIN_THRESH) -> np.ndarray:
    """Drop the lower-confidence box of any nested pair. Returns a keep-mask.

    Two boxes are "nested" when the smaller one is almost entirely inside the
    larger one (intersection / smaller-area > thresh) even though their IoU is
    low — exactly the case greedy NMS leaves behind.
    """
    n = len(boxes)
    keep = np.ones(n, dtype=bool)
    if n < 2:
        return keep
    area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    order = np.argsort(-scores)  # high confidence first
    for rank_i, i in enumerate(order):
        if not keep[i]:
            continue
        for j in order[rank_i + 1:]:
            if not keep[j]:
                continue
            x1 = max(boxes[i, 0], boxes[j, 0]); y1 = max(boxes[i, 1], boxes[j, 1])
            x2 = min(boxes[i, 2], boxes[j, 2]); y2 = min(boxes[i, 3], boxes[j, 3])
            inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
            if inter <= 0:
                continue
            if inter / min(area[i], area[j]) > thresh:
                keep[j] = False  # j has lower confidence (later in order)
    return keep


def postprocess(image: Image.Image, boxes: np.ndarray, scores: np.ndarray) -> np.ndarray:
    """Combined keep-mask: drop nested duplicates AND price-tag false positives."""
    return dedup_contained(boxes, scores) & filter_price_tags(image, boxes)


MERGE_X_OVERLAP = 0.55   # two boxes share a column if smaller x-extent overlaps >= this
MERGE_GAP_FRAC = 0.35    # merge if vertical gap < this * shorter height (a bottle's cap/label/body touch)


def _x_overlap_fraction(box_a: np.ndarray, box_b: np.ndarray) -> float:
    inter = max(0.0, min(box_a[2], box_b[2]) - max(box_a[0], box_b[0]))
    return inter / max(1.0, min(box_a[2] - box_a[0], box_b[2] - box_b[0]))


def merge_fragments(boxes: np.ndarray, scores: np.ndarray) -> tuple:
    """Union same-column boxes that vertically touch (a bottle split into cap/label/body) into
    one facing. Stacked boxes are NOT nested, so dedup_contained misses them; the large
    cross-row shelf gap keeps separate products from merging."""
    n = len(boxes)
    if n < 2:
        return boxes, scores
    parent = list(range(n))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for i in range(n):
        for j in range(i + 1, n):
            if _x_overlap_fraction(boxes[i], boxes[j]) < MERGE_X_OVERLAP:
                continue
            top, bottom = (boxes[i], boxes[j]) if boxes[i][1] <= boxes[j][1] else (boxes[j], boxes[i])
            gap = bottom[1] - top[3]
            shorter = min(boxes[i][3] - boxes[i][1], boxes[j][3] - boxes[j][1])
            if gap < MERGE_GAP_FRAC * shorter:
                parent[find(i)] = find(j)

    groups = {}
    for i in range(n):
        groups.setdefault(find(i), []).append(i)
    merged_boxes, merged_scores = [], []
    for members in groups.values():
        g = boxes[members]
        merged_boxes.append([g[:, 0].min(), g[:, 1].min(), g[:, 2].max(), g[:, 3].max()])
        merged_scores.append(float(scores[members].max()))
    return np.array(merged_boxes), np.array(merged_scores)


# Detection config — must match the locked deployment settings.
IMGSZ, CONF, NMS_IOU, MAX_DET = 1280, 0.20, 0.7, 1000


def detect_products(model, image: Image.Image, device: int = 0) -> tuple:
    """THE entry point for demo/pipeline: YOLO detect + post-process.

    Returns (boxes_xyxy, scores) of clean PRODUCT detections only — price-tag
    false positives and nested duplicate boxes already removed. Use this instead
    of calling model.predict directly anywhere boxes are drawn or counted.
    """
    result = model.predict(image, imgsz=IMGSZ, conf=CONF, iou=NMS_IOU,
                           max_det=MAX_DET, verbose=False, device=device)[0]
    boxes = result.boxes.xyxy.cpu().numpy()
    scores = result.boxes.conf.cpu().numpy()
    keep = postprocess(image, boxes, scores)
    return merge_fragments(boxes[keep], scores[keep])           # collapse cap/label/body splits
