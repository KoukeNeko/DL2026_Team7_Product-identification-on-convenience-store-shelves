"""v2 YOLOv8-seg structural detection for the live demo overlay.

Returns shelf MASK BANDS (class 0) and door MASKS (class 1, convex-hulled) as pixel
polygons. Drawing the mask band directly (not a fitted line) keeps the shelf indicator
aligned to the photo even when the mask is imperfect on reflective glass. Lazy-loaded, cached.
"""
import os

import cv2
import numpy as np
from ultralytics import YOLO

MODELS_DIR = os.environ.get("MODELS_DIR", os.path.join(os.path.dirname(os.path.abspath(__file__)), "models"))
SEG_WEIGHTS = os.path.join(MODELS_DIR, "shelf_struct_v2.pt")
CLASS_SHELF = 0
CLASS_DOOR = 1
SEG_CONF = 0.25
SEG_IMGSZ = 1024
SIMPLIFY_EPS_PX = 1.5          # approxPolyDP tolerance: trims mask points, keeps the shape

_model = None


def load():
    global _model
    if _model is None:
        _model = YOLO(SEG_WEIGHTS)
    return _model


def _to_int_points(points):
    return [[int(round(x)), int(round(y))] for x, y in points]


def _simplify(polygon):
    points = np.asarray(polygon, dtype=np.float32).reshape(-1, 1, 2)
    approx = cv2.approxPolyDP(points, SIMPLIFY_EPS_PX, True).reshape(-1, 2)
    return _to_int_points(approx)


def _convex_hull(polygon):
    points = np.asarray(polygon, dtype=np.float32).reshape(-1, 1, 2)
    return _to_int_points(cv2.convexHull(points).reshape(-1, 2))


def detect(pil_image):
    """Return (shelf_bands, door_masks): lists of [[x, y], ...] pixel polygons."""
    result = load().predict(pil_image, conf=SEG_CONF, imgsz=SEG_IMGSZ,
                            retina_masks=True, verbose=False)[0]
    shelf_bands, door_masks = [], []
    if result.masks is None:
        return shelf_bands, door_masks
    classes = result.boxes.cls.cpu().numpy().astype(int)
    for polygon, predicted_class in zip(result.masks.xy, classes):
        if len(polygon) < 3:
            continue
        if predicted_class == CLASS_SHELF:
            shelf_bands.append(_simplify(polygon))
        elif predicted_class == CLASS_DOOR:
            door_masks.append(_convex_hull(polygon))
    return shelf_bands, door_masks
