"""Perspective-robust shelf-row grouping (sequential RANSAC on YOLO box centres).
Each row gets its OWN slope (handles vanishing-point convergence the global-slope shear can't);
inlier band ∝ each box's own height (perspective shrinks box height & row gap together → self-tuning).
Recipe: DeepScribe arXiv:2306.01268 §4.4 + engineering guards (x-span, cy-gap merge, fallback).
Deterministic (fixed seed). boxes: (N,4) int (x1,y1,x2,y2)."""
import numpy as np

BAND_FRAC   = 0.55   # inlier band = BAND_FRAC * box height (per-point). merged->lower, split->raise
MIN_INLIERS = 3      # a real row needs >= this many boxes
MAX_SLOPE   = 0.70   # |slope| cap (~35°); blocks columns/diagonals, allows steep aisle rows
MIN_SPAN    = 0.40   # a real row spans >= this fraction of the shelf width (replaces filter_sparse)
MERGE_FRAC  = 0.70   # merge two consecutive rows if centre gap < MERGE_FRAC * local box height
ITERS       = 500
MIN_DX_FRAC = 0.06   # sampled pair must span >= this fraction of width (avoid near-vertical fits)

def group_rows(boxes, W, seed=0):
    n = len(boxes)
    if n < MIN_INLIERS:
        return [list(range(n))] if n else []
    cx = (boxes[:, 0] + boxes[:, 2]) / 2.0; cy = (boxes[:, 1] + boxes[:, 3]) / 2.0
    h = (boxes[:, 3] - boxes[:, 1]).astype(float)
    span_w = float(boxes[:, 2].max() - boxes[:, 0].min()); min_dx = MIN_DX_FRAC * span_w
    rng = np.random.default_rng(seed); remaining = list(range(n)); rows = []
    while len(remaining) >= MIN_INLIERS:
        rem = np.array(remaining); best_in = []; best = -1.0
        for _ in range(ITERS):
            a, b = rng.choice(rem, size=2, replace=False)
            if abs(cx[a] - cx[b]) < min_dx:
                continue
            m = (cy[b] - cy[a]) / (cx[b] - cx[a])
            if abs(m) > MAX_SLOPE:
                continue
            c = cy[a] - m * cx[a]
            d = np.abs(cy[rem] - (m * cx[rem] + c))
            inl = rem[d < BAND_FRAC * h[rem]]                 # per-point band (each box's own height)
            if len(inl) < MIN_INLIERS:
                continue
            sp = (cx[inl].max() - cx[inl].min()) / span_w
            if sp < MIN_SPAN:                                 # real rows cross the shelf
                continue
            score = len(inl) + sp
            if score > best:
                best = score; best_in = inl
        if len(best_in) < MIN_INLIERS:
            break
        rows.append([int(k) for k in best_in])
        rs = set(int(k) for k in best_in); remaining = [k for k in remaining if k not in rs]
    if remaining and rows:                                   # leftovers -> nearest row by cy
        rcy = [np.median([cy[k] for k in r]) for r in rows]
        for k in remaining:
            rows[int(np.argmin([abs(cy[k] - rc) for rc in rcy]))].append(k)
    elif remaining:
        rows.append(remaining)
    rows.sort(key=lambda r: np.median([cy[k] for k in r]))
    merged = []                                              # merge over-split consecutive rows
    for r in rows:
        if merged:
            pcy = np.median([cy[k] for k in merged[-1]]); ccy = np.median([cy[k] for k in r])
            mh = np.median([h[k] for k in merged[-1] + r])
            if ccy - pcy < MERGE_FRAC * mh:
                merged[-1] = merged[-1] + r; continue
        merged.append(list(r))
    return merged

def cluster_rows_robust(boxes, W, old_method, filter_fn=None, img_w=None):
    """Drop-in for the row pipeline: perspective-robust grouping; falls back to old_method
    (then filter_fn, the old filter_sparse_rows) when degenerate — too few boxes, or only one
    row is found (where the global-slope method is already correct). The group_rows path needs
    no filter_sparse: its MIN_INLIERS + MIN_SPAN gates already drop promo-header / sparse rows."""
    def _fb():
        r = old_method(boxes)
        return filter_fn(r, boxes, img_w) if (filter_fn is not None and img_w is not None) else r
    if len(boxes) < 2 * MIN_INLIERS:
        return _fb()
    try:
        rows = group_rows(boxes, W)
    except Exception:
        return _fb()
    if len(rows) <= 1:
        return _fb()
    return rows
