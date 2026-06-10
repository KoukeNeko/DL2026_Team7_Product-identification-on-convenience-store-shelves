"""Place products onto the shelves the v2 seg model found (the point of detecting them):

    door masks  -> bays   : a product belongs to the door bay its centre-x falls in
    shelf bands -> rows    : a product sits on the band nearest its BOTTOM edge (gravity prior)

Each band is fitted to a line y = m*x + b so a tilted (perspective) shelf still reads off the
right height at the product's x. For any bay where seg found < 2 bands (seg recall ~0.7), fall
back to the rowlib RANSAC clustering so placement degrades gracefully instead of collapsing.
"""
import numpy as np

import rowlib
import virtual_shelf as vs

MIN_BANDS_PER_BAY = 2
BAND_X_OVERLAP = 0.2     # a band belongs to a bay if it overlaps >= this fraction of the bay width
BAND_MERGE_FRAC = 0.04   # collapse bands closer than this (* image height) — seg fires duplicates on glass
DOOR_NMS_OVERLAP = 0.5   # drop a door whose x-extent overlaps a bigger one by >= this (nested over-detect)
DOOR_MIN_AREA_FRAC = 0.3 # drop a door smaller than this * the biggest door's area (peripheral background unit)
DOOR_OUTSIDE_MARGIN = 0.05  # a product this far (* width) outside every kept door is background -> dropped


def _poly_xrange(polygon):
    pts = np.asarray(polygon, dtype=float)
    return float(pts[:, 0].min()), float(pts[:, 0].max())


def _band_line(polygon):
    """Fit y = m*x + b to a band polygon; return (m, b, x_min, x_max)."""
    pts = np.asarray(polygon, dtype=float)
    xs, ys = pts[:, 0], pts[:, 1]
    if xs.max() - xs.min() < 1.0:                         # near-vertical/degenerate -> flat line
        return 0.0, float(ys.mean()), float(xs.min()), float(xs.max())
    slope, intercept = np.polyfit(xs, ys, 1)
    return float(slope), float(intercept), float(xs.min()), float(xs.max())


def _door_boxes(seg_doors):
    out = []
    for polygon in seg_doors:
        a = np.asarray(polygon, dtype=float)
        x1, x2 = float(a[:, 0].min()), float(a[:, 0].max())
        y1, y2 = float(a[:, 1].min()), float(a[:, 1].max())
        out.append((x1, x2, y1, y2, (x2 - x1) * (y2 - y1)))
    return out


def _clean_doors(seg_doors):
    """NMS overlapping seg doors + drop tiny peripheral ones; return bay x-ranges left->right."""
    kept = []
    for door in sorted(_door_boxes(seg_doors), key=lambda d: -d[4]):     # largest area first
        nested = any((min(door[1], k[1]) - max(door[0], k[0])) >
                     DOOR_NMS_OVERLAP * min(door[1] - door[0], k[1] - k[0]) for k in kept)
        if not nested:
            kept.append(door)
    if not kept:
        return []
    max_area = kept[0][4]
    kept = [k for k in kept if k[4] >= DOOR_MIN_AREA_FRAC * max_area]    # peripheral background unit
    kept.sort(key=lambda d: d[0])
    return [(k[0], k[1]) for k in kept]


def assign_doors(boxes, seg_doors, width):
    """Return (door_id per box, n_doors); door_id -1 = far-outside background (dropped).
    No usable doors -> one open bay (keep all)."""
    ranges = _clean_doors(seg_doors)
    if not ranges:
        return [0] * len(boxes), 1
    margin = DOOR_OUTSIDE_MARGIN * width
    door_ids = []
    for box in boxes:
        center_x = (box[0] + box[2]) / 2.0
        inside = next((d for d, (lo, hi) in enumerate(ranges) if lo <= center_x <= hi), None)
        if inside is None:
            dists = [min(abs(center_x - lo), abs(center_x - hi)) for lo, hi in ranges]
            nearest = int(np.argmin(dists))
            inside = nearest if dists[nearest] <= margin else -1        # far outside -> background
        door_ids.append(inside)
    return door_ids, len(ranges)


def _rowlib_fallback(member_boxes, width):
    bay_width = float(member_boxes[:, 2].max() - member_boxes[:, 0].min())
    return rowlib.cluster_rows_robust(member_boxes, float(width), vs.cluster_rows,
                                      vs.filter_sparse_rows, bay_width)


def _merge_bands(bands, center_x, tolerance):
    """Collapse bands whose height at center_x is within tolerance (seg duplicates on glass)."""
    if not bands:
        return bands
    ordered = sorted(bands, key=lambda bd: bd[0] * center_x + bd[1])
    kept = [ordered[0]]
    for band in ordered[1:]:
        if (band[0] * center_x + band[1]) - (kept[-1][0] * center_x + kept[-1][1]) >= tolerance:
            kept.append(band)
    return kept


def assign(boxes, seg_shelf, seg_doors, width, height):
    """Return (door_ids, row_ids, n_doors, max_rows) for the products in `boxes`."""
    n = len(boxes)
    if n == 0:
        return [], [], 0, 0
    door_ids, n_doors = assign_doors(boxes, seg_doors, width)
    bands = [_band_line(p) for p in seg_shelf if len(p) >= 2]
    row_ids = [0] * n

    for bay in range(n_doors):
        members = [i for i in range(n) if door_ids[i] == bay]
        if not members:
            continue
        bay_x1 = min(boxes[i][0] for i in members)
        bay_x2 = max(boxes[i][2] for i in members)
        bay_span = max(1.0, bay_x2 - bay_x1)
        bay_center_x = (bay_x1 + bay_x2) / 2.0
        bay_bands = [bd for bd in bands
                     if (min(bd[3], bay_x2) - max(bd[2], bay_x1)) > BAND_X_OVERLAP * bay_span]
        bay_bands = _merge_bands(bay_bands, bay_center_x, BAND_MERGE_FRAC * height)

        if len(bay_bands) < MIN_BANDS_PER_BAY:            # seg too sparse here -> rowlib fallback
            member_boxes = np.array([boxes[i] for i in members])
            for ri, row in enumerate(_rowlib_fallback(member_boxes, width)):
                for k in row:
                    row_ids[members[k]] = ri
            continue

        bay_bands.sort(key=lambda bd: bd[0] * bay_center_x + bd[1])   # top -> bottom
        for i in members:
            center_x = (boxes[i][0] + boxes[i][2]) / 2.0
            bottom_y = boxes[i][3]
            band_y = [bd[0] * center_x + bd[1] for bd in bay_bands]
            row_ids[i] = int(np.argmin([abs(y - bottom_y) for y in band_y]))

    max_rows = max(row_ids) + 1 if row_ids else 0
    return door_ids, row_ids, n_doors, max_rows
