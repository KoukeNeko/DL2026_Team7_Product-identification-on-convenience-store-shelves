"""Virtual-shelf + counting demo: a messy shelf photo -> a clean 'understood' planogram.

Pipeline: detect (locked YOLO) -> L1 category + L2 brand (L1-gated probe + water guard)
-> cluster detections into shelf rows -> render a clean schematic planogram (per-category
colours, brand labels) beside the real photo, with KPI counts (facings, per-category,
identified %, top brands). This is the "shelf understanding" deliverable, not box-drawing.
"""
import os
import sys
from collections import Counter

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from ultralytics import YOLO

import e2e_recognition as e2e

NAME = sys.argv[1] if len(sys.argv) > 1 else "family_new2.jpg"
OUT = sys.argv[2] if len(sys.argv) > 2 else "virtual_shelf.jpg"

PANEL_W = 980
ROW_H = 92
CELL_GAP = 6
PAD = 26
HEADER_H = 168
BAR_H = 200
BG = (22, 24, 28)
FG = (238, 238, 238)
MUTED = (150, 150, 150)
ROW_TOL_FRAC = 0.6          # two boxes share a row if their shelf-coords are within 0.6*median-height
MIN_ROW_COVERAGE = 0.22     # a real shelf row's products span >= this fraction of width;
                            # sparser "rows" are header cards / background behind the rack -> dropped
MAX_SHELF_SLOPE = 0.6       # clamp shelf-baseline tilt (dy/dx ~= ±31°); guards against outliers
MIN_BOXES_FOR_SLOPE = 8     # too few boxes -> slope estimate unreliable, assume horizontal
SAME_SHELF_DY = 0.5         # a pair is "same shelf" if |dy| < this * median product height
NEAR_X_MIN = 0.03           # ignore pairs closer than this (fraction of x-span) -> dy/dx noise
NEAR_X_MAX = 0.30           # ignore pairs farther than this -> perspective slope isn't constant
SLOPE_DEADZONE = 0.05       # |slope| below this (~3°) is treated as head-on (no shear correction)
BACK_ROW_FRAC = 0.5         # a row whose median box-height < this * the shelf's full-row height is an
                            # occluded BACK row (only product caps peek over the front) → merged forward
BACK_ROW_NEIGHBOR = 1.5     # ...AND both vertically-adjacent rows must be >= this * its height (the
                            # alternating front/back signature) — rejects a monotonic PERSPECTIVE gradient
                            # where lower shelves are simply shorter in the image but are real rows
BAY_BANDS = 80              # column resolution for door/bay detection
BAY_COV_MAX = 0.35          # a frame column carries product coverage below this (no products on it)
BAY_EDGE_MIN = 0.45         # ...AND a full-height vertical edge above this (the physical frame)
BAY_MARGIN = 0.12           # ignore frame hits within this fraction of the left/right edge
BAY_MIN_WIDTH = 0.10        # merge bays narrower than this fraction of the image width
BAY_SHEAR_MAX = 0.5         # widest x-shear searched for tilted door frames (oblique shots)
BAY_SHEAR_STEPS = 21        # shear-search resolution over [-BAY_SHEAR_MAX, +BAY_SHEAR_MAX]
BAY_OBLIQUE_MIN = 0.10      # only search a frame shear when the shelf is clearly oblique (~6°)
MIN_BAY_BOXES = 4           # a real door holds >= this many products; sparser "doors" (edge
                            # slivers / walls) get merged into the neighbour
MIN_BAY_FRAC = 0.35         # also drop a door tiny vs its siblings (e.g. 9 vs 98 = spurious edge split)
# --- M-LSD deep door-frame detection (the root-fix path used by the main planogram) ---
FRAME_VERT_ANG = 62         # a door-frame line is within (90 − this)° of vertical
FRAME_MIN_H = 0.50          # ...and spans >= this fraction of the image height
FRAME_CLUSTER_X = 0.05      # merge frame lines whose mid-x are within this fraction of width
FRAME_EDGE_MARGIN = 0.08    # ignore frames within this fraction of the left/right edge
DOOR_GAP = 14               # gap between door columns in the planogram
DOOR_LABEL_H = 30           # height reserved for each door's "Door k" label

_FONTS = {}
_MLSD = {}


def font(size):
    if size not in _FONTS:
        _FONTS[size] = ImageFont.truetype(e2e.FONT, size)
    return _FONTS[size]


def _shelf_baseline_slope(cx, cy, median_h):
    """Estimate the shelf-baseline slope (dy/dx) from LOCAL neighbours. Two products on the
    same shelf sit at nearly the same height (|dy| small) but different x; in an oblique
    shot that pair is tilted by the perspective angle. We take the median slope over all
    nearby same-shelf pairs — robust, and ≈0 for a head-on shot (so upright shelves are
    untouched). Projecting to cy - m*cx then straightens the rows before clustering."""
    n = len(cx)
    if n < MIN_BOXES_FOR_SLOPE:
        return 0.0
    x_span = float(cx.max() - cx.min())
    if x_span < 1e-6:
        return 0.0
    dx = cx[None, :] - cx[:, None]                 # dx[i, j] = cx[j] - cx[i]
    dy = cy[None, :] - cy[:, None]
    same_shelf = (dx > NEAR_X_MIN * x_span) & (dx < NEAR_X_MAX * x_span) \
        & (np.abs(dy) < median_h * SAME_SHELF_DY)  # nearby, to the right, ~same height
    if not same_shelf.any():
        return 0.0
    slope = float(np.median(dy[same_shelf] / dx[same_shelf]))
    if abs(slope) < SLOPE_DEADZONE:
        return 0.0                                 # head-on shot — don't let estimate noise shear it
    return float(np.clip(slope, -MAX_SHELF_SLOPE, MAX_SHELF_SLOPE))


def cluster_rows(boxes):
    cx = (boxes[:, 0] + boxes[:, 2]) / 2.0
    cy = (boxes[:, 1] + boxes[:, 3]) / 2.0
    median_h = float(np.median(boxes[:, 3] - boxes[:, 1]))
    coord = cy - _shelf_baseline_slope(cx, cy, median_h) * cx   # perspective-compensated shelf coord
    order = list(np.argsort(coord))
    rows, current = [], [order[0]]
    for idx in order[1:]:
        if coord[idx] - coord[current[-1]] <= median_h * ROW_TOL_FRAC:
            current.append(idx)
        else:
            rows.append(current); current = [idx]
    rows.append(current)
    return [sorted(r, key=lambda i: float(boxes[i, 0])) for r in rows]


def filter_sparse_rows(rows, boxes, img_w):
    """Drop rows whose products span too little width — promo header cards and
    background products behind a display rack, not real shelf levels. Keeps the
    rack's actual levels (which span the width). Falls back to all rows if every
    row would be dropped (e.g. a genuinely sparse shelf)."""
    kept = [r for r in rows
            if sum(int(boxes[i][2] - boxes[i][0]) for i in r) >= MIN_ROW_COVERAGE * img_w]
    return kept if kept else rows


def merge_back_rows(rows, boxes):
    """On open multi-deep display stands shot from above, one physical shelf shows a FRONT
    row (full products) plus a BACK row whose products are occluded — only their caps peek
    over the front, so those boxes are much SHORTER, AND a back row sits between two full rows.
    cluster_rows then splits one shelf into two rows (a 4-shelf stand reads as ~7). Detect such
    short, SANDWICHED rows and merge each into the nearest full row, recovering the true shelf
    count. The sandwiched test (both neighbours much taller = the alternating front/back
    signature) is what separates a real back row from a row that is merely shorter because of
    PERSPECTIVE (a steeply-angled shot makes lower shelves shorter in the image, monotonically —
    those are real rows, not back rows). Self-gating: ordinary shelves merge nothing.
    Returns (rows, back_idx) where back_idx is the set of box indices marked back-row."""
    if len(rows) < 3:
        return rows, set()
    stats = []
    for r in rows:
        cy = float(np.median([(boxes[i][1] + boxes[i][3]) / 2.0 for i in r]))
        h = float(np.median([boxes[i][3] - boxes[i][1] for i in r]))
        stats.append({"cy": cy, "h": h, "idx": list(r)})
    stats.sort(key=lambda s: s["cy"])
    ref_h = float(np.percentile([s["h"] for s in stats], 70))      # robust full-row height
    is_back = [False] * len(stats)
    for i in range(1, len(stats) - 1):                             # interior rows only
        h = stats[i]["h"]
        if (h < BACK_ROW_FRAC * ref_h
                and stats[i - 1]["h"] >= BACK_ROW_NEIGHBOR * h
                and stats[i + 1]["h"] >= BACK_ROW_NEIGHBOR * h):
            is_back[i] = True
    if not any(is_back):
        return rows, set()
    full = [s for i, s in enumerate(stats) if not is_back[i]]
    back_idx = set()
    for i, s in enumerate(stats):
        if is_back[i]:
            nearest = min(full, key=lambda f: abs(f["cy"] - s["cy"]))
            nearest["idx"].extend(s["idx"])
            back_idx.update(s["idx"])
    for s in full:
        s["idx"].sort(key=lambda i: float(boxes[i][0]))            # left-to-right within merged row
    full.sort(key=lambda s: s["cy"])
    return [s["idx"] for s in full], back_idx


def _bay_frames(boxes, gray, img_w, shear):
    """Door-frame boundary fractions (sheared-x / img_w) for one horizontal shear. A frame
    column BOTH carries no products (low coverage) AND a strong full-height vertical edge.
    Shearing x by `shear`·(y − H/2) straightens tilted frames (oblique shots) before
    profiling; shear=0 is the head-on case (vertical frames)."""
    height, width = gray.shape
    if shear:
        m = np.float32([[1, shear, -shear * height / 2], [0, 1, 0]])
        gray = cv2.warpAffine(gray, m, (width, height), borderValue=255)
    coverage = np.zeros(BAY_BANDS)
    for box in boxes:
        dx = shear * ((box[1] + box[3]) / 2 - height / 2)
        lo = int((box[0] + dx) / img_w * BAY_BANDS)
        hi = int(np.ceil((box[2] + dx) / img_w * BAY_BANDS))
        coverage[max(0, lo):max(0, hi)] += 1
    coverage = coverage / max(1.0, coverage.max())

    vert = np.abs(cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3))
    full_height_edge = (vert > vert.mean() + vert.std()).mean(0)
    edge = np.array([full_height_edge[int(i * width / BAY_BANDS):int((i + 1) * width / BAY_BANDS)].mean()
                     for i in range(BAY_BANDS)])
    edge = (edge - edge.min()) / (edge.max() - edge.min() + 1e-9)

    is_frame = (coverage < BAY_COV_MAX) & (edge > BAY_EDGE_MIN)
    cuts, i = [], 0
    while i < BAY_BANDS:
        if is_frame[i]:
            j = i
            while j < BAY_BANDS and is_frame[j]:
                j += 1
            x = (i + j) / 2 / BAY_BANDS
            if BAY_MARGIN < x < 1 - BAY_MARGIN and (not cuts or x - cuts[-1] > BAY_MIN_WIDTH):
                cuts.append(x)
            i = j
        else:
            i += 1
    return cuts


def detect_bays(boxes, gray, img_w, shelf_slope=0.0):
    """Split a multi-door cooler into vertical bays (doors). Returns (shear, cut_fractions)
    where cuts are boundaries on the sheared x-axis. Head-on shelves (|shelf_slope| <=
    deadzone) use shear=0 (vertical frames). For an oblique shelf the frames are tilted, so
    we search a shear that straightens them and keep the one revealing the most frames
    (tie → smallest shear). Gating on obliqueness stops the search inventing spurious frames
    on a head-on single unit."""
    if abs(shelf_slope) <= BAY_OBLIQUE_MIN:
        return 0.0, _bay_frames(boxes, gray, img_w, 0.0)
    best_shear, best_cuts = 0.0, _bay_frames(boxes, gray, img_w, 0.0)
    for shear in np.linspace(-BAY_SHEAR_MAX, BAY_SHEAR_MAX, BAY_SHEAR_STEPS):
        cuts = _bay_frames(boxes, gray, img_w, float(shear))
        if (len(cuts), -abs(shear)) > (len(best_cuts), -abs(best_shear)):
            best_shear, best_cuts = float(shear), cuts
    return best_shear, best_cuts


def _drop_sliver_cuts(cuts, cx_sheared, img_w):
    """Remove boundaries that carve off an under-populated bay (cooler edge / wall sliver
    with almost no products). Repeatedly drops the boundary next to the smallest such bay,
    merging it into a neighbour, until every remaining bay holds >= MIN_BAY_BOXES."""
    cuts = list(cuts)
    while cuts:
        bounds = [-1e9] + [c * img_w for c in cuts] + [1e9]
        counts = [int(((cx_sheared >= bounds[b]) & (cx_sheared < bounds[b + 1])).sum())
                  for b in range(len(bounds) - 1)]
        small = [b for b, n in enumerate(counts) if n < MIN_BAY_BOXES]
        if not small:
            break
        b = min(small, key=lambda idx: counts[idx])
        if b == 0:
            cuts.pop(0)
        elif b == len(counts) - 1:
            cuts.pop(-1)
        else:                                      # merge into the larger neighbour
            cuts.pop(b - 1 if counts[b - 1] >= counts[b + 1] else b)
    return cuts


def cluster_rows_in_bays(boxes, gray, img_w):
    """Per-bay row clustering: detect doors (shear-aware for oblique shots), then cluster
    rows inside each door independently so different doors may have different shelf counts.
    Returns (shear, cuts, [(bay_index, rows)]) with rows holding indices into `boxes`."""
    cx = (boxes[:, 0] + boxes[:, 2]) / 2.0
    cy = (boxes[:, 1] + boxes[:, 3]) / 2.0
    median_h = float(np.median(boxes[:, 3] - boxes[:, 1]))
    shelf_slope = _shelf_baseline_slope(cx, cy, median_h)
    shear, cuts = detect_bays(boxes, gray, img_w, shelf_slope)

    cx_sheared = cx + shear * (cy - gray.shape[0] / 2)        # straighten frames before assigning
    cuts = _drop_sliver_cuts(cuts, cx_sheared, img_w)
    bounds = [-1e9] + [c * img_w for c in cuts] + [1e9]
    bays = []
    for bay_index in range(len(bounds) - 1):
        lo, hi = bounds[bay_index], bounds[bay_index + 1]
        members = [i for i in range(len(boxes)) if lo <= cx_sheared[i] < hi]
        if not members:
            continue
        mem_boxes = boxes[members]
        bay_w = float(mem_boxes[:, 2].max() - mem_boxes[:, 0].min())   # this door's own width
        sub_rows = filter_sparse_rows(cluster_rows(mem_boxes), mem_boxes, bay_w)
        bays.append((bay_index, [[members[k] for k in row] for row in sub_rows]))
    return shear, cuts, bays


def load_mlsd():
    """Load the M-LSD deep line-segment detector once (weights auto-download from HF).
    Returns None if controlnet_aux/weights are unavailable, so callers degrade to 1 door."""
    if "model" not in _MLSD:
        try:
            from controlnet_aux import MLSDdetector
            model = MLSDdetector.from_pretrained("lllyasviel/Annotators").model
            try:
                model = model.cuda()
            except Exception:
                pass
            _MLSD["model"] = model.eval()
        except Exception:
            _MLSD["model"] = None
    return _MLSD["model"]


def detect_door_frames(image, mlsd_model):
    """Find real door frames with M-LSD: long near-vertical line segments, each kept with
    its REAL angle (extended to x@y=0 and x@y=H), clustered to one line per frame. Returns
    [(x_top, x_bot)] for interior frames; [] for a single-unit shelf or if M-LSD is absent."""
    if mlsd_model is None:
        return []
    try:
        from controlnet_aux.mlsd.utils import pred_lines
    except Exception:
        return []
    width, height = image.size
    lines = np.array(pred_lines(np.asarray(image), mlsd_model, [512, 512], 0.1, 20), dtype=float)
    if len(lines) == 0:
        return []
    if lines[:, [0, 2]].max() <= 520:                       # lines came back in 512-space
        lines[:, [0, 2]] *= width / 512.0
        lines[:, [1, 3]] *= height / 512.0
    cand = []
    for x1, y1, x2, y2 in lines:
        dx, dy = x2 - x1, y2 - y1
        if (dx * dx + dy * dy) ** 0.5 < FRAME_MIN_H * height or abs(dy) < 1:
            continue
        if abs(np.degrees(np.arctan2(dy, dx))) < FRAME_VERT_ANG:
            continue
        slope = dx / dy
        cand.append((x1 + slope * (0 - y1), x1 + slope * (height - y1)))
    cand.sort(key=lambda f: (f[0] + f[1]) / 2)
    clusters = []
    for f in cand:
        mid = (f[0] + f[1]) / 2
        if clusters and mid - (clusters[-1][-1][0] + clusters[-1][-1][1]) / 2 < FRAME_CLUSTER_X * width:
            clusters[-1].append(f)
        else:
            clusters.append([f])
    reps = [(float(np.median([c[0] for c in cl])), float(np.median([c[1] for c in cl]))) for cl in clusters]
    return [(xt, xb) for xt, xb in reps if FRAME_EDGE_MARGIN * width < (xt + xb) / 2 < (1 - FRAME_EDGE_MARGIN) * width]


def assign_doors(boxes, frames, img_h):
    """Assign each product to a door by which side of every SLANTED frame it falls on (the
    frame's x evaluated at the product's own y). Frames carving off an under-populated door
    (cooler edge / wall) are dropped. Returns (kept_frames, door_id_per_box)."""
    cx = (boxes[:, 0] + boxes[:, 2]) / 2.0
    cy = (boxes[:, 1] + boxes[:, 3]) / 2.0

    def door_ids(fr):
        return [sum(1 for xt, xb in fr if xt + (xb - xt) * (cy[i] / img_h) < cx[i])
                for i in range(len(boxes))]

    frames = list(frames)
    while frames:
        ids = door_ids(frames)
        counts = [ids.count(d) for d in range(len(frames) + 1)]
        avg = sum(counts) / len(counts)
        small = [d for d, n in enumerate(counts) if n < MIN_BAY_BOXES or n < MIN_BAY_FRAC * avg]
        if not small:
            break
        d = min(small, key=lambda idx: counts[idx])
        frames.pop(0 if d == 0 else -1 if d == len(counts) - 1 else (d - 1 if counts[d - 1] >= counts[d + 1] else d))
    return frames, door_ids(frames)


def fit_font(draw, text, max_w, start=22, floor=11):
    size = start
    while size > floor and draw.textlength(text, font=font(size)) > max_w - 8:
        size -= 1
    return font(size)


def draw_planogram_doors(doors, preds, height, name):
    """Door-aware planogram: doors laid out left-to-right as equal columns, each with its
    own shelf rows stacked top-down (so different doors can show different shelf counts).
    A single-door shelf is just one full-width column — same as the old flat planogram."""
    panel = Image.new("RGB", (PANEL_W, height), BG)
    draw = ImageDraw.Draw(panel)
    kept_preds = [preds[i] for rows in doors for row in rows for i in row]
    l1_counts = Counter(l1 for l1, _ in kept_preds)
    identified = [l2 for _, l2 in kept_preds if l2 != "?"]
    total = len(kept_preds)
    id_rate = 100 * len(identified) // max(1, total)

    draw.text((PAD, PAD), f"Virtual Shelf — {name}", fill=FG, font=font(30))
    draw.text((PAD, PAD + 42),
              f"{total} facings   identified {len(identified)} ({id_rate}%)   "
              f"{len(doors)} door{'s' if len(doors) != 1 else ''}",
              fill=MUTED, font=font(20))
    x, y = PAD, PAD + 84
    for category in e2e.CLASSES:
        draw.rectangle([x, y, x + 150, y + 36], fill=e2e.L1_COLOR[category])
        draw.text((x + 8, y + 7), f"{e2e.L1_EN[category]}: {l1_counts.get(category, 0)}",
                  fill=(255, 255, 255), font=font(18))
        x += 158

    n = max(1, len(doors))
    door_w = (PANEL_W - 2 * PAD - (n - 1) * DOOR_GAP) // n
    x = PAD
    for door_index, rows in enumerate(doors):
        if n > 1:
            draw.text((x + 2, HEADER_H), f"Door {door_index + 1}", fill=MUTED, font=font(16))
        y = HEADER_H + DOOR_LABEL_H
        for row in rows:
            count = len(row)
            gap = CELL_GAP if count <= 12 else 1
            cell_w = max(1, (door_w - (count - 1) * gap) // count)
            cx = x
            for i in row:
                l1, l2 = preds[i]
                draw.rectangle([cx, y, cx + cell_w, y + ROW_H], fill=e2e.L1_COLOR[l1], outline=(12, 12, 12))
                label = l2 if l2 != "?" else "·"
                face = fit_font(draw, label, cell_w)
                draw.text((cx + 3, y + ROW_H // 2 - face.size // 2), label, fill=(255, 255, 255), font=face)
                cx += cell_w + gap
            y += ROW_H + CELL_GAP
        if door_index < n - 1:
            sx = x + door_w + DOOR_GAP // 2
            draw.line([(sx, HEADER_H), (sx, height - PAD)], fill=(70, 70, 78), width=2)
        x += door_w + DOOR_GAP
    return panel


def draw_brand_bar(preds, width, height):
    panel = Image.new("RGB", (width, height), BG)
    draw = ImageDraw.Draw(panel)
    brand_counts = Counter(l2 for _, l2 in preds if l2 not in ("?", "水"))
    draw.text((PAD, PAD), "Top brands (facings)", fill=FG, font=font(22))
    top = brand_counts.most_common(8)
    if not top:
        return panel
    max_n = top[0][1]
    bar_x, bar_w = 170, width - 170 - PAD
    y = PAD + 40
    for brand, n in top:
        draw.text((PAD, y + 2), brand, fill=FG, font=font(18))
        draw.rectangle([bar_x, y, bar_x + int(bar_w * n / max_n), y + 22], fill=(70, 150, 255))
        draw.text((bar_x + int(bar_w * n / max_n) + 6, y + 2), str(n), fill=MUTED, font=font(18))
        y += 30
    return panel


def load_models():
    """Load detector + L1 centroids + L2 probe + M-LSD once (reuse across a batch)."""
    return (YOLO(e2e.DET_MODEL), e2e.l1_centroids(), *e2e.load_l2_probe(), load_mlsd())


def render_shelf(image, name, out_path, det, cents, Wp, bp, brands, brands_l1, mlsd_model=None):
    """Recognise one shelf photo and write its composite (photo + door-aware planogram +
    KPI). M-LSD detects real (slanted) door frames; products are assigned to doors and rows
    clustered per door. Models are passed in so a batch loads them only once."""
    boxes, _ = e2e.detect_products(det, image)
    boxes = boxes.astype(int)
    valid = (boxes[:, 2] > boxes[:, 0]) & (boxes[:, 3] > boxes[:, 1])   # drop degenerate boxes
    boxes = boxes[valid]
    if len(boxes) == 0:                                 # nothing detected -> photo-only output
        image.save(out_path, quality=90)
        return {"facings": 0, "rows": 0, "doors": 0, "id_pct": 0, "l1": {}, "top_brands": []}
    crops = [image.crop(tuple(b)) for b in boxes]
    preds = e2e.recognize(crops, cents, Wp, bp, brands, brands_l1)

    frames, door_ids = assign_doors(boxes, detect_door_frames(image, mlsd_model), image.size[1])
    doors = []                                          # one entry per door: its kept shelf rows
    for door in range(len(frames) + 1):
        members = [i for i in range(len(boxes)) if door_ids[i] == door]
        if not members:
            continue
        mem = boxes[members]
        bay_w = float(mem[:, 2].max() - mem[:, 0].min())
        sub_rows = filter_sparse_rows(cluster_rows(mem), mem, bay_w)
        doors.append([[members[k] for k in row] for row in sub_rows])
    kept = sorted({i for rows in doors for row in rows for i in row})

    max_rows = max((len(rows) for rows in doors), default=1)
    plan_h = HEADER_H + DOOR_LABEL_H + max_rows * (ROW_H + CELL_GAP) + PAD
    planogram = draw_planogram_doors(doors, preds, plan_h, name)
    brand_bar = draw_brand_bar([preds[i] for i in kept], PANEL_W, BAR_H)

    photo = image.copy()
    pdraw = ImageDraw.Draw(photo)
    line_w = max(2, photo.size[0] // 360)
    for i in kept:
        x1, y1, x2, y2 = boxes[i]
        pdraw.rectangle([x1, y1, x2, y2], outline=e2e.L1_COLOR[preds[i][0]], width=line_w)
    for x_top, x_bot in frames:                          # real (slanted) door separators
        pdraw.line([(x_top, 0), (x_bot, image.size[1])], fill=(255, 255, 255), width=line_w + 2)
    right_h = plan_h + BAR_H
    scale = right_h / photo.size[1]
    photo = photo.resize((int(photo.size[0] * scale), right_h))

    composite = Image.new("RGB", (photo.size[0] + PANEL_W, right_h), BG)
    composite.paste(photo, (0, 0))
    composite.paste(planogram, (photo.size[0], 0))
    composite.paste(brand_bar, (photo.size[0], plan_h))
    composite.save(out_path, quality=90)

    l1_counts = Counter(preds[i][0] for i in kept)
    brand_counts = Counter(preds[i][1] for i in kept if preds[i][1] not in ("?", "水"))
    identified = sum(1 for i in kept if preds[i][1] != "?")
    id_pct = round(100 * identified / max(1, len(kept)))
    return {"facings": len(kept), "rows": sum(len(rows) for rows in doors), "doors": len(doors),
            "id_pct": id_pct, "l1": dict(l1_counts), "top_brands": brand_counts.most_common(10)}


def main():
    image = Image.open(os.path.join(os.environ.get("DATA_DIR", "data"), "samples", NAME)).convert("RGB")
    models = load_models()
    stats = render_shelf(image, NAME, OUT, *models)
    print(f"saved {OUT}")
    print(f"{NAME}: {stats['facings']} facings | {stats['doors']} doors | rows {stats['rows']} | L1 {stats['l1']}")
    print(f"top brands: {stats['top_brands']}")


if __name__ == "__main__":
    main()
