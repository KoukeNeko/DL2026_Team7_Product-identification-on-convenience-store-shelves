"""Demo web backend: upload a shelf photo -> run the full recognition pipeline ->
return structured results (items with door/row/col, KPIs) for the 3D shelf + scan UI.
Reuses the deployed virtual_shelf / e2e_recognition pipeline (YOLO + Chinese-CLIP + M-LSD)."""
import io, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from collections import Counter, defaultdict
import numpy as np
from PIL import Image
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

import e2e_recognition as e2e
import virtual_shelf as vs
import rowlib  # perspective-robust row grouping (sequential RANSAC); falls back to vs.cluster_rows
import vlm_rerank  # Gemini 3.5 Flash gated re-rank of unknown crops (no-op without GEMINI_KEY)
import struct_seg  # v2 YOLOv8-seg: shelf mask bands + door masks for the overlay + placement
import shelf_assign  # place products on the seg shelves: doors -> bays, bands -> rows

app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])
_M = {}

# Low-confidence pruning: only an unknown that ALSO looks like junk is dropped —
# a real (full-size, decently-scored) bottle is kept as a grey "unidentified facing" so the
# planogram count stays honest even when the brand is unreadable (glass glare).
JUNK_AREA_FRAC = 0.33   # smaller than this * the median facing -> likely a fragment / background bit
JUNK_SCORE = 0.28       # ...or a near-detection-floor score -> likely background junk


def models():
    if "m" not in _M:
        _M["m"] = vs.load_models()
    return _M["m"]


@app.on_event("startup")
def warm():
    models()
    struct_seg.load()       # preload the seg model so the first upload isn't slow
    print("models loaded, ready")


@app.post("/recognize")
async def recognize(file: UploadFile = File(...)):
    raw = await file.read()
    img = Image.open(io.BytesIO(raw)).convert("RGB")
    try:                                              # save upload for offline debugging
        import time
        updir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "uploads")
        os.makedirs(updir, exist_ok=True)
        safe = (file.filename or "shelf.jpg").replace("/", "_")
        open(os.path.join(updir, f"{int(time.time())}_{safe}"), "wb").write(raw)
    except Exception:
        pass
    width, height = img.size
    seg_shelf, seg_doors = struct_seg.detect(img)     # mask bands (shelf) + masks (door) for the overlay
    det, cents, Wp, bp, brands, brands_l1, mlsd = models()
    boxes, scores = e2e.detect_products(det, img)
    boxes = boxes.astype(int)
    valid = (boxes[:, 2] > boxes[:, 0]) & (boxes[:, 3] > boxes[:, 1])
    boxes, scores = boxes[valid], scores[valid]
    if len(boxes) == 0:
        return {"w": width, "h": height, "doors": 0, "max_rows": 0, "frames": [], "items": [],
                "seg_shelf": seg_shelf, "seg_doors": seg_doors,
                "kpi": {"facings": 0, "identified": 0, "id_pct": 0, "l1": {}, "top": []},
                "colors": {k: list(e2e.L1_COLOR[k]) for k in e2e.CLASSES}, "en": e2e.L1_EN}
    crops = [img.crop(tuple(b)) for b in boxes]
    preds = e2e.recognize(crops, cents, Wp, bp, brands, brands_l1)
    logits = e2e.encode(crops) @ Wp.T + bp                  # L2 softmax max prob = confidence
    logits = logits - logits.max(1, keepdims=True)
    prob = np.exp(logits); prob /= prob.sum(1, keepdims=True)
    confs = prob.max(1)
    brand2l1 = dict(zip(brands, brands_l1))
    _unk = [i for i in range(len(boxes)) if preds[i][1] == "?"]
    # 壓鮮食誤觸: keep 鮮食 brands out of the VLM candidate pool so the rerank can't
    # reintroduce 關東煮-style fresh-food brands onto a drink shelf.
    _jobs = [(i, [brands[j] for j in np.argsort(prob[i])[::-1] if brands_l1[j] != "鮮食"][:5]) for i in _unk]
    vlm_map, vlm_ok = vlm_rerank.rerank(crops, _jobs)  # map + whether Gemini was reachable (quota/net)
    # PLACE products on the seg structure: door masks -> bays, shelf bands -> rows.
    door_ids, row_ids, _, _ = shelf_assign.assign(boxes, seg_shelf, seg_doors, width, height)
    box_areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    median_area = float(np.median(box_areas)) if len(box_areas) else 0.0
    cells = defaultdict(list)
    for i in range(len(boxes)):
        if door_ids[i] == -1:                                   # far-outside background -> drop
            continue
        cells[(door_ids[i], row_ids[i])].append(i)
    items = []
    for (door, row), idxs in cells.items():
        idxs.sort(key=lambda i: boxes[i][0] + boxes[i][2])      # left -> right within the row
        built = []
        for gi in idxs:
            l1c, l2c, src = preds[gi][0], preds[gi][1], "clip"
            if l2c == "?" and gi in vlm_map:                    # VLM pass on an unrecognised crop
                _rec = vlm_map[gi]
                if _rec.get("brand"):                           # recovered a brand from the probe's top-5
                    l2c = _rec["brand"]; l1c = brand2l1.get(l2c) or l1c; src = "vlm"
                elif _rec.get("l1") and _rec["l1"] != "鮮食":   # category fallback, never 鮮食
                    l1c = _rec["l1"]; src = "vlm_l1"
            if l1c == "unknown" and vlm_ok:                     # Gemini ran and still couldn't ID it
                area = (boxes[gi][2] - boxes[gi][0]) * (boxes[gi][3] - boxes[gi][1])
                if area < JUNK_AREA_FRAC * median_area or scores[gi] < JUNK_SCORE:
                    continue                                     # small / low-score -> junk, drop it
                # otherwise a real bottle whose brand is unreadable -> keep as a grey facing
            built.append((gi, l1c, l2c, src))
        for ci, (gi, l1c, l2c, src) in enumerate(built):
            x1, y1, x2, y2 = boxes[gi]
            items.append({"x1": int(x1), "y1": int(y1), "x2": int(x2), "y2": int(y2),
                          "l1": l1c, "l2": l2c, "conf": round(float(confs[gi]) * 100), "src": src,
                          "door": int(door), "row": int(row), "col": ci, "rowlen": len(built),
                          "back": False})
    # drop empty bays: remap door ids to the bays that actually hold products
    present = sorted({it["door"] for it in items})
    remap = {d: i for i, d in enumerate(present)}
    for it in items:
        it["door"] = remap[it["door"]]
    kept = [(it["l1"], it["l2"]) for it in items]
    idn = sum(1 for _, l2 in kept if l2 != "?")
    kpi = {"facings": len(items), "identified": idn,
           "id_pct": round(100 * idn / max(1, len(items))),
           "vlm": sum(1 for it in items if it.get("src") == "vlm"),
           "vlm_l1": sum(1 for it in items if it.get("src") == "vlm_l1"),
           "l1": dict(Counter(l1 for l1, _ in kept)),
           "top": Counter(l2 for _, l2 in kept if l2 not in ("?", "水")).most_common(8)}
    return {"w": width, "h": height, "doors": max(1, len(present)),
            "max_rows": max((it["row"] for it in items), default=0) + 1,
            "frames": [], "row_lines": [],
            "seg_shelf": seg_shelf, "seg_doors": seg_doors,
            "items": items, "kpi": kpi,
            "colors": {k: list(e2e.L1_COLOR[k]) for k in e2e.CLASSES}, "en": e2e.L1_EN}


_here = os.path.dirname(os.path.abspath(__file__))
app.mount("/", StaticFiles(directory=os.path.join(_here, "static"), html=True), name="static")
