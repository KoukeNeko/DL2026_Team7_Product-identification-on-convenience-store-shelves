"""End-to-end recognition: detect (clean) -> L1 category centroid + L2 brand probe.

L1: nearest of 4 per-category centroids (locked classifier, unknown if maxsim<0.65).
L2: trained linear brand probe (emb/l2_probe.npz); softmax, unknown if max prob < L2_MIN.
Renders boxes coloured by L1 with the L2 brand label (CJK font). English L1, CJK brand.
"""
import os
import sys
from collections import Counter

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
from transformers import ChineseCLIPModel, ChineseCLIPProcessor

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from detection_postprocess import detect_products
from ultralytics import YOLO

_ROOT = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.environ.get("MODELS_DIR", os.path.join(_ROOT, "models"))
DATA_DIR = os.environ.get("DATA_DIR", os.path.join(_ROOT, "data"))
FONT = os.path.join(MODELS_DIR, "NotoSansTC.otf")
DET_MODEL = os.path.join(MODELS_DIR, "yolov8l_sku110k_finetuned.pt")
MODEL_ID = "OFA-Sys/chinese-clip-vit-base-patch16"
CLASSES = ["飲料", "泡麵", "零食", "鮮食"]
L1_EN = {"飲料": "Drink", "泡麵": "Noodle", "零食": "Snack", "鮮食": "Fresh", "unknown": "unknown"}
L1_COLOR = {"飲料": (70, 150, 255), "泡麵": (255, 150, 40), "零食": (60, 220, 90),
            "鮮食": (235, 110, 200), "unknown": (150, 150, 150)}
L1_MIN, L2_MIN, DEVICE = 0.65, 0.45, "cuda"  # brand_l1-driven L1 + L2_MIN 0.45: drop broken centroid, recover correct-but-shy brands
# Water guard: clear/transparent bottles are visually indistinguishable to CLIP
# (water embeds next to light sports drinks -> confident-wrong, e.g. 寶礦力 p0.77).
# A low-saturation crop with a low-confidence brand is almost certainly clear liquid;
# label it 水 (honest) instead of a wrong brand. High-conf coloured items (silver
# Monster sat0.14/p0.97) stay untouched because the prob gate excludes them.
WATER_SAT_MAX, WATER_PROB_MAX, WATER_LABEL = 0.12, 0.55, "水"
# A very confident brand corrects an uncertain/wrong L1 centroid (e.g. 阿薩姆 p0.96 on a
# crop the centroid scored 鮮食<0.65 -> would otherwise be gated to unknown).
L2_OVERRIDE = 0.75
# 鮮食 brands (關東煮/雞胸肉/御飯糰…) are a visual magnet on drink & snack shelves: their
# logits over-fire on unrelated packaging. Demand a higher confidence for a 鮮食 brand to be
# accepted, and below it prefer the best non-鮮食 brand before giving up.
FRESH_L1 = "鮮食"
FRESH_MIN = 0.62

_model = ChineseCLIPModel.from_pretrained(MODEL_ID).to(DEVICE).eval()
_proc = ChineseCLIPProcessor.from_pretrained(MODEL_ID)


def encode(images):
    out = []
    with torch.no_grad():
        for s in range(0, len(images), 64):
            inp = _proc(images=images[s:s + 64], return_tensors="pt").to(DEVICE)
            f = _model.get_image_features(**inp)
            out.append((f / f.norm(dim=-1, keepdim=True)).cpu().numpy())
    return np.concatenate(out, 0).astype(np.float32)


def l1_centroids():
    # allow_pickle: self-generated npz (trusted).
    g = np.load(os.path.join(MODELS_DIR, "gallery_cnclip_clean.npz"), allow_pickle=True)
    feats, labels = g["feats"].astype(np.float32), g["L1"].astype(str)
    return np.stack([_unit(feats[labels == c].mean(0)) for c in CLASSES])


def load_l2_probe():
    p = np.load(os.path.join(MODELS_DIR, "l2_probe.npz"), allow_pickle=True)  # self-generated
    brands = p["brands"].astype(str)
    brands_l1 = p["brands_l1"].astype(str) if "brands_l1" in p else np.array([""] * len(brands))
    return p["W"].astype(np.float32), p["b"].astype(np.float32), brands, brands_l1


def _unit(v):
    return v / np.linalg.norm(v)


def _mean_saturation(crop):
    hsv = np.asarray(crop.convert("HSV"), dtype=np.float32)
    return float(hsv[:, :, 1].mean() / 255.0)


def recognize(crops, cents, W, bias, brands, brands_l1):
    feats = encode(crops)
    l1_sim = feats @ cents.T
    logits = feats @ W.T + bias
    logits = logits - logits.max(1, keepdims=True)
    prob = np.exp(logits); prob /= prob.sum(1, keepdims=True)
    out = []
    for i in range(len(crops)):
        # L1 centroid is near-useless (4 classes near-collinear; 鮮食 is a magnet for
        # seafood/food packaging, e.g. 蝦味先 -> 鮮食) -> trust the discriminative probe's
        # brand AND adopt its own category.
        k = int(prob[i].argmax())
        # suppress the 鮮食 magnet: if the top brand is 鮮食 but not strongly confident,
        # prefer the best non-鮮食 brand; if none clears L2_MIN the crop stays 鮮食 and is
        # gated to unknown below (a drink shelf rarely actually holds 鮮食).
        if brands_l1[k] == FRESH_L1 and float(prob[i, k]) < FRESH_MIN:
            alt = [j for j in range(len(brands)) if brands_l1[j] != FRESH_L1]
            if alt:
                ka = alt[int(np.argmax(prob[i, alt]))]
                if float(prob[i, ka]) >= L2_MIN:
                    k = ka
        max_prob = float(prob[i, k])
        brand_l1 = brands_l1[k]
        accept = max_prob >= L2_MIN and not (brand_l1 == FRESH_L1 and max_prob < FRESH_MIN)
        if accept:
            l1, l2 = (brand_l1 if brand_l1 else "unknown"), brands[k]
        else:
            l1, l2 = "unknown", "?"
            # clear bottle: low-confidence drink-brand guess + low saturation -> 水 (honest)
            if brand_l1 == "飲料" and _mean_saturation(crops[i]) < WATER_SAT_MAX:
                l1, l2 = "飲料", WATER_LABEL
        out.append((l1, l2))
    return out


def main():
    name = sys.argv[1] if len(sys.argv) > 1 else "7eleven_0145.jpg"
    image = Image.open(os.path.join(DATA_DIR, "samples", name)).convert("RGB")
    W_img, H_img = image.size
    cents = l1_centroids()
    Wp, bp, brands, brands_l1 = load_l2_probe()
    boxes, _ = detect_products(YOLO(DET_MODEL), image)
    preds = recognize([image.crop(tuple(b.astype(int))) for b in boxes], cents, Wp, bp, brands, brands_l1)

    draw = ImageDraw.Draw(image)
    font = ImageFont.truetype(FONT, max(13, W_img // 95))
    width = max(2, W_img // 360)
    for (x1, y1, x2, y2), (l1, l2) in zip(boxes.astype(int), preds):
        color = L1_COLOR[l1]
        draw.rectangle([x1, y1, x2, y2], outline=color, width=width)
        label = f"{L1_EN[l1]}/{l2}"
        draw.rectangle([x1, y1, x1 + len(label) * (W_img // 120) + 6, y1 + W_img // 62], fill=color)
        draw.text((x1 + 3, y1 + 1), label, fill=(255, 255, 255), font=font)

    l1c = Counter(p[0] for p in preds)
    bar = max(34, W_img // 28)
    canvas = Image.new("RGB", (W_img, H_img + bar), (18, 18, 18))
    canvas.paste(image, (0, bar))
    summ = "  ".join(f"{L1_EN[c]}:{l1c.get(c, 0)}" for c in CLASSES) + f"  unknown:{l1c.get('unknown', 0)}"
    ImageDraw.Draw(canvas).text((8, 6), f"E2E  {name}  total {len(boxes)}   {summ}", fill=(255, 255, 0), font=font)
    canvas.save("e2e_reco.jpg", quality=90)   # demo visualisation written to cwd
    print(f"{name}: detected {len(boxes)} | L1 {dict(l1c)} (viz -> e2e_reco.jpg)")
    print(f"L2 brands: {Counter(p[1] for p in preds).most_common(12)}")


if __name__ == "__main__":
    main()
