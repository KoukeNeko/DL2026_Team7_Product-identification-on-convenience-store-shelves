"""Train the DEPLOYABLE L2 brand probe (Chinese-CLIP features -> linear softmax).

Broad, de-fragmented data + per-brand L1 tag so the e2e pipeline can gate L2 by L1
(a 鮮食 category only shows on a 鮮食 crop, a drink brand only on a 飲料 crop) — this
kills cross-category catch-all leak (e.g. 關東煮 leaking onto a drink shelf).

Sources
  clean product images : data_clean/ref_ecommerce/{飲料,泡麵,零食}/{brand}/
  web supplements      : data_clean/ref_web/{飲料,泡麵,零食}/{brand}/
  7-ELEVEN fresh food  : data_clean/ref_7fresh/{鮮食}/{category}/
  original real crops   : cropset/labels.json   (+ cropset/crops.json)
  new mined crops       : cropset2/labels.json  (+ cropset2/crops/)

Every brand name is canonicalised through canon_map.json before counting.
Saves W, b, brands, brands_l1 to emb/l2_probe.npz.
"""
import glob
import json
import os
import sys
from collections import Counter, defaultdict

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from transformers import ChineseCLIPModel, ChineseCLIPProcessor

HOME = os.environ.get("DATA_DIR", "data")
EC = f"{HOME}/data_clean/ref_ecommerce"
WEB = f"{HOME}/data_clean/ref_web"
FRESH = f"{HOME}/data_clean/ref_7fresh"
MANUAL = f"{HOME}/data_clean/ref_manual"   # 手動抓圖 (crop_tool.html output, by brand)
DS2 = f"{HOME}/data_clean/ds2"             # 資料集2 (classmates' latest product imgs by source/cat/brand)
DS2_BOXED = f"{HOME}/data_clean/ds2_boxed"   # per-box crops from annotation tool (L1/brand)
CROPS1 = f"{HOME}/cropset/crops"
LABELS1 = f"{HOME}/cropset/labels.json"
IDMAP1 = f"{HOME}/cropset/crops.json"
CROPS2 = f"{HOME}/cropset2/crops"
LABELS2 = f"{HOME}/cropset2/labels.json"
CANON_PATH = f"{HOME}/cropset2/canon_map.json"
OUT = f"{HOME}/demo/emb/l2_probe.npz"
BACKUP = f"{HOME}/demo/emb/l2_probe_BAK.npz"

MODEL_ID = "OFA-Sys/chinese-clip-vit-base-patch16"
DEVICE = "cuda"
IMG_EXT = (".jpg", ".jpeg", ".png", ".webp")
DEL_PREFIX = "del_"  # soft-deleted files (annotation tool) excluded from training
UNK = {"", "?", "？", "unknown", "Unknown", "未辨識", "未知"}
EXCLUDE = {"全家", "關東煮"}   # 關東煮 over-fires on drink shelves -> drop the class
DROP_L1 = {"其他"}          # non-food crops -> not a product class
MIN_TOTAL = 3
DRINK_MIN = 2               # drinks admitted lower (demo shelves are drinks)
NEW_OK_CONF = {"high", "med"}


def load_canon():
    if os.path.exists(CANON_PATH):
        return json.load(open(CANON_PATH, encoding="utf-8"))
    return {}


def canon(brand, cmap):
    b = brand.strip()
    return cmap.get(b, cmap.get(b.lower(), b))


def load_folder(root, cats, cmap):
    """Load a {cat}/{brand}/ image tree; the L1 of every sample is its top cat."""
    paths, brands, l1s = [], [], []
    for cat in cats:
        base = f"{root}/{cat}"
        if not os.path.isdir(base):
            continue
        for brand in sorted(os.listdir(base)):
            bd = f"{base}/{brand}"
            if not os.path.isdir(bd) or brand.startswith("_"):
                continue
            for p in glob.glob(f"{bd}/*"):
                if p.lower().endswith(IMG_EXT) and not os.path.basename(p).startswith(DEL_PREFIX):
                    paths.append(p); brands.append(canon(brand, cmap)); l1s.append(cat)
    return paths, brands, l1s


def load_crops1(cmap):
    labels = json.load(open(LABELS1, encoding="utf-8"))
    idmap = {c["id"]: c["file"] for c in json.load(open(IDMAP1, encoding="utf-8"))}
    paths, brands, l1s = [], [], []
    for lab in labels:
        b = (lab.get("brand") or "").strip()
        if b in UNK or lab["id"] not in idmap or lab.get("l1") in DROP_L1:
            continue
        paths.append(f"{CROPS1}/{idmap[lab['id']]}")
        brands.append(canon(b, cmap)); l1s.append(lab.get("l1", "飲料"))
    return paths, brands, l1s


def load_crops2(cmap):
    if not os.path.exists(LABELS2):
        return [], [], []
    labels = json.load(open(LABELS2, encoding="utf-8"))
    paths, brands, l1s = [], [], []
    for lab in labels:
        b = (lab.get("brand") or "").strip()
        if (not lab.get("readable")) or b in UNK or lab.get("conf") not in NEW_OK_CONF:
            continue
        if lab.get("l1") in DROP_L1:
            continue
        fp = f"{CROPS2}/{lab['crop_file']}"
        if not os.path.exists(fp):
            continue
        paths.append(fp); brands.append(canon(b, cmap)); l1s.append(lab.get("l1", "飲料"))
    return paths, brands, l1s


def encode(model, proc, paths):
    out = []
    with torch.no_grad():
        for s in range(0, len(paths), 64):
            imgs = [Image.open(p).convert("RGB") for p in paths[s:s + 64]]
            inp = proc(images=imgs, return_tensors="pt").to(DEVICE)
            f = model.get_image_features(**inp)
            out.append((f / f.norm(dim=-1, keepdim=True)).cpu().numpy())
    return np.concatenate(out, 0).astype(np.float32)


def build_vocab(brands, l1s):
    cnt = Counter(brands)
    l1_votes = defaultdict(Counter)
    for b, l1 in zip(brands, l1s):
        l1_votes[b][l1] += 1
    vocab, brand_l1 = [], {}
    for b, n in cnt.items():
        if b in EXCLUDE or b in UNK:
            continue
        majority_l1 = l1_votes[b].most_common(1)[0][0]
        floor = DRINK_MIN if majority_l1 == "飲料" else MIN_TOTAL
        if n >= floor:
            vocab.append(b); brand_l1[b] = majority_l1
    return sorted(vocab), brand_l1


def main():
    cmap = load_canon()
    paths, brands, l1s = [], [], []
    sources = [
        load_folder(EC, ["飲料", "泡麵", "零食"], cmap),
        load_folder(WEB, ["飲料", "泡麵", "零食"], cmap),
        load_folder(FRESH, ["鮮食"], cmap),
        load_folder(MANUAL, ["飲料", "泡麵", "零食", "鮮食"], cmap),
        load_folder(DS2_BOXED, ["飲料", "泡麵", "零食", "鮮食"], cmap),
        load_crops1(cmap),
        load_crops2(cmap),
    ]
    for p, b, l in sources:
        paths += p; brands += b; l1s += l
    print(f"total samples: {len(paths)} | unique raw brands: {len(set(brands))}")

    vocab, brand_l1 = build_vocab(brands, l1s)
    print(f"L2 vocab: {len(vocab)} 品牌 (by L1: {dict(Counter(brand_l1.values()))})")
    b2i = {b: i for i, b in enumerate(vocab)}
    keep = [i for i, b in enumerate(brands) if b in b2i]
    Xpaths = [paths[i] for i in keep]
    y = np.array([b2i[brands[i]] for i in keep])

    model = ChineseCLIPModel.from_pretrained(MODEL_ID).to(DEVICE).eval()
    proc = ChineseCLIPProcessor.from_pretrained(MODEL_ID)
    X = encode(model, proc, Xpaths)
    print(f"training on {len(X)} samples")

    Xt = torch.tensor(X, device=DEVICE); yt = torch.tensor(y, device=DEVICE)
    clf = nn.Linear(X.shape[1], len(vocab)).to(DEVICE)
    opt = torch.optim.AdamW(clf.parameters(), lr=1e-2, weight_decay=1e-3)
    lossf = nn.CrossEntropyLoss()
    for _ in range(500):
        opt.zero_grad(); lossf(clf(Xt), yt).backward(); opt.step()

    if os.path.exists(OUT) and not os.path.exists(BACKUP):
        os.rename(OUT, BACKUP); print(f"backed up old probe -> {BACKUP}")
    W = clf.weight.detach().cpu().numpy(); bsv = clf.bias.detach().cpu().numpy()
    brands_l1 = np.array([brand_l1[b] for b in vocab])
    np.savez(OUT, W=W, b=bsv, brands=np.array(vocab), brands_l1=brands_l1)
    print(f"saved probe -> {OUT}  (W {W.shape})")
    print(f"飲料 in vocab: {sorted(b for b in vocab if brand_l1[b] == '飲料')}")
    print(f"鮮食 in vocab: {sorted(b for b in vocab if brand_l1[b] == '鮮食')}")


if __name__ == "__main__":
    main()
