"""L2 brand classification — does training help, and does adding real crops help?

Features: frozen Chinese-CLIP. Classifier: linear softmax probe.
Train sources vary; ALL configs are evaluated on the SAME held-out REAL shelf crops
(貨架擷取, VLM-brand-labelled) so the comparison is clean and reflects deployment.

Configs (eval = real-test brand top-1):
  A  kNN over clean refs            (current zero-shot baseline)
  B  probe trained on clean refs    (train on product images only)
  C  probe trained on clean + real-train  (product images + real crops)
  D  probe trained on real-train only     (real crops only)
"""
import glob
import json
import os
import sys
from collections import Counter

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from transformers import ChineseCLIPModel, ChineseCLIPProcessor

HOME = os.environ.get("DATA_DIR", "data")
EC = f"{HOME}/data_clean/ref_ecommerce"          # clean product images (by brand)
CROPS = f"{HOME}/cropset/crops"                  # real shelf crops
CROP_LABELS = f"{HOME}/cropset/labels.json"      # corrected brand per crop
MODEL_ID = "OFA-Sys/chinese-clip-vit-base-patch16"
IMG_EXT = (".jpg", ".jpeg", ".png", ".webp")
UNK = {"", "?", "？", "unknown", "Unknown", "未辨識"}
MIN_CLEAN, MIN_REAL = 5, 4   # a brand must have enough of both to train+test
DEVICE = "cuda"
SEED = 0

ALIAS = {"doritos": "多力多滋", "doritos 多力多滋": "多力多滋", "cheetos": "奇多",
         "lay's": "樂事", "lays": "樂事", "lay's 樂事": "樂事", "pringles": "品客",
         "monster": "魔爪", "魔爪 monster": "魔爪", "pocky": "百奇",
         "familymart 全家": "全家", "全家 familymart": "全家",
         "familymart collection": "全家", "fami collection": "全家"}


def canon(b: str) -> str:
    return ALIAS.get(b.strip().lower(), b.strip())


_model = ChineseCLIPModel.from_pretrained(MODEL_ID).to(DEVICE).eval()
_proc = ChineseCLIPProcessor.from_pretrained(MODEL_ID)


def encode(paths: list) -> np.ndarray:
    feats = []
    with torch.no_grad():
        for s in range(0, len(paths), 64):
            imgs = [Image.open(p).convert("RGB") for p in paths[s:s + 64]]
            inp = _proc(images=imgs, return_tensors="pt").to(DEVICE)
            f = _model.get_image_features(**inp)
            feats.append((f / f.norm(dim=-1, keepdim=True)).cpu().numpy())
    return np.concatenate(feats, 0).astype(np.float32)


def load_clean() -> tuple:
    paths, brands = [], []
    for cat in ["飲料", "泡麵", "零食"]:
        base = f"{EC}/{cat}"
        if not os.path.isdir(base):
            continue
        for brand in sorted(os.listdir(base)):
            bd = f"{base}/{brand}"
            if not os.path.isdir(bd) or brand.startswith("_"):
                continue
            for p in glob.glob(f"{bd}/*"):
                if p.lower().endswith(IMG_EXT):
                    paths.append(p); brands.append(canon(brand))
    return paths, np.array(brands)


def load_real() -> tuple:
    labels = json.load(open(CROP_LABELS, encoding="utf-8"))
    idmap = {c["id"]: c["file"] for c in json.load(open(f"{HOME}/cropset/crops.json", encoding="utf-8"))}
    paths, brands = [], []
    for lab in labels:
        b = (lab.get("brand") or "").strip()
        if b in UNK or lab["id"] not in idmap:
            continue
        paths.append(f"{CROPS}/{idmap[lab['id']]}"); brands.append(canon(b))
    return paths, np.array(brands)


def knn_predict(gallery_X, gallery_y, test_X, k=5):
    sim = test_X @ gallery_X.T
    return np.array([Counter(gallery_y[np.argsort(-sim[i])[:k]]).most_common(1)[0][0]
                     for i in range(len(test_X))])


def train_probe(X, y_idx, n_classes, epochs=400):
    Xt = torch.tensor(X, device=DEVICE); yt = torch.tensor(y_idx, device=DEVICE)
    clf = nn.Linear(X.shape[1], n_classes).to(DEVICE)
    opt = torch.optim.AdamW(clf.parameters(), lr=1e-2, weight_decay=1e-3)
    lossf = nn.CrossEntropyLoss()
    for _ in range(epochs):
        opt.zero_grad(); loss = lossf(clf(Xt), yt); loss.backward(); opt.step()
    return clf


def probe_predict(clf, test_X, idx2brand):
    with torch.no_grad():
        pred = clf(torch.tensor(test_X, device=DEVICE)).argmax(1).cpu().numpy()
    return np.array([idx2brand[i] for i in pred])


def main() -> None:
    rng = np.random.RandomState(SEED)
    clean_paths, clean_brands = load_clean()
    real_paths, real_brands = load_real()

    clean_cnt, real_cnt = Counter(clean_brands), Counter(real_brands)
    brands = sorted(b for b in (set(clean_cnt) & set(real_cnt))
                    if clean_cnt[b] >= MIN_CLEAN and real_cnt[b] >= MIN_REAL)
    print(f"clean brands {len(clean_cnt)} | real brands {len(real_cnt)} | testable (≥{MIN_CLEAN}clean,≥{MIN_REAL}real): {len(brands)}")
    print(f"brands: {brands}")
    if len(brands) < 4:
        print("交集太小，實驗意義有限");
    brand2idx = {b: i for i, b in enumerate(brands)}
    idx2brand = {i: b for b, i in brand2idx.items()}

    # filter + encode
    cmask = np.isin(clean_brands, brands)
    Xc = encode([p for p, m in zip(clean_paths, cmask) if m]); yc = clean_brands[cmask]
    rmask = np.isin(real_brands, brands)
    Xr = encode([p for p, m in zip(real_paths, rmask) if m]); yr = real_brands[rmask]

    # split real per brand 50/50
    train_i, test_i = [], []
    for b in brands:
        idx = np.where(yr == b)[0]; rng.shuffle(idx)
        cut = max(1, len(idx) // 2)
        train_i += list(idx[:cut]); test_i += list(idx[cut:])
    train_i, test_i = np.array(train_i), np.array(test_i)
    yr_test = yr[test_i]
    print(f"clean {len(Xc)} | real {len(Xr)} (train {len(train_i)} / test {len(test_i)})\n")

    def acc(pred): return (pred == yr_test).mean() * 100

    results = {}
    results["A kNN(clean)"] = acc(knn_predict(Xc, yc, Xr[test_i]))
    yc_idx = np.array([brand2idx[b] for b in yc])
    results["B probe(clean)"] = acc(probe_predict(train_probe(Xc, yc_idx, len(brands)), Xr[test_i], idx2brand))
    Xcr = np.concatenate([Xc, Xr[train_i]]); ycr = np.concatenate([yc_idx, [brand2idx[b] for b in yr[train_i]]])
    results["C probe(clean+real)"] = acc(probe_predict(train_probe(Xcr, ycr, len(brands)), Xr[test_i], idx2brand))
    yrt_idx = np.array([brand2idx[b] for b in yr[train_i]])
    results["D probe(real only)"] = acc(probe_predict(train_probe(Xr[train_i], yrt_idx, len(brands)), Xr[test_i], idx2brand))

    print("=== L2 品牌 top-1 (held-out 真實 crop) ===")
    for k, v in results.items():
        print(f"  {k:<22} {v:.1f}%")


if __name__ == "__main__":
    main()
