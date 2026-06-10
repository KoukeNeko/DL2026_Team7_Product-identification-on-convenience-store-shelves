"""Fine-tune (domain-adapt) YOLOv8l SKU-110K on the human-corrected shelf boxes.

Small-data transfer learning: freeze the backbone, train neck+head on 48 shelves,
hold out 12 for a preliminary honest before/after. Final held-out test = user's
future photos. Saves the fine-tuned model + training curves.
"""
import json
import os
import random

from ultralytics import YOLO

HOME = os.environ.get("DATA_DIR", "data")
VALSET = f"{HOME}/verify/valset"          # images/ + labels/ (60 shelves, 3167 boxes)
BASE_WEIGHTS = f"{HOME}/models/yolov8l_sku110k.pt"
RUN_DIR = f"{HOME}/runs_ft"
IMGSZ, EPOCHS, BATCH, FREEZE, PATIENCE = 1280, 60, 4, 10, 15
SEED = 42


def stratified_split() -> tuple:
    """80/20 split, stratified by source prefix so val spans all store types."""
    images = sorted(os.listdir(f"{VALSET}/images"))

    def bucket(name: str) -> str:
        if name.startswith("7eleven") or name.startswith("7-ELEVEN"):
            return "711"
        if name.startswith("全家"):
            return "familymart"
        if name.startswith("IMG_75"):
            return "img75"
        if name.startswith("IMG_77") or name.startswith("IMG_78"):
            return "img77"
        return "other"

    groups = {}
    for img in images:
        groups.setdefault(bucket(img), []).append(img)

    rng = random.Random(SEED)
    train, val = [], []
    for _, items in sorted(groups.items()):
        rng.shuffle(items)
        n_val = max(1, round(len(items) * 0.2))
        val.extend(items[:n_val])
        train.extend(items[n_val:])
    return sorted(train), sorted(val)


def write_dataset(train: list, val: list) -> str:
    with open(f"{VALSET}/train.txt", "w") as fh:
        fh.write("\n".join(f"{VALSET}/images/{n}" for n in train) + "\n")
    with open(f"{VALSET}/val.txt", "w") as fh:
        fh.write("\n".join(f"{VALSET}/images/{n}" for n in val) + "\n")
    yaml_path = f"{VALSET}/data_ft.yaml"
    with open(yaml_path, "w") as fh:
        fh.write(f"path: {VALSET}\ntrain: train.txt\nval: val.txt\nnames:\n  0: object\n")
    return yaml_path


def eval_on_val(weights: str, yaml_path: str, tag: str) -> dict:
    box = YOLO(weights).val(
        data=yaml_path, imgsz=IMGSZ, iou=0.7, max_det=1000, conf=0.001,
        split="val", plots=False, verbose=False, name=f"eval_{tag}", project=RUN_DIR, exist_ok=True,
    ).box
    return {"mAP50": round(box.map50 * 100, 1), "mAP50_95": round(box.map * 100, 1),
            "P": round(box.mp * 100, 1), "R": round(box.mr * 100, 1)}


def main() -> None:
    train, val = stratified_split()
    yaml_path = write_dataset(train, val)
    print(f"split: train {len(train)} / val {len(val)}")
    print(f"val images: {val}")

    before = eval_on_val(BASE_WEIGHTS, yaml_path, "zeroshot")
    print(f"[zero-shot] {before}")

    model = YOLO(BASE_WEIGHTS)
    model.train(
        data=yaml_path, imgsz=IMGSZ, epochs=EPOCHS, batch=BATCH, freeze=FREEZE,
        patience=PATIENCE, lr0=0.002, cos_lr=True, optimizer="AdamW",
        degrees=0.0, flipud=0.0, fliplr=0.5, mosaic=1.0, close_mosaic=10,
        seed=SEED, project=RUN_DIR, name="finetune", exist_ok=True, verbose=True, device=0,
    )
    best = f"{RUN_DIR}/finetune/weights/best.pt"
    after = eval_on_val(best, yaml_path, "finetuned")
    print(f"[fine-tuned] {after}")

    summary = {"split": {"train": len(train), "val": len(val)}, "val_images": val,
               "zero_shot": before, "fine_tuned": after, "best_weights": best,
               "curves": f"{RUN_DIR}/finetune/results.png"}
    json.dump(summary, open(f"{RUN_DIR}/ft_summary.json", "w"), ensure_ascii=False, indent=2)
    print("\n=== BEFORE vs AFTER (held-out 12) ===")
    print(f"  mAP@50    : {before['mAP50']}%  ->  {after['mAP50']}%")
    print(f"  mAP@50-95 : {before['mAP50_95']}%  ->  {after['mAP50_95']}%")
    print(f"  P / R     : {before['P']}/{before['R']}  ->  {after['P']}/{after['R']}")
    print(f"summary -> {RUN_DIR}/ft_summary.json")


if __name__ == "__main__":
    main()
