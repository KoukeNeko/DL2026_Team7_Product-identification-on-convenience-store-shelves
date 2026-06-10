import os
from ultralytics import YOLO
H=os.environ.get("DATA_DIR", "data"); V=f"{H}/verify/valset"
imgs=sorted(os.listdir(f"{V}/images"))
open(f"{V}/all60.txt","w").write("\n".join(f"{V}/images/{n}" for n in imgs)+"\n")
open(f"{V}/data_all.yaml","w").write(f"path: {V}\ntrain: all60.txt\nval: all60.txt\nnames:\n  0: object\n")
print(f"train on ALL {len(imgs)} shelves (deploy model)")
YOLO(f"{H}/models/yolov8l_sku110k.pt").train(
    data=f"{V}/data_all.yaml", imgsz=1280, epochs=60, batch=4, freeze=10, patience=999,
    lr0=0.002, cos_lr=True, optimizer="AdamW", degrees=0.0, flipud=0.0, fliplr=0.5,
    mosaic=1.0, close_mosaic=10, seed=42, project=f"{H}/runs_ft",
    name="finetune_all60", exist_ok=True, verbose=True, device=0)
print("DONE -> runs_ft/finetune_all60/weights/best.pt")
