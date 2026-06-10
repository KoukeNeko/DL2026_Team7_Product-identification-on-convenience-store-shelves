# 模型權重 (Model Weights)

權重檔較大,**不放在 repo**。請從雲端下載後放到本資料夾 (`src/models/`),或設定環境變數 `MODELS_DIR` 指向存放位置:

| 檔名 | 用途 | 下載 |
|---|---|---|
| `yolov8l_sku110k_finetuned.pt` | 商品偵測 (SKU-110K 微調) | (雲端連結待補) |
| `l2_probe.npz` | L2 品牌線性探針 | (雲端連結待補) |
| `gallery_cnclip_clean.npz` | L1 類別中心 (Chinese-CLIP 特徵庫) | (雲端連結待補) |
| `shelf_struct_v2.pt` | 貨架結構 YOLOv8-seg (列基準線 + 門) | (雲端連結待補) |
| `NotoSansTC.otf` | 標籤字型 (僅獨立視覺化用) | 任一 Noto Sans TC 字型 |

Chinese-CLIP 主幹由 HuggingFace 自動下載 (`OFA-Sys/chinese-clip-vit-base-patch16`,Apache-2.0)。

> ⚠️ **授權**:`*.pt`(YOLO 系)權重受 Ultralytics **AGPL-3.0** 約束;所有權重均衍生自 SKU-110K(學術/非商業)與有版權的品牌參考圖,**僅供學術研究、不可商用、不宜公開轉散**。雲端連結建議設為授權存取(僅分享給助教/老師)。詳見根目錄 `README.md` §八。
