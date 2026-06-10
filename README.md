# 便利商店貨架商品辨識與 Planogram 合規分析
Convenience-Store Shelf Product Recognition & Planogram Compliance

> 深度學習概論 期末專題 — 第 7 組 `DL2026_Team7_Product-identification-on-convenience-store-shelves`

## 一、創作目標 (Goals)
針對台灣便利商店的密集貨架,建立一條端到端流程:
**偵測每個商品排面 → 辨識品類(L1)與品牌(L2)→ 解析貨架結構(層列 / 門)→ 重建為 3D Planogram 並計算合規指標**(排面數、辨識率、各品類計數)。對標零售貨架理解的四階段流程(偵測 → 辨識 → 結構 → planogram)。

## 二、系統架構 (Pipeline)
1. **商品偵測**:YOLOv8l(以 SKU-110K 微調)偵測密集排面;後處理去除巢狀重複框、價標假陽性、瓶身過度切割。
2. **品類 / 品牌辨識**:Chinese-CLIP 影像特徵 →
   - L1 四類(飲料 / 泡麵 / 零食 / 鮮食):類別中心最近鄰。
   - L2 品牌(240 類):線性 softmax 探針;低信心時以「受限候選重排」補強(僅能從探針 top-5 內挑,不可自由生成)。
3. **結構解析**:自訓 YOLOv8-seg 偵測「列基準線 + 門」,依商品底邊把每個排面放到對應的層 / 欄。
4. **視覺化**:FastAPI 後端 + React / Three.js 前端,上傳照片即時重建 3D 貨架與 KPI。

## 三、檔案說明 (Repository Structure)
```
src/
  server.py                 # FastAPI 後端入口 (上傳 → 推論 → JSON)
  detection_postprocess.py  # YOLO 偵測 + 後處理 (去重 / 價標過濾 / 瓶身合併)
  pricetag_filter.py        #   價標假陽性過濾
  e2e_recognition.py        # Chinese-CLIP → L1 類別中心 + L2 品牌探針
  vlm_rerank.py             # 低信心時的受限候選重排 (可選元件)
  virtual_shelf.py          # 模型載入 / 結構工具
  rowlib.py                 # 列分群 (sequential RANSAC)
  struct_seg.py             # YOLOv8-seg 列基準線 + 門 偵測
  shelf_assign.py           # 依結構把商品放到層 / 欄
  web-react/                # 前端 (Vite + React + Three.js)
  train/                    # 訓練腳本 (偵測微調 + L2 品牌探針)
  models/                   # 權重存放處 (大檔放雲端,見 models/README.md)
README.md   LICENSE   requirements.txt   .gitignore
```
> 程式以相對路徑 / 環境變數 `MODELS_DIR`、`DATA_DIR` 讀取權重與資料;**資料與大型權重不放在 repo**(見下兩節)。

## 四、資料集 (Datasets)
> ⚠️ **本 repo 不包含任何商品 / 貨架影像**:參考圖多為公開網路與官方型錄蒐集,具版權、不隨 repo 散布;貨架實拍含店家環境亦不公開。以下為來源與取得方式。

| 用途 | 資料 | 是否公開 | 連結 / 取得方式 |
|---|---|---|---|
| 偵測微調 | SKU-110K | 公開 | https://github.com/eg4000/SKU110K_CVPR19 |
| 偵測 backbone 權重 | 第三方 YOLOv8l-SKU110k（`sbfisher/yolov8l-sku110k` → `best_imgsz640.pt`） | 公開 | [HuggingFace](https://huggingface.co/sbfisher/yolov8l-sku110k) |
| 品牌特徵 | Chinese-CLIP | 公開 | https://github.com/OFA-Sys/Chinese-CLIP |
| 品牌庫參考圖 | 自公開電商/官方型錄蒐集 | **不公開(版權)** | 取得方式:聯絡 @KoukeNeko / 團隊雲端 (連結待補) |
| 貨架實拍 | 組員自行拍攝 | **不公開** | 聯絡 @KoukeNeko / 團隊雲端 (連結待補) |

## 五、模型 (Models)
權重檔較大,存放於雲端(永久連結):
- 偵測 YOLOv8l(SKU-110K 微調):[Google Drive](https://drive.google.com/file/d/1aoUbYst-gHT2T33Vy5U6IjMJA1SsWXWQ/view?usp=sharing)
- L2 品牌探針 `l2_probe.npz`:[Google Drive](https://drive.google.com/file/d/1pQdm6H8Fba5mmUOjw4VRA5F2QBW82WXN/view?usp=sharing)
- L1 類別中心 `gallery_cnclip_clean.npz`:[Google Drive](https://drive.google.com/file/d/1tyDe8Hx1EGaxybNZFQBLvNh_rxyUqUSb/view?usp=sharing)
- 結構 YOLOv8-seg `shelf_struct_v2.pt`:[Google Drive](https://drive.google.com/file/d/1tVT0f231DWQe1De-tghhb8nX_hcbf5G_/view?usp=sharing)
- Chinese-CLIP:由 HuggingFace 自動下載(`OFA-Sys/chinese-clip-vit-base-patch16`)

## 六、使用方法 (Usage)
```bash
# 1. 安裝相依套件
pip install -r requirements.txt
# 2. 放置模型權重到 src/models/(連結見第五節);Chinese-CLIP 會由 HuggingFace 自動下載
# 3. 建置前端 → 產物複製到 src/static/(FastAPI 由此提供頁面)
cd src/web-react && npm install && npm run build && cp -r dist/* ../static/ && cd ..
# 4. 啟動後端(server.py 在 src/,服務 src/static/)
uvicorn server:app --host 0.0.0.0 --port 8000
# 5. 瀏覽器開 http://localhost:8000,上傳貨架照片
```
### 環境變數

| 變數 | 必要性 | 說明 |
|---|---|---|
| `MODELS_DIR` | 選填 | 權重資料夾(預設 `src/models/`) |
| `DATA_DIR` | 選填 | 參考圖 / 資料路徑 |
| `GEMINI_KEY` | **選填** | 啟用 Gemini VLM 輔助(見下) |

**`GEMINI_KEY` —— 可選的 VLM 輔助(預設關閉)**

對 CLIP 探針標為 `unknown` 的排面,讓 VLM 從**探針自己的 top-5 候選**中挑一個(或回報 `none`)。它**不能自由生成品牌**,只能在候選內選,因此能救回「看得出但探針沒把握」的品牌而不會幻覺。模型用 `gemini-2.5-flash-lite`。

- **取得金鑰**:Google AI Studio <https://aistudio.google.com/apikey>(免費額度即可)。
- **啟用**:`export GEMINI_KEY=你的金鑰`,再啟動後端。
- **未設定時自動略過** → 系統以純 DL 流程(YOLO + Chinese-CLIP + seg)運作,結果完全可重現。
- ⚠️ 金鑰**請勿**寫進程式或 commit 進 repo(`.gitignore` 已擋 `.env` 與 `start_web.sh`)。

## 七、結果 (Results)
- 商品偵測:held-out box **mAP@50 ≈ 0.91**
- 品類 L1(乾淨參考):**≈ 94%**
- 貨架結構(列基準線):box mAP@50 ≈ 0.80(v2)

## 八、授權與聲明 (License & Notes)
**程式碼:GNU AGPL-3.0**(見 `LICENSE`)。本系統使用 Ultralytics YOLO(AGPL-3.0),依其 copyleft 規定,整個衍生作品一併以 AGPL-3.0 釋出。

**第三方元件與資料授權:**

| 元件 / 資料 | 授權 | 備註 |
|---|---|---|
| Ultralytics YOLO(偵測 / seg) | AGPL-3.0 | 含本 repo 程式與 `*.pt` 衍生權重 |
| Chinese-CLIP（`OFA-Sys/chinese-clip-vit-base-patch16`） | Apache-2.0 | 由 HuggingFace 自動下載 |
| Noto Sans CJK TC 字型（`NotoSansTC.otf`） | SIL OFL 1.1 | 隨 repo 提供,見 `src/models/OFL.txt` |
| SKU-110K 資料集 | 學術 / 非商業(需向作者申請) | 由其微調的偵測權重僅供研究 |
| 品牌參考圖 / 貨架實拍 | 版權屬原始來源 / 拍攝者 | 不隨 repo 散布,僅供學術研究 |

> **模型權重**(`yolov8l_sku110k_finetuned.pt`、`l2_probe.npz`、`gallery_cnclip_clean.npz`、`shelf_struct_v2.pt`)衍生自上述資料集與有版權的參考圖,**僅供學術研究、不可商用、不宜自由轉散**;雲端連結建議設為授權存取(僅分享給助教 / 老師)。

- AI 使用揭露:_(如課程要求,請於此據實補充)_

## 九、團隊 (Team) — 第 7 組
| 角色 | 姓名 | 學號 | GitHub |
|---|---|---|---|
| 組長 | 陳德生 | 614410091 | [@KoukeNeko](https://github.com/KoukeNeko) |
| 組員 | 蕭政雲 | 613410079 | [@FNKnohe](https://github.com/FNKnohe) |
| 組員 | 鄭雅瀞 | 614410139 | [@jane400715](https://github.com/jane400715) |
