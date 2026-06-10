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
| 偵測 backbone 權重 | 第三方 YOLOv8l-SKU110k | 公開 | (來源連結待補) |
| 品牌特徵 | Chinese-CLIP | 公開 | https://github.com/OFA-Sys/Chinese-CLIP |
| 品牌庫參考圖 | 自公開電商/官方型錄蒐集 | **不公開(版權)** | 取得方式:聯絡 @KoukeNeko / 團隊雲端 (連結待補) |
| 貨架實拍 | 組員自行拍攝 | **不公開** | 聯絡 @KoukeNeko / 團隊雲端 (連結待補) |

## 五、模型 (Models)
權重檔較大,存放於雲端(永久連結):
- 偵測 YOLOv8l(SKU-110K 微調):(雲端連結待補)
- L2 品牌探針 `l2_probe.npz`:(雲端連結待補)
- 結構 YOLOv8-seg `shelf_struct_v2.pt`:(雲端連結待補)
- Chinese-CLIP:由 HuggingFace 自動下載(`OFA-Sys/chinese-clip-vit-base-patch16`)

## 六、使用方法 (Usage)
```bash
# 1. 環境
pip install -r requirements.txt
# 2. 放置模型權重到 models/(連結見上節)
# 3. 啟動後端
cd src/web && uvicorn server:app --host 0.0.0.0 --port 8000
# 4. 前端
cd src/web-react && npm install && npm run build   # 產物在 dist/,由後端 static 提供
# 5. 瀏覽器開 http://localhost:8000 上傳貨架照片
```
> (可選) 設定環境變數 `GEMINI_KEY` 可啟用低信心排面的「受限候選重排」(僅從 CLIP 探針 top-5 內挑選,不可自由生成);未設定則自動略過,系統以純 DL 流程(YOLO + Chinese-CLIP + seg)運作。
> 權重與資料路徑可用環境變數 `MODELS_DIR`、`DATA_DIR` 覆寫(預設讀 `src/models/`)。

## 七、結果 (Results)
- 商品偵測:held-out box **mAP@50 ≈ 0.91**
- 品類 L1(乾淨參考):**≈ 94%**
- 貨架結構(列基準線):box mAP@50 ≈ 0.80(v2)

## 八、授權與聲明 (License & Notes)
- 程式碼:MIT(見 `LICENSE`)。
- 資料:**不適用本授權**,版權屬原始來源 / 拍攝者,僅供學術研究。
- AI 使用揭露:_(如課程要求,請於此據實補充)_

## 九、團隊 (Team)
| 姓名 | 學號 | GitHub |
|---|---|---|
| 陳德生 | (待補) | [@KoukeNeko](https://github.com/KoukeNeko) |
| 鄭雅瀞 | (待補) | [@jane400715](https://github.com/jane400715) |
| 蕭政雲 | (待補) | [@FNKnohe](https://github.com/FNKnohe) |
