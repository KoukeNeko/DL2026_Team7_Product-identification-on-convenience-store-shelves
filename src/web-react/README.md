# 貨架辨識 3D Demo — React 版

Vite + React 18 + react-three-fiber + drei。後端用 FastAPI `/recognize`(web/server.py)。

## 建置
    pnpm install
    pnpm build          # -> dist/(上傳到遠端 web/static/ 給 FastAPI 服務)
    pnpm dev            # 本機開發(proxy /recognize 到 127.0.0.1:8000)

## 結構
- src/App.jsx        版面 + 狀態 + 上傳 fetch + KPI/Legend/CountUp
- src/Shelf3D.jsx    react-three-fiber 3D 貨架(真實材質+陰影+前後排深度+點選)
- src/ImagePanel.jsx 上傳 + 磨砂掃描特效 + 偵測框疊圖
- src/InfoCard.jsx   選取資訊卡(品牌/信心度/前後排/類別)
- src/lib.js         顏色/紋理/版面計算(含 back 旗標前後排放置)

## 重點
- 前後排深度:用後端 `back` 旗標把後排商品放到前排「後面」(computeLayout)
- 4 層修正:後端 merge_back_rows
