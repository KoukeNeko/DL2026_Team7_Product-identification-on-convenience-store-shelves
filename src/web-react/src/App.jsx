import { useEffect, useState } from 'react'
import Shelf3D from './Shelf3D.jsx'
import ImagePanel from './ImagePanel.jsx'
import { STATUS_BANDS } from './lib.js'

const CATS = [
  ['飲料', 'Drink', '飲料'], ['泡麵', 'Noodle', '泡麵'], ['零食', 'Snack', '零食'],
  ['鮮食', 'Fresh', '新鮮'], ['unknown', 'Other', '其他'],
]

function CountUp({ to }) {
  const [v, setV] = useState(0)
  useEffect(() => {
    if (typeof to !== 'number') return
    let cur = 0
    const step = Math.max(1, Math.ceil(to / 28))
    const t = setInterval(() => {
      cur = Math.min(to, cur + step)
      setV(cur)
      if (cur >= to) clearInterval(t)
    }, 22)
    return () => clearInterval(t)
  }, [to])
  return <>{typeof to === 'number' ? v : '—'}</>
}

function Kpi({ icon, bg, color, lab, children }) {
  return (
    <div className="kpi">
      <div className="ic" style={{ background: bg, color }}>{icon}</div>
      <div><div className="lab">{lab}</div><div className="num">{children}</div></div>
    </div>
  )
}

function Legend({ data }) {
  return (
    <div id="legend">
      <div className="leg">
        <h4>品類分類</h4>
        {CATS.map(([k, en, zh]) => {
          const c = data.colors[k] || [138, 147, 160]   // same palette as image boxes + 3D
          return (
            <div className="row" key={k}>
              <span className="badge-cat" style={{ background: `rgb(${c[0]},${c[1]},${c[2]})` }}>{en}</span>{zh}
              <span className="n">{data.kpi.l1[k] || 0}</span>
            </div>
          )
        })}
      </div>
      <div className="leg">
        <h4>狀態圖例</h4>
        {STATUS_BANDS.map(([lab, c], i) => (
          <div className="row" key={lab}>
            <span className="dot" style={{ background: c }} />{i === 2 ? '⚠ ' : ''}{lab}
          </div>
        ))}
      </div>
    </div>
  )
}

export default function App() {
  const [data, setData] = useState(null)
  const [imgUrl, setImgUrl] = useState(null)
  const [fileName, setFileName] = useState('')
  const [procTime, setProcTime] = useState('')
  const [processing, setProcessing] = useState(false)
  const [selected, setSelected] = useState(null)
  const [autoRotate, setAutoRotate] = useState(false)
  const [resetSignal, setResetSignal] = useState(0)
  const [shared, setShared] = useState(false)
  const [hover, setHover] = useState(false)

  // stage: landing(no image) -> scanning(centered photo + FX) -> scene(photo docks left, 3D fills bg)
  const stage = !imgUrl ? 'landing' : (data && !processing) ? 'scene' : 'scanning'

  async function handleFile(file) {
    if (!file) return
    setFileName(file.name)
    setImgUrl(URL.createObjectURL(file))
    setData(null)
    setSelected(null)
    setProcessing(true)
    try {
      const fd = new FormData()
      fd.append('file', file)
      const r = await fetch('/recognize', { method: 'POST', body: fd })
      const d = await r.json()
      await new Promise((res) => setTimeout(res, 500))
      setData(d)
      setProcTime(new Date().toLocaleString('zh-TW', { hour12: false }))
    } catch (e) {
      console.error(e)
    } finally {
      setProcessing(false)
    }
  }

  function pick(e) {
    const f = e.target.files?.[0] || e.dataTransfer?.files?.[0]
    if (f) handleFile(f)
  }
  function reset() {
    setImgUrl(null); setData(null); setSelected(null); setFileName(''); setProcTime('')
  }
  function share() {
    navigator.clipboard?.writeText(location.href)
    setShared(true)
    setTimeout(() => setShared(false), 1500)
  }

  const low = data ? data.items.filter((it) => it.l2 !== '?' && it.conf < 50).length : '—'
  const unk = data ? data.items.filter((it) => it.l2 === '?').length : '—'
  const catCount = data ? Object.values(data.kpi.l1).filter((v) => v > 0).length : '—'

  return (
    <div id="app" className={`st-${stage}`}>
      {/* fullscreen 3D scene as the page background once recognition is done */}
      <div id="scene3d" aria-hidden={stage !== 'scene'}>
        {data && (
          <Shelf3D data={data} autoRotate={autoRotate} resetSignal={resetSignal} selected={selected} onSelect={setSelected} />
        )}
      </div>

      <header id="top">
        <div className="brand">🛰 貨架辨識 — 即時 <span className="d">3D</span> Demo</div>
        <div className="pipe">YOLO·SKU-110K • Chinese-CLIP • 結構解析 → 3D Planogram</div>
        <div className="sp" />
        {stage === 'scene' && (
          <>
            <button className="tbtn" onClick={() => setResetSignal((s) => s + 1)}>⬚ 等軸測</button>
            <button className={'tbtn' + (autoRotate ? ' on' : '')} onClick={() => setAutoRotate((a) => !a)}>↻ 旋轉</button>
          </>
        )}
        <button className="tbtn" onClick={share}>{shared ? '✓ 已複製' : '⇪ 分享'}</button>
        {imgUrl && <button className="tbtn" onClick={reset}>↺ 換一張</button>}
      </header>

      {/* landing hero: just the upload UI */}
      {stage === 'landing' && (
        <div id="hero">
          <div
            id="drop"
            className={hover ? 'hover' : ''}
            onClick={() => document.getElementById('fileIn').click()}
            onDragOver={(e) => { e.preventDefault(); setHover(true) }}
            onDragLeave={() => setHover(false)}
            onDrop={(e) => { e.preventDefault(); setHover(false); pick(e) }}
          >
            <div className="ic">📷</div>
            <h2>拖曳或點擊上傳貨架照片</h2>
            <p>上傳後即時掃描，完成後整個畫面將重建為 3D 貨架場景</p>
            <div className="hint">JPG / PNG · 單張貨架正面照效果最佳</div>
          </div>
        </div>
      )}

      {/* the photo: centered while scanning, floats to the left when the scene appears */}
      {imgUrl && (
        <div id="photowrap" className={stage === 'scene' ? 'docked' : ''}>
          <ImagePanel imgUrl={imgUrl} processing={processing} data={data} fileName={fileName} procTime={procTime} />
        </div>
      )}

      {/* overlays that belong to the 3D scene */}
      {stage === 'scene' && <Legend data={data} />}
      {stage === 'scene' && (
        <div className="kpis">
          <Kpi icon="▦" bg="#11304a" color="#4f8cff" lab="總儲位"><CountUp to={data ? data.kpi.facings : '—'} /></Kpi>
          <Kpi icon="✓" bg="#0e2a1e" color="#46c97e" lab="辨識成功">
            <CountUp to={data ? data.kpi.identified : '—'} />{data && <small> ({data.kpi.id_pct}%)</small>}
          </Kpi>
          <Kpi icon="◎" bg="#33260e" color="#f59e3b" lab="低信心度"><CountUp to={low} /></Kpi>
          <Kpi icon="⚑" bg="#33141d" color="#ec5e9c" lab="未辨識"><CountUp to={unk} /></Kpi>
          <Kpi icon="⬡" bg="#1a2740" color="#9bbccc" lab="商品類別"><CountUp to={catCount} /></Kpi>
        </div>
      )}

      <input id="fileIn" type="file" accept="image/*" hidden onChange={pick} />
    </div>
  )
}
