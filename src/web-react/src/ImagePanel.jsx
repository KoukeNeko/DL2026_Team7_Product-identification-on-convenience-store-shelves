import { useEffect, useRef } from 'react'

function drawFrost(ctx, W, H, t) {
  ctx.clearRect(0, 0, W, H)
  ctx.fillStyle = 'rgba(196,218,236,.10)'
  ctx.fillRect(0, 0, W, H)
  for (let k = 0; k < 3; k++) {
    const a = t * 0.6 + k * 2.1
    const bx = (0.5 + 0.42 * Math.cos(a)) * W
    const by = (0.5 + 0.42 * Math.sin(a * 1.3)) * H
    const g = ctx.createRadialGradient(bx, by, 0, bx, by, Math.max(W, H) * 0.5)
    g.addColorStop(0, k === 1 ? 'rgba(255,150,210,.14)' : 'rgba(120,210,255,.14)')
    g.addColorStop(1, 'rgba(0,0,0,0)')
    ctx.fillStyle = g
    ctx.fillRect(0, 0, W, H)
  }
  const p = ((t * 0.4) % 1.5) - 0.25
  const cx = p * W
  const sg = ctx.createLinearGradient(cx - W * 0.25, 0, cx + W * 0.45, H)
  sg.addColorStop(0, 'rgba(190,240,255,0)')
  sg.addColorStop(0.5, 'rgba(205,245,255,.22)')
  sg.addColorStop(1, 'rgba(190,240,255,0)')
  ctx.fillStyle = sg
  ctx.fillRect(0, 0, W, H)
  ctx.globalAlpha = 0.045
  for (let i = 0; i < 160; i++) {
    ctx.fillStyle = Math.random() > 0.5 ? '#d4f1ff' : '#9fe'
    ctx.fillRect(Math.random() * W, Math.random() * H, 2, 2)
  }
  ctx.globalAlpha = 1
}

function tracePolygon(ctx, polygon, sx, sy) {
  ctx.beginPath()
  polygon.forEach(([x, y], i) => (i ? ctx.lineTo(x * sx, y * sy) : ctx.moveTo(x * sx, y * sy)))
  ctx.closePath()
}

// v2 YOLOv8-seg structural overlay: door MASKS (amber) + shelf mask BANDS (cyan),
// drawn behind the product boxes. The band is the mask polygon itself (not a fitted
// line), so it stays aligned to the photo even when the mask is rough on glass.
function drawStructure(ctx, W, H, data, alpha) {
  const sx = W / data.w, sy = H / data.h
  ;(data.seg_doors || []).forEach((poly) => {
    if (!poly || poly.length < 3) return
    tracePolygon(ctx, poly, sx, sy)
    ctx.fillStyle = `rgba(245,158,59,${0.18 * alpha})`
    ctx.fill()
    ctx.strokeStyle = `rgba(245,158,59,${0.85 * alpha})`
    ctx.lineWidth = Math.max(1.5, W / 300)
    ctx.stroke()
  })
  ;(data.seg_shelf || []).forEach((poly) => {
    if (!poly || poly.length < 3) return
    tracePolygon(ctx, poly, sx, sy)
    ctx.fillStyle = `rgba(61,214,232,${0.34 * alpha})`
    ctx.fill()
    ctx.strokeStyle = `rgba(61,214,232,${0.9 * alpha})`
    ctx.lineWidth = Math.max(1, W / 460)
    ctx.stroke()
  })
}

function drawBoxes(ctx, W, H, data, t) {
  ctx.clearRect(0, 0, W, H)
  const sx = W / data.w, sy = H / data.h
  drawStructure(ctx, W, H, data, Math.min(1, t * 1.6))   // structure fades in behind the boxes
  data.items.forEach((it, i) => {
    const fr = Math.min(1, Math.max(0, t * data.items.length - i))
    if (fr <= 0) return
    const col = data.colors[it.l1] || [120, 160, 200]
    const x = it.x1 * sx, y = it.y1 * sy, w = (it.x2 - it.x1) * sx, h = (it.y2 - it.y1) * sy
    ctx.strokeStyle = `rgba(${col[0]},${col[1]},${col[2]},${0.95 * fr})`
    ctx.lineWidth = 2
    ctx.strokeRect(x + (w * (1 - fr)) / 2, y + (h * (1 - fr)) / 2, w * fr, h * fr)
    if (it.l2 !== '?' && fr > 0.8) {
      ctx.fillStyle = `rgba(${col[0]},${col[1]},${col[2]},.92)`
      ctx.font = "10px 'Noto Sans TC',sans-serif"
      const tw = ctx.measureText(it.l2).width + 6
      ctx.fillRect(x, y - 13, tw, 12)
      ctx.fillStyle = '#05121a'
      ctx.fillText(it.l2, x + 3, y - 3.5)
    }
  })
}

export default function ImagePanel({ imgUrl, processing, data, fileName, procTime }) {
  const imgRef = useRef(), fxRef = useRef(), ovRef = useRef()

  function sizeCanvases() {
    const img = imgRef.current
    if (!img) return
    ;[fxRef.current, ovRef.current].forEach((c) => {
      if (!c) return
      c.width = img.clientWidth
      c.height = img.clientHeight
    })
  }

  // frosted scan FX while processing
  useEffect(() => {
    if (!processing || !fxRef.current) return
    sizeCanvases()
    let raf, t = 0, running = true
    const ctx = fxRef.current.getContext('2d')
    const loop = () => {
      if (!running || !fxRef.current) return
      t += 0.016
      drawFrost(ctx, fxRef.current.width, fxRef.current.height, t)
      raf = requestAnimationFrame(loop)
    }
    loop()
    return () => { running = false; cancelAnimationFrame(raf); const c = fxRef.current; if (c) c.getContext('2d').clearRect(0, 0, c.width, c.height) }
  }, [processing])

  // animated box reveal once results arrive
  useEffect(() => {
    if (processing || !data || !ovRef.current) return
    sizeCanvases()
    let raf, t = 0, running = true
    const ctx = ovRef.current.getContext('2d')
    const loop = () => {
      if (!running || !ovRef.current) return
      t = Math.min(1, t + 0.02)
      drawBoxes(ctx, ovRef.current.width, ovRef.current.height, data, t)
      if (t < 1) raf = requestAnimationFrame(loop)
    }
    loop()
    return () => { running = false; cancelAnimationFrame(raf) }
  }, [data, processing])

  // keep the box overlay aligned when the image is resized (layout/window changes)
  useEffect(() => {
    if (!imgRef.current) return
    const ro = new ResizeObserver(() => {
      sizeCanvases()
      if (data && !processing && ovRef.current) {
        drawBoxes(ovRef.current.getContext('2d'), ovRef.current.width, ovRef.current.height, data, 1)
      }
    })
    ro.observe(imgRef.current)
    return () => ro.disconnect()
  }, [data, processing, imgUrl])

  const filter = processing ? 'blur(18px) saturate(.6) brightness(.92)' : 'none'

  return (
    <div className="card" id="imgcard">
      <div className="hd">
        <h3>偵測影像</h3>
        {data && !processing && <span className="badge">✓ 處理完成</span>}
      </div>
      <div id="imgwrap">
        <div className="stage">
          <img id="photo" ref={imgRef} src={imgUrl} onLoad={sizeCanvases} style={{ filter, transition: processing ? 'filter .3s' : 'filter .95s cubic-bezier(.2,.7,.2,1)' }} alt="shelf" />
          <canvas className="fxcanvas" ref={fxRef} />
          <canvas className="ovcanvas" ref={ovRef} />
          {processing && <div id="status">ANALYZING…</div>}
        </div>
      </div>
      <div className="meta">
        <div><div className="k">影像來源</div><div className="v">{fileName || '—'}</div></div>
        <div><div className="k">處理時間</div><div className="v">{procTime || '—'}</div></div>
        <div><div className="k">偵測結果</div><div className="v">{data ? <>{data.kpi.facings} 個儲位 · <span className="g">{data.kpi.id_pct}% 信心度</span></> : '—'}</div></div>
      </div>
    </div>
  )
}
