import * as THREE from 'three'

export const L1_EN = { 飲料: 'Drink', 泡麵: 'Noodle', 零食: 'Snack', 鮮食: 'Fresh', unknown: 'Other' }
export const CONF_BANDS = [
  ['高 (80–100)', '#46c97e'],
  ['中 (50–79)', '#f59e3b'],
  ['低 (<50)', '#7d90a4'],
]
export const STATUS_BANDS = [
  ['已識別', '#4f8cff'],
  ['未識別', '#8a93a0'],
  ['低辨識度', '#f3c34a'],
]

export function confColor(p) {
  return p >= 80 ? '#46c97e' : p >= 50 ? '#f59e3b' : '#7d90a4'
}

// curated colours for well-known brands (their real packaging hue); the rest hash to a
// stable distinct colour so every product on the shelf is visually separable.
const BRAND_COLORS = {
  可樂: [214, 47, 49], 可口可樂: [214, 47, 49], 百事: [40, 90, 170],
  雪碧: [46, 165, 82], 芬達: [247, 148, 29], 沙士: [140, 45, 42], 黑松沙士: [140, 45, 42],
  紅牛: [44, 96, 172], 能量飲料: [60, 110, 200], 魔爪: [86, 196, 92],
  水: [110, 180, 225], 礦泉水: [150, 168, 188], 運動飲料: [124, 192, 74], 寶礦力: [120, 196, 210],
  綠茶: [96, 168, 66], 奶茶: [182, 150, 112], 檸檬紅茶: [214, 170, 58], 麥香: [206, 160, 70],
  柳橙汁: [242, 158, 40], 蘋果汁: [150, 190, 70], 葡萄汁: [140, 90, 170], 蘆薈汁: [120, 195, 120],
  麥仔茶: [150, 110, 70], 冬瓜茶: [120, 85, 60], 茶裏王: [90, 165, 70], 御茶園: [150, 180, 90],
  午後時光: [200, 120, 90], 貝納頌: [120, 80, 60], 立頓: [230, 200, 60], 泰山: [120, 160, 200],
  波蜜: [230, 150, 60], 可爾必思: [223, 228, 235], FIN: [70, 150, 210],
  泡麵: [200, 60, 50], 杯麵: [240, 150, 40],
  洋芋片: [225, 180, 60], 品客: [225, 180, 60], 餅乾: [210, 95, 72], 巧克力: [104, 66, 48],
  口香糖: [88, 190, 200],
}

function hslToRgb(h, s, l) {
  const k = (n) => (n + h * 12) % 12
  const a = s * Math.min(l, 1 - l)
  const f = (n) => l - a * Math.max(-1, Math.min(k(n) - 3, Math.min(9 - k(n), 1)))
  return [Math.round(f(0) * 255), Math.round(f(8) * 255), Math.round(f(4) * 255)]
}

export function productColor(item) {
  if (!item || item.l2 === '?') return { col: [138, 147, 160], known: false }
  if (BRAND_COLORS[item.l2]) return { col: BRAND_COLORS[item.l2], known: true }
  let n = 0
  for (let i = 0; i < item.l2.length; i++) n = (n * 131 + item.l2.charCodeAt(i)) >>> 0
  return { col: hslToRgb((n % 360) / 360, 0.55, 0.56), known: true }
}

// brand label painted onto the product face. The canvas aspect matches the face (h/w) so the
// label is NOT stretched, and the font fills the face width for readability (function-first).
export function productTexture(col, label, hw = 1) {
  const W = 180
  const H = Math.round(W * Math.min(2.6, Math.max(0.6, hw)))
  const c = document.createElement('canvas')
  c.width = W
  c.height = H
  const x = c.getContext('2d')
  x.fillStyle = `rgb(${col[0]},${col[1]},${col[2]})`
  x.fillRect(0, 0, W, H)
  const g = x.createLinearGradient(0, 0, 0, H)
  g.addColorStop(0, 'rgba(255,255,255,.17)')
  g.addColorStop(0.5, 'rgba(255,255,255,0)')
  g.addColorStop(1, 'rgba(0,0,0,.15)')
  x.fillStyle = g
  x.fillRect(0, 0, W, H)
  if (label && label !== '?') {
    const chars = [...label]                          // vertical (直書): one character per line
    const fs = Math.max(12, Math.min(Math.round(W * 0.34), Math.round((H * 0.86) / chars.length)))
    x.font = `700 ${fs}px 'Noto Sans TC',sans-serif`
    x.textAlign = 'center'
    x.textBaseline = 'middle'
    x.shadowColor = 'rgba(0,0,0,.45)'
    x.shadowBlur = 6
    x.shadowOffsetY = 2
    x.fillStyle = '#ffffff'
    const lineHeight = fs * 1.06
    const top = H / 2 - (lineHeight * (chars.length - 1)) / 2
    chars.forEach((ch, i) => x.fillText(ch, W / 2, top + i * lineHeight))
    x.shadowColor = 'transparent'
  }
  const t = new THREE.CanvasTexture(c)
  t.anisotropy = 4
  t.colorSpace = THREE.SRGBColorSpace
  return t
}

// place products on shelves: FRONT facings + BACK stock (behind, slightly raised so labels read)
export function computeLayout(data) {
  const DW = 6, GAP = 0.7, ROW_H = 2.6, SD = 3.1
  const doors = data.doors, maxRows = data.max_rows
  const totalW = doors * DW + (doors - 1) * GAP
  const x0 = -totalW / 2 + DW / 2
  const totalH = maxRows * ROW_H
  const y0 = totalH / 2 - ROW_H / 2

  const cells = {}
  data.items.forEach((it) => {
    const k = it.door + '_' + it.row
    if (!cells[k]) cells[k] = { front: [], back: [] }
    ;(it.back ? cells[k].back : cells[k].front).push(it)
  })

  const products = []
  const place = (arr, cx, plankY, z, scaleY, raise) => {
    const n = arr.length
    if (!n) return
    arr.sort((a, b) => a.col - b.col)
    arr.forEach((it, i) => {
      const w = Math.min(1.05, (DW / Math.max(n, 1)) * 0.86)
      const h = Math.min(ROW_H * 0.7, Math.max(w * 1.5, 1.1)) * scaleY   // chunky, not tall slab
      const d = Math.max(0.5, w * 1.05)
      const x = cx + ((i + 0.5) / n - 0.5) * DW * 0.92
      const y = plankY + 0.09 + h / 2 + raise
      products.push({ item: it, w, h, d, position: [x, y, z], key: `${it.door}-${it.row}-${it.back ? 'b' : 'f'}-${i}` })
    })
  }
  // N-deep depth from APPARENT BOX HEIGHT: within a shelf, the front facing is full-height,
  // products further back are occluded so only a shorter cap shows → shorter box = deeper row.
  // Bin height ratios into front/middle/back, compact (no gaps), and tuck each level behind.
  const FRONT_Z = SD / 2 - 0.6, DEPTH_STEP = 0.72
  // Convenience-store shelves are single-facing: box HEIGHT varies by product, not by depth,
  // so height-based front/back binning invents phantom rows. Force single-deep. (Toggle to re-enable.)
  const SINGLE_DEEP = true
  Object.keys(cells).forEach((k) => {
    const dd = +k.split('_')[0], rr = +k.split('_')[1]
    const cx = x0 + dd * (DW + GAP)
    const plankY = y0 - rr * ROW_H - ROW_H * 0.5
    const items = [...cells[k].front, ...cells[k].back]
    const fh = Math.max(...items.map((it) => it.y2 - it.y1)) || 1
    items.forEach((it) => {
      const r = (it.y2 - it.y1) / fh
      it._rd = SINGLE_DEEP ? 0 : r > 0.65 ? 0 : r > 0.38 ? 1 : 2   // raw depth (gated to single-deep)
    })
    const present = [...new Set(items.map((it) => it._rd))].sort((a, b) => a - b)
    const remap = {}; present.forEach((d, i) => { remap[d] = i })   // compact to 0,1,2… no gaps
    items.forEach((it) => { it.depth = remap[it._rd]; it.depthN = present.length })
    const byDepth = {}
    items.forEach((it) => { (byDepth[it.depth] = byDepth[it.depth] || []).push(it) })
    Object.keys(byDepth).map(Number).sort((a, b) => a - b).forEach((dep) => {
      place(byDepth[dep], cx, plankY, FRONT_Z - dep * DEPTH_STEP, 1.0, dep === 0 ? 0 : 0.08)
    })
  })

  const rowAnchors = []
  for (let rr = 0; rr < maxRows; rr++) {
    rowAnchors.push({ label: '層 ' + (rr + 1), pos: [-totalW / 2 - 1.1, y0 - rr * ROW_H, 0] })
  }

  return { DW, GAP, ROW_H, SD, doors, maxRows, totalW, totalH, x0, y0, products, rowAnchors,
    span: Math.max(totalW, totalH) }
}
