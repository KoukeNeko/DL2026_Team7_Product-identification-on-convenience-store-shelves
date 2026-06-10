import { confColor, L1_EN } from './lib.js'

export default function InfoCard({ item, data }) {
  const col = data.colors[item.l1] || [125, 144, 164]
  const cc = confColor(item.conf)
  return (
    <div className="infocard">
      <h5>選取位置</h5>
      <div className="pos">層 {item.row + 1} · 位置 {String(item.col + 1).padStart(2, '0')}</div>
      <div className="fld">
        <div className="lab">預測商品</div>
        <div className="prod">{item.l2 === '?' ? <span style={{ color: '#7d90a4' }}>未辨識</span> : '🏷 ' + item.l2}</div>
      </div>
      <div className="fld">
        <div className="lab">信心度</div>
        <div className="pct" style={{ color: cc }}>{item.conf}%</div>
        <div className="bar"><i style={{ width: item.conf + '%', background: cc }} /></div>
      </div>
      <div className="fld">
        <div className="lab">排面</div>
        {(() => {
          const dn = item.depthN || 1, dp = item.depth || 0
          const label = dn <= 1 || dp === 0 ? '前排 facing' : dp === dn - 1 ? '後排（庫存）' : `第 ${dp + 1} 排`
          const bg = dp === 0 ? '#2f8f6b' : dp === dn - 1 ? '#5b6b7d' : '#8a7d3f'
          return <span className="cat" style={{ background: bg }}>{label}{dn > 1 ? `　${dn} 排深` : ''}</span>
        })()}
      </div>
      <div className="fld">
        <div className="lab">類別</div>
        <span className="cat" style={{ background: `rgb(${col[0]},${col[1]},${col[2]})` }}>{L1_EN[item.l1] || item.l1}</span>
      </div>
    </div>
  )
}
