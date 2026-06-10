import { useEffect, useMemo, useRef } from 'react'
import { Canvas, useThree } from '@react-three/fiber'
import { OrbitControls, RoundedBox, Edges, ContactShadows, Html } from '@react-three/drei'
import * as THREE from 'three'
import { computeLayout, productColor, productTexture } from './lib.js'
import InfoCard from './InfoCard.jsx'

function Product({ p, data, onSelect, isSelected }) {
  const known = p.item.l2 !== '?'
  // colour by L1 category, matching the detection boxes on the left image + the legend
  const col = data.colors[p.item.l1] || [138, 147, 160]
  const tex = useMemo(() => productTexture(col, p.item.l2, p.h / p.w), [col[0], col[1], col[2], p.item.l2, p.h, p.w])
  const emissive = useMemo(() => new THREE.Color(col[0] / 255, col[1] / 255, col[2] / 255), [col[0], col[1], col[2]])
  return (
    <RoundedBox
      args={[p.w, p.h, p.d]}
      radius={0.05}
      smoothness={4}
      position={p.position}
      castShadow
      receiveShadow
      onClick={(e) => { e.stopPropagation(); onSelect(p.item) }}
      onPointerOver={(e) => { e.stopPropagation(); document.body.style.cursor = 'pointer' }}
      onPointerOut={() => { document.body.style.cursor = 'auto' }}
    >
      <meshStandardMaterial map={tex} roughness={0.46} metalness={0.08} emissive={emissive} emissiveIntensity={known ? 0.1 : 0.02} />
      {isSelected && <Edges scale={1.05} threshold={15} color="#3dd6e8" />}
    </RoundedBox>
  )
}

function Cabinet({ layout }) {
  const { totalW, totalH, SD, x0, y0, doors, maxRows, ROW_H, DW, GAP } = layout
  const plank = { color: '#6f8cb5', roughness: 0.7, metalness: 0.1 }
  const planks = []
  for (let rr = 0; rr < maxRows; rr++) planks.push(y0 - rr * ROW_H - ROW_H * 0.5)
  const dividers = []
  for (let dd = 1; dd < doors; dd++) dividers.push(x0 + dd * (DW + GAP) - (DW + GAP) / 2)
  return (
    <group>
      {planks.map((yy, i) => (
        <mesh key={i} position={[0, yy, 0]} receiveShadow castShadow>
          <boxGeometry args={[totalW + 0.5, 0.16, SD + 0.1]} />
          <meshStandardMaterial {...plank} />
        </mesh>
      ))}
      {dividers.map((dx, i) => (
        <mesh key={i} position={[dx, 0, -SD / 2]}>
          <boxGeometry args={[0.1, totalH + 0.6, 0.12]} />
          <meshStandardMaterial color="#33414f" roughness={0.6} metalness={0.3} />
        </mesh>
      ))}
    </group>
  )
}

function RowLabels({ layout }) {
  return layout.rowAnchors.map((a, i) => (
    <Html key={i} position={a.pos} className="rowlab" zIndexRange={[2, 0]} prepend>
      {a.label}
    </Html>
  ))
}

function CameraRig({ layout, resetSignal, controlsRef }) {
  const { camera } = useThree()
  useEffect(() => {
    const r = layout.span * 1.12
    camera.position.set(r * 0.14, r * 0.34, r * 1.12)
    camera.updateProjectionMatrix()
    if (controlsRef.current) {
      controlsRef.current.target.set(0, -0.5, 0)
      controlsRef.current.update()
    }
  }, [layout, resetSignal, camera, controlsRef])
  return null
}

export default function Shelf3D({ data, autoRotate, resetSignal, selected, onSelect }) {
  const controlsRef = useRef()
  const layout = useMemo(() => computeLayout(data), [data])
  const selPos = selected ? layout.products.find((p) => p.item === selected)?.position : null
  return (
    <Canvas
      shadows
      dpr={[1, 2]}
      camera={{ fov: 40, near: 0.1, far: 300, position: [12, 8, 20] }}
      gl={{ antialias: true, toneMapping: THREE.ACESFilmicToneMapping, toneMappingExposure: 1.04 }}
      onPointerMissed={() => onSelect(null)}
    >
      <color attach="background" args={['#0a1018']} />
      <hemisphereLight args={['#e8f0fb', '#3a4453', 1.0]} />
      <ambientLight intensity={0.45} />
      <directionalLight
        position={[9, 18, 13]}
        intensity={1.0}
        color="#ffffff"
        castShadow
        shadow-mapSize={[2048, 2048]}
        shadow-bias={-0.0004}
        shadow-camera-near={1}
        shadow-camera-far={90}
        shadow-camera-left={-26}
        shadow-camera-right={26}
        shadow-camera-top={26}
        shadow-camera-bottom={-26}
      />
      <directionalLight position={[-10, 8, 14]} intensity={0.5} color="#dce8f5" />

      <Cabinet layout={layout} />
      {layout.products.map((p) => (
        <Product key={p.key} p={p} data={data} onSelect={onSelect} isSelected={selected === p.item} />
      ))}
      <RowLabels layout={layout} />

      <ContactShadows position={[0, -layout.totalH / 2 - 0.78, 0]} opacity={0.5} scale={layout.totalW + 8} blur={2.4} far={6} color="#000000" />
      <gridHelper args={[130, 65, '#1d3548', '#121d29']} position={[0, -9.37, 0]} />
      <mesh rotation={[-Math.PI / 2, 0, 0]} position={[0, -9.4, 0]} receiveShadow>
        <planeGeometry args={[220, 220]} />
        <meshStandardMaterial color="#10151c" roughness={0.96} metalness={0} />
      </mesh>

      {selPos && (
        <Html position={selPos} zIndexRange={[30, 0]} style={{ pointerEvents: 'none' }}>
          <div style={{ transform: 'translate(26px, -72px)' }}>
            <InfoCard item={selected} data={data} />
          </div>
        </Html>
      )}

      <CameraRig layout={layout} resetSignal={resetSignal} controlsRef={controlsRef} />
      <OrbitControls
        ref={controlsRef}
        enableDamping
        dampingFactor={0.08}
        autoRotate={autoRotate}
        autoRotateSpeed={0.5}
        minDistance={12}
        maxDistance={80}
        maxPolarAngle={Math.PI * 0.54}
        target={[0, -0.5, 0]}
      />
    </Canvas>
  )
}
