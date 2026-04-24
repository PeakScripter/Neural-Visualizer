import { useRef, useMemo, useState } from 'react';
import { Canvas, useFrame } from '@react-three/fiber';
import { OrbitControls, Line } from '@react-three/drei';
import * as THREE from 'three';
import type { NetworkGraph, NetworkNode, NetworkEdge } from '../../types';

const LAYER_COLORS: Record<string, string> = {
  input:         '#3b82f6',
  hidden:        '#10b981',
  output:        '#ef4444',
  conv:          '#8b5cf6',
  fc:            '#6366f1',
  rnn:           '#f59e0b',
  lstm:          '#ec4899',
  generator:     '#14b8a6',
  discriminator: '#f97316',
  gen_output:    '#06b6d4',
  latent:        '#a78bfa',
  attention:     '#fb923c',
  embedding:     '#818cf8',
  feedforward:   '#34d399',
  encoder:       '#2dd4bf',
  decoder:       '#fb7185',
  bottleneck:    '#c084fc',
};

// ── Node sphere ─────────────────────────────────────────────────────────────
function NodeSphere({
  node,
  position,
  isActive,
  selected,
  onClick,
}: {
  node: NetworkNode;
  position: [number, number, number];
  isActive: boolean;
  selected: boolean;
  onClick: () => void;
}) {
  const meshRef = useRef<THREE.Mesh>(null);
  const hex = LAYER_COLORS[node.layer_type] ?? '#6b7280';
  const color = useMemo(() => new THREE.Color(hex), [hex]);

  // Tiny radius: 0.07 base, slightly larger for IO nodes
  const r = node.layer_type === 'output' ? 0.1 : node.layer_type === 'input' ? 0.09 : 0.07;

  useFrame((_, delta) => {
    if (!meshRef.current) return;
    if (isActive) {
      const t = Date.now() * 0.003;
      meshRef.current.scale.setScalar(1 + Math.sin(t * 4) * 0.12);
    } else {
      meshRef.current.scale.lerp(new THREE.Vector3(1, 1, 1), delta * 8);
    }
  });

  return (
    <group position={position} onClick={(e) => { e.stopPropagation(); onClick(); }}>
      <mesh ref={meshRef}>
        <sphereGeometry args={[r, 12, 12]} />
        <meshStandardMaterial
          color={color}
          emissive={color}
          emissiveIntensity={selected ? 1.8 : isActive ? 0.9 : 0.3}
          roughness={0.3}
          metalness={0.5}
        />
      </mesh>

      {/* ring around selected node */}
      {selected && (
        <mesh rotation={[Math.PI / 2, 0, 0]}>
          <torusGeometry args={[r * 2.2, 0.012, 8, 32]} />
          <meshBasicMaterial color={color} transparent opacity={0.8} />
        </mesh>
      )}
    </group>
  );
}

// ── Edge line (no per-edge particle — too expensive at 2k+ edges) ───────────
function EdgeLine({
  from, to, weight, isActive, edgeType,
}: {
  from: [number, number, number];
  to: [number, number, number];
  weight: number;
  isActive: boolean;
  edgeType?: string;
}) {
  const color = edgeType === 'skip'
    ? '#3b82f6'
    : edgeType === 'recurrent'
    ? '#f59e0b'
    : edgeType === 'attention'
    ? '#fb923c'
    : weight > 0 ? '#34d399' : '#f87171';

  const opacity = isActive
    ? Math.min(0.75, 0.15 + Math.abs(weight) * 0.5)
    : 0.04;

  return (
    <Line
      points={[from, to]}
      color={color}
      lineWidth={isActive ? 1.2 : 0.4}
      transparent
      opacity={opacity}
    />
  );
}

// ── Travelling signal particles (only for ≤300 active edges) ────────────────
function SignalParticles({
  edges,
  nodePositions,
  activeEdgeIds,
}: {
  edges: NetworkEdge[];
  nodePositions: Map<number, [number, number, number]>;
  activeEdgeIds?: Set<number>;
}) {
  const active = useMemo(() => {
    if (!activeEdgeIds) return [];
    return edges.map((e, i) => ({ e, i })).filter(({ i }) => activeEdgeIds.has(i));
  }, [edges, activeEdgeIds]);

  if (!active.length) return null;
  return (
    <>
      {active.map(({ e, i }) => {
        const from = nodePositions.get(e.source);
        const to = nodePositions.get(e.target);
        if (!from || !to) return null;
        return <Particle key={i} from={from} to={to} offset={i * 0.13} />;
      })}
    </>
  );
}

function Particle({
  from, to, offset,
}: {
  from: [number, number, number];
  to: [number, number, number];
  offset: number;
}) {
  const ref = useRef<THREE.Mesh>(null);
  const t = useRef((offset % 1 + 1) % 1);

  useFrame((_, delta) => {
    if (!ref.current) return;
    t.current = (t.current + delta * 1.1) % 1;
    const p = t.current;
    ref.current.position.set(
      from[0] + (to[0] - from[0]) * p,
      from[1] + (to[1] - from[1]) * p,
      from[2] + (to[2] - from[2]) * p,
    );
    (ref.current.material as THREE.MeshBasicMaterial).opacity = Math.sin(p * Math.PI) * 0.85;
  });

  return (
    <mesh ref={ref}>
      <sphereGeometry args={[0.025, 6, 6]} />
      <meshBasicMaterial color="#ffffff" transparent depthWrite={false} />
    </mesh>
  );
}

// ── Starfield ────────────────────────────────────────────────────────────────
function Starfield() {
  const positions = useMemo(() => {
    const arr = new Float32Array(500 * 3);
    for (let i = 0; i < 500; i++) {
      arr[i * 3]     = (Math.random() - 0.5) * 60;
      arr[i * 3 + 1] = (Math.random() - 0.5) * 60;
      arr[i * 3 + 2] = (Math.random() - 0.5) * 60;
    }
    return arr;
  }, []);
  return (
    <points>
      <bufferGeometry>
        <bufferAttribute attach="attributes-position" args={[positions, 3]} />
      </bufferGeometry>
      <pointsMaterial color="#94a3b8" size={0.035} transparent opacity={0.35} sizeAttenuation />
    </points>
  );
}


// ── Tooltip HTML overlay (avoids 3D Text performance hit) ───────────────────
function NodeTooltip({
  node,
  position,
  camera,
  size,
}: {
  node: NetworkNode;
  position: [number, number, number];
  camera: THREE.Camera;
  size: { width: number; height: number };
}) {
  const vec = new THREE.Vector3(...position).project(camera);
  const x = (vec.x * 0.5 + 0.5) * size.width;
  const y = (-vec.y * 0.5 + 0.5) * size.height;

  return (
    <div
      style={{
        position: 'absolute',
        left: x + 12,
        top: y - 10,
        pointerEvents: 'none',
        background: 'rgba(9,13,19,0.92)',
        border: '1px solid #374151',
        borderRadius: 8,
        padding: '7px 10px',
        fontSize: 11,
        color: '#e2e8f0',
        whiteSpace: 'nowrap',
        zIndex: 20,
      }}
    >
      <div style={{ color: '#60a5fa', fontWeight: 700 }}>{node.name}</div>
      <div style={{ color: '#9ca3af' }}>Type: <span style={{ color: '#a3e635' }}>{node.layer_type}</span></div>
      <div style={{ color: '#9ca3af' }}>Value: <span style={{ color: '#34d399' }}>{node.value?.toFixed(4)}</span></div>
      {node.activation && <div style={{ color: '#9ca3af' }}>Act: <span style={{ color: '#fb923c' }}>{node.activation}</span></div>}
    </div>
  );
}

// ── Scene ────────────────────────────────────────────────────────────────────
function Scene({
  graph,
  activeNodeIds,
  activeEdgeIds,
  autoOrbit,
  selected,
  setSelected,
}: {
  graph: NetworkGraph;
  activeNodeIds?: Set<number>;
  activeEdgeIds?: Set<number>;
  autoOrbit: boolean;
  selected: NetworkNode | null;
  setSelected: (n: NetworkNode | null) => void;
}) {
  // ── Position mapping ─────────────────────────────────────────────────────
  const nodePositions = useMemo(() => {
    if (!graph.nodes.length) return new Map<number, [number, number, number]>();

    // Group by layer
    const byLayer = new Map<number, NetworkNode[]>();
    graph.nodes.forEach((n) => {
      if (!byLayer.has(n.layer)) byLayer.set(n.layer, []);
      byLayer.get(n.layer)!.push(n);
    });

    const numLayers = byLayer.size;
    const maxPerLayer = Math.max(...[...byLayer.values()].map((a) => a.length));

    // Scale the scene so it always fits: target ~12 units wide, 6 tall
    const xSpread = Math.min(14, numLayers * 2.5);
    const ySpread = Math.min(8, maxPerLayer * 0.55);

    const map = new Map<number, [number, number, number]>();
    [...byLayer.entries()].forEach(([layerIdx, nodes]) => {
      const x = numLayers <= 1 ? 0 : ((layerIdx / (numLayers - 1)) - 0.5) * xSpread;
      nodes.forEach((n, i) => {
        const y = nodes.length <= 1 ? 0 : ((i / (nodes.length - 1)) - 0.5) * ySpread;
        // Slight Z wobble to break up flat look without being extreme
        const z = Math.sin(n.id * 2.39996) * 0.25;
        map.set(n.id, [x, y, z]);
      });
    });
    return map;
  }, [graph.nodes]);

  const edgesToRender = graph.edges;

  return (
    <>
      <color attach="background" args={['#060a10']} />

      <ambientLight intensity={0.4} />
      <pointLight position={[8, 6, 8]}   intensity={1.2} color="#60a5fa" />
      <pointLight position={[-8, -6, 4]} intensity={0.7} color="#a78bfa" />

      <Starfield />

      {/* Edges */}
      {edgesToRender.map((edge: NetworkEdge, i: number) => {
        const src = nodePositions.get(edge.source);
        const tgt = nodePositions.get(edge.target);
        if (!src || !tgt) return null;
        const isActive = activeEdgeIds ? activeEdgeIds.has(i) : true;
        return (
          <EdgeLine key={i} from={src} to={tgt} weight={edge.weight} isActive={isActive} edgeType={edge.type} />
        );
      })}

      {/* Signal particles only when stepping through propagation */}
      <SignalParticles edges={graph.edges} nodePositions={nodePositions} activeEdgeIds={activeEdgeIds} />

      {/* Nodes */}
      {graph.nodes.map((node: NetworkNode) => {
        const pos = nodePositions.get(node.id);
        if (!pos) return null;
        const isActive = activeNodeIds ? activeNodeIds.has(node.id) : true;
        return (
          <NodeSphere
            key={node.id}
            node={node}
            position={pos}
            isActive={isActive}
            selected={selected?.id === node.id}
            onClick={() => setSelected(selected?.id === node.id ? null : node)}
          />
        );
      })}

      <OrbitControls
        makeDefault
        enableDamping
        dampingFactor={0.08}
        autoRotate={autoOrbit}
        autoRotateSpeed={0.8}
        enablePan={false}
        zoomToCursor
      />
    </>
  );
}

// ── Public component ─────────────────────────────────────────────────────────
interface Props {
  graph: NetworkGraph;
  activeNodeIds?: Set<number>;
  activeEdgeIds?: Set<number>;
  mode?: 'architecture' | 'forward' | 'backward';
}

export function Network3DView({ graph, activeNodeIds, activeEdgeIds }: Props) {
  const [autoOrbit, setAutoOrbit] = useState(true);
  const [selected, setSelected] = useState<NetworkNode | null>(null);
  const canvasRef = useRef<HTMLDivElement>(null);

  if (!graph.nodes.length) {
    return (
      <div className="flex flex-col items-center justify-center h-full gap-3" style={{ color: 'var(--text-muted)' }}>
        <div className="w-16 h-16 rounded-2xl flex items-center justify-center" style={{ background: 'var(--bg-card)' }}>
          <svg width="32" height="32" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5">
            <circle cx="5" cy="12" r="2" /><circle cx="19" cy="5" r="2" /><circle cx="19" cy="19" r="2" />
            <line x1="7" y1="12" x2="17" y2="6" /><line x1="7" y1="12" x2="17" y2="18" />
          </svg>
        </div>
        <p className="text-sm font-medium" style={{ color: 'var(--text-muted)' }}>No network built yet</p>
        <p className="text-xs" style={{ color: 'var(--text-faint)' }}>Configure and click "Build Network"</p>
      </div>
    );
  }

  return (
    <div ref={canvasRef} className="w-full h-full relative" onClick={() => setSelected(null)}>
      <Canvas
        camera={{ position: [0, 3, 16], fov: 52 }}
        gl={{ antialias: true, alpha: false }}
        style={{ borderRadius: '0.5rem' }}
        onClick={(e) => e.stopPropagation()}
      >
        <Scene
          graph={graph}
          activeNodeIds={activeNodeIds}
          activeEdgeIds={activeEdgeIds}
          autoOrbit={autoOrbit}
          selected={selected}
          setSelected={setSelected}
        />
      </Canvas>

      {/* HTML tooltip — positioned over canvas using CSS absolute */}
      {selected && (() => {
        const hex = LAYER_COLORS[selected.layer_type] ?? '#6b7280';
        return (
          <div
            style={{
              position: 'absolute',
              top: 12,
              left: '50%',
              transform: 'translateX(-50%)',
              background: 'rgba(9,13,19,0.94)',
              border: `1px solid ${hex}55`,
              borderRadius: 10,
              padding: '8px 14px',
              fontSize: 12,
              color: '#e2e8f0',
              whiteSpace: 'nowrap',
              pointerEvents: 'none',
              zIndex: 20,
              display: 'flex',
              gap: 14,
            }}
          >
            <span style={{ color: hex, fontWeight: 700 }}>{selected.name}</span>
            <span style={{ color: '#9ca3af' }}>type: <b style={{ color: '#a3e635' }}>{selected.layer_type}</b></span>
            <span style={{ color: '#9ca3af' }}>val: <b style={{ color: '#34d399' }}>{selected.value?.toFixed(4)}</b></span>
            {selected.activation && <span style={{ color: '#9ca3af' }}>act: <b style={{ color: '#fb923c' }}>{selected.activation}</b></span>}
            {selected.bias !== undefined && <span style={{ color: '#9ca3af' }}>bias: <b style={{ color: '#c084fc' }}>{selected.bias.toFixed(4)}</b></span>}
          </div>
        );
      })()}

      {/* Controls */}
      <div className="absolute top-3 right-3">
        <button
          onClick={(e) => { e.stopPropagation(); setAutoOrbit((v) => !v); }}
          className="text-xs px-2.5 py-1.5 rounded-lg border transition-all"
          style={{
            background: 'rgba(6,10,16,0.8)',
            borderColor: autoOrbit ? 'var(--accent)' : 'var(--border-soft)',
            color: autoOrbit ? 'var(--accent)' : 'var(--text-muted)',
            backdropFilter: 'blur(6px)',
          }}
        >
          {autoOrbit ? '⏸ Stop' : '▶ Orbit'}
        </button>
      </div>

      <div
        className="absolute bottom-3 left-3 text-xs px-2 py-1 rounded-md border"
        style={{ background: 'rgba(6,10,16,0.8)', borderColor: 'var(--border)', color: 'var(--text-faint)', backdropFilter: 'blur(6px)' }}
      >
        Drag: rotate · Scroll: zoom to cursor
      </div>
    </div>
  );
}
