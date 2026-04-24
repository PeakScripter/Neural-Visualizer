import { useEffect, useRef, useMemo } from 'react';
import type { NetworkGraph } from '../../types';

interface Props { graph: NetworkGraph }

const LAYER_COLORS: Record<string, string> = {
  input: '#3b82f6', hidden: '#10b981', output: '#ef4444',
  conv: '#8b5cf6', fc: '#6366f1', rnn: '#f59e0b', lstm: '#ec4899',
  generator: '#14b8a6', discriminator: '#f97316', attention: '#fb923c',
  embedding: '#818cf8', feedforward: '#34d399', encoder: '#2dd4bf',
  decoder: '#fb7185', bottleneck: '#c084fc',
};

function lerp3(t: number, c0: [number,number,number], c1: [number,number,number], c2: [number,number,number]) {
  if (t < 0.5) {
    const s = t * 2;
    return c0.map((v, i) => Math.round(v + (c1[i] - v) * s));
  }
  const s = (t - 0.5) * 2;
  return c1.map((v, i) => Math.round(v + (c2[i] - v) * s));
}

export function LayerActivationHeatmap({ graph }: Props) {
  const canvasRef = useRef<HTMLCanvasElement>(null);

  const byLayer = useMemo(() => {
    const map = new Map<number, typeof graph.nodes>();
    graph.nodes.forEach(n => {
      if (!map.has(n.layer)) map.set(n.layer, []);
      map.get(n.layer)!.push(n);
    });
    return [...map.entries()].sort((a, b) => a[0] - b[0]);
  }, [graph.nodes]);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas || !byLayer.length) return;
    const ctx = canvas.getContext('2d')!;
    const W = canvas.width, H = canvas.height;
    ctx.clearRect(0, 0, W, H);

    const PAD_LEFT = 60, PAD_RIGHT = 20, PAD_TOP = 10, PAD_BOTTOM = 24;
    const plotW = W - PAD_LEFT - PAD_RIGHT;
    const rowH = (H - PAD_TOP - PAD_BOTTOM) / byLayer.length;

    // Global value range
    const allVals = graph.nodes.map(n => n.value ?? 0);
    const vMin = Math.min(...allVals), vMax = Math.max(...allVals);
    const vRange = vMax - vMin || 1;

    byLayer.forEach(([, nodes], ri) => {
      const y0 = PAD_TOP + ri * rowH;
      const layerType = nodes[0].layer_type;
      const accent = LAYER_COLORS[layerType] ?? '#6b7280';
      const cellW = plotW / nodes.length;

      // Layer label
      ctx.fillStyle = accent;
      ctx.font = 'bold 9px Inter, sans-serif';
      ctx.textAlign = 'right';
      ctx.textBaseline = 'middle';
      ctx.fillText(nodes[0].layer_type.slice(0, 8), PAD_LEFT - 4, y0 + rowH / 2);

      // Cells
      nodes.forEach((node, ci) => {
        const v = node.value ?? 0;
        const t = (v - vMin) / vRange;
        // cool→neutral→hot: blue(0,120,255) → dark(20,20,40) → orange(255,140,0)
        const [r, g, b] = lerp3(t,
          [0, 80, 220],
          [20, 20, 45],
          [255, 130, 0],
        );
        ctx.fillStyle = `rgb(${r},${g},${b})`;
        ctx.fillRect(PAD_LEFT + ci * cellW, y0 + 2, cellW - 0.5, rowH - 4);

        // Tick for dead neurons (value ≈ 0)
        if (Math.abs(v) < 0.01) {
          ctx.fillStyle = 'rgba(239,68,68,0.6)';
          ctx.fillRect(PAD_LEFT + ci * cellW, y0 + 2, Math.max(1, cellW - 0.5), rowH - 4);
        }
      });

      // Row border
      ctx.strokeStyle = 'rgba(55,65,81,0.5)';
      ctx.lineWidth = 0.5;
      ctx.strokeRect(PAD_LEFT, y0 + 2, plotW, rowH - 4);

      // Neuron count
      ctx.fillStyle = 'rgba(156,163,175,0.6)';
      ctx.font = '8px Inter';
      ctx.textAlign = 'left';
      ctx.textBaseline = 'top';
      ctx.fillText(`${nodes.length}n`, PAD_LEFT + plotW + 2, y0 + 2);
    });

    // Color scale bar at bottom
    const barY = H - PAD_BOTTOM + 4;
    for (let i = 0; i < plotW; i++) {
      const t = i / plotW;
      const [r, g, b] = lerp3(t, [0, 80, 220], [20, 20, 45], [255, 130, 0]);
      ctx.fillStyle = `rgb(${r},${g},${b})`;
      ctx.fillRect(PAD_LEFT + i, barY, 1, 8);
    }
    ctx.fillStyle = 'rgba(156,163,175,0.7)';
    ctx.font = '8px Inter';
    ctx.textAlign = 'left';  ctx.textBaseline = 'top';
    ctx.fillText(vMin.toFixed(2), PAD_LEFT, barY + 10);
    ctx.textAlign = 'right';
    ctx.fillText(vMax.toFixed(2), PAD_LEFT + plotW, barY + 10);
    ctx.textAlign = 'center';
    ctx.fillText('activation value', PAD_LEFT + plotW / 2, barY + 10);
  }, [byLayer, graph.nodes]);

  if (!graph.nodes.length) {
    return (
      <div className="flex items-center justify-center h-full text-sm" style={{ color: 'var(--text-muted)' }}>
        Build a network to see activations.
      </div>
    );
  }

  return (
    <div className="flex flex-col h-full gap-2">
      <div className="flex items-center justify-between">
        <span className="text-xs font-medium uppercase tracking-wide" style={{ color: 'var(--text-muted)' }}>
          Layer-by-Layer Activation Map — {graph.nodes.length} neurons
        </span>
        <div className="flex gap-2 text-xs">
          <span className="flex items-center gap-1"><span className="w-2 h-2 rounded-sm inline-block bg-blue-500" />low</span>
          <span className="flex items-center gap-1"><span className="w-2 h-2 rounded-sm inline-block bg-orange-500" />high</span>
          <span className="flex items-center gap-1"><span className="w-2 h-2 rounded-sm inline-block bg-red-600" />dead</span>
        </div>
      </div>
      <div className="flex-1 rounded-xl overflow-hidden border" style={{ borderColor: 'var(--border)' }}>
        <canvas ref={canvasRef} className="w-full h-full" width={900} height={500} />
      </div>
    </div>
  );
}
