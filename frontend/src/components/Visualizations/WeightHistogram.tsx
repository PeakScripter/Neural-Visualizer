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

function histogram(values: number[], bins: number): { x: number; count: number }[] {
  if (!values.length) return [];
  const min = Math.min(...values);
  const max = Math.max(...values);
  const range = max - min || 1;
  const w = range / bins;
  const counts = new Array(bins).fill(0);
  values.forEach(v => {
    const i = Math.min(bins - 1, Math.floor((v - min) / w));
    counts[i]++;
  });
  return counts.map((count, i) => ({ x: min + (i + 0.5) * w, count }));
}

export function WeightHistogram({ graph }: Props) {
  const canvasRef = useRef<HTMLCanvasElement>(null);

  // Group edges by layer
  const byLayer = useMemo(() => {
    const map = new Map<number, { weights: number[]; type: string }>();
    graph.edges.forEach(e => {
      const src = graph.nodes.find(n => n.id === e.source);
      const layer = src?.layer ?? e.layer;
      const type = src?.layer_type ?? 'hidden';
      if (!map.has(layer)) map.set(layer, { weights: [], type });
      map.get(layer)!.weights.push(e.weight);
    });
    return [...map.entries()].sort((a, b) => a[0] - b[0]);
  }, [graph]);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas || !byLayer.length) return;
    const ctx = canvas.getContext('2d')!;
    const W = canvas.width, H = canvas.height;
    ctx.clearRect(0, 0, W, H);

    const rows = byLayer.length;
    const rowH = H / rows;
    const BINS = 40;
    const pad = { left: 52, right: 16, top: 8, bottom: 18 };

    byLayer.forEach(([, { weights, type }], ri) => {
      const y0 = ri * rowH;
      const color = LAYER_COLORS[type] ?? '#6b7280';
      const hist = histogram(weights, BINS);
      const maxCount = Math.max(...hist.map(h => h.count), 1);
      const plotW = W - pad.left - pad.right;
      const plotH = rowH - pad.top - pad.bottom;

      // Row label
      ctx.fillStyle = color;
      ctx.font = 'bold 9px Inter, sans-serif';
      ctx.textAlign = 'right';
      ctx.textBaseline = 'middle';
      ctx.fillText(`L${ri}`, pad.left - 4, y0 + pad.top + plotH / 2);

      // Zero line
      const allW = weights;
      const wMin = Math.min(...allW), wMax = Math.max(...allW);
      const zeroX = pad.left + ((-wMin) / (wMax - wMin || 1)) * plotW;
      ctx.strokeStyle = 'rgba(255,255,255,0.12)';
      ctx.lineWidth = 0.5;
      ctx.beginPath();
      ctx.moveTo(zeroX, y0 + pad.top);
      ctx.lineTo(zeroX, y0 + pad.top + plotH);
      ctx.stroke();

      // Bars
      const barW = plotW / BINS;
      hist.forEach((h, i) => {
        const bh = (h.count / maxCount) * plotH;
        const bx = pad.left + i * barW;
        const by = y0 + pad.top + plotH - bh;
        ctx.fillStyle = h.x < 0 ? '#ef4444bb' : color + 'bb';
        ctx.fillRect(bx + 0.5, by, barW - 1, bh);
      });

      // Mean line
      const mean = weights.reduce((a, b) => a + b, 0) / (weights.length || 1);
      const meanX = pad.left + ((mean - wMin) / (wMax - wMin || 1)) * plotW;
      ctx.strokeStyle = '#facc15';
      ctx.lineWidth = 1;
      ctx.setLineDash([2, 2]);
      ctx.beginPath();
      ctx.moveTo(meanX, y0 + pad.top);
      ctx.lineTo(meanX, y0 + pad.top + plotH);
      ctx.stroke();
      ctx.setLineDash([]);

      // Stats
      ctx.fillStyle = 'rgba(156,163,175,0.8)';
      ctx.font = '8px Inter, sans-serif';
      ctx.textAlign = 'left';
      ctx.textBaseline = 'top';
      ctx.fillText(`n=${weights.length}  μ=${mean.toFixed(3)}`, pad.left + 2, y0 + pad.top + 1);

      // Row border
      ctx.strokeStyle = 'rgba(55,65,81,0.4)';
      ctx.lineWidth = 0.5;
      ctx.beginPath();
      ctx.moveTo(0, y0 + rowH - 0.5);
      ctx.lineTo(W, y0 + rowH - 0.5);
      ctx.stroke();
    });

    // Legend
    ctx.fillStyle = 'rgba(250,204,21,0.9)';
    ctx.font = '9px Inter';
    ctx.textAlign = 'left';
    ctx.fillText('— mean', W - 58, 4);
    ctx.fillStyle = '#ef4444bb';
    ctx.fillRect(W - 58, 14, 8, 8);
    ctx.fillStyle = 'rgba(156,163,175,0.7)';
    ctx.fillText('neg wts', W - 48, 14);
  }, [byLayer]);

  if (!graph.edges.length) {
    return (
      <div className="flex items-center justify-center h-full text-sm" style={{ color: 'var(--text-muted)' }}>
        Build a network to see weight distributions.
      </div>
    );
  }

  return (
    <div className="flex flex-col h-full gap-2">
      <div className="flex items-center justify-between">
        <span className="text-xs font-medium uppercase tracking-wide" style={{ color: 'var(--text-muted)' }}>
          Weight Distribution — {graph.edges.length} weights across {byLayer.length} layers
        </span>
        <span className="badge-blue text-xs">per layer</span>
      </div>
      <div className="flex-1 rounded-xl overflow-hidden border" style={{ borderColor: 'var(--border)' }}>
        <canvas ref={canvasRef} className="w-full h-full" width={900} height={600} />
      </div>
      <p className="text-xs text-center" style={{ color: 'var(--text-faint)' }}>
        Yellow dashed = mean · Red bars = negative weights · Green = positive
      </p>
    </div>
  );
}
