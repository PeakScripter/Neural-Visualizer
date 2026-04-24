import { useEffect, useRef } from 'react';
import type { NetworkGraph, PropStep } from '../../types';

interface Props {
  graph: NetworkGraph;
  currentStep: PropStep | null;
  modelType: string;
}

export function AttentionHeatmap({ graph, currentStep, modelType }: Props) {
  const canvasRef = useRef<HTMLCanvasElement>(null);

  // Only relevant for Transformer / attention-having models
  const attentionNodes = graph.nodes.filter((n) => n.layer_type === 'attention');

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    const n = Math.max(attentionNodes.length, 4);
    // Generate synthetic attention weights biased by node values and current step
    const matrix: number[][] = Array.from({ length: n }, (_, i) =>
      Array.from({ length: n }, (_, j) => {
        const base = Math.exp(-Math.abs(i - j) * 0.5);
        const nodeVal = attentionNodes[i]?.value ?? Math.random();
        const step = currentStep?.step ?? 0;
        return Math.max(0, base + nodeVal * 0.3 + Math.sin(i + j + step) * 0.1);
      }),
    );

    // Normalize rows
    for (let i = 0; i < n; i++) {
      const row = matrix[i];
      const sum = row.reduce((a, b) => a + b, 0) || 1;
      matrix[i] = row.map((v) => v / sum);
    }

    const w = canvas.width;
    const h = canvas.height;
    ctx.clearRect(0, 0, w, h);

    const pad = 40;
    const cellW = (w - pad * 2) / n;
    const cellH = (h - pad * 2) / n;

    // Draw cells
    for (let i = 0; i < n; i++) {
      for (let j = 0; j < n; j++) {
        const v = matrix[i][j];
        const x = pad + j * cellW;
        const y = pad + i * cellH;

        // Heat colorscale: dark → purple → orange → yellow
        const r = Math.round(v * 255);
        const g = Math.round(v * v * 180);
        const b = Math.round((1 - v) * 220);
        ctx.fillStyle = `rgba(${r},${g},${b},0.9)`;
        ctx.fillRect(x, y, cellW - 1, cellH - 1);

        // Value text
        if (n <= 8) {
          ctx.fillStyle = v > 0.4 ? '#000' : '#fff';
          ctx.font = `${Math.min(11, cellW * 0.3)}px Inter, sans-serif`;
          ctx.textAlign = 'center';
          ctx.textBaseline = 'middle';
          ctx.fillText(v.toFixed(2), x + cellW / 2, y + cellH / 2);
        }
      }
    }

    // Labels
    const labels = attentionNodes.length
      ? attentionNodes.map((nd) => nd.name.slice(0, 6))
      : Array.from({ length: n }, (_, i) => `Head ${i + 1}`);

    ctx.font = '10px Inter, sans-serif';
    ctx.fillStyle = '#9ca3af';
    ctx.textAlign = 'center';
    ctx.textBaseline = 'middle';

    for (let j = 0; j < n; j++) {
      ctx.fillText(labels[j] ?? `T${j}`, pad + j * cellW + cellW / 2, pad / 2);
    }
    for (let i = 0; i < n; i++) {
      ctx.textAlign = 'right';
      ctx.fillText(labels[i] ?? `T${i}`, pad - 4, pad + i * cellH + cellH / 2);
    }

    // Color bar
    const barX = w - 18;
    const barH = h - pad * 2;
    for (let py = 0; py < barH; py++) {
      const v = 1 - py / barH;
      const r = Math.round(v * 255);
      const g = Math.round(v * v * 180);
      const b = Math.round((1 - v) * 220);
      ctx.fillStyle = `rgb(${r},${g},${b})`;
      ctx.fillRect(barX, pad + py, 10, 1);
    }
    ctx.fillStyle = '#9ca3af';
    ctx.font = '9px Inter';
    ctx.textAlign = 'left';
    ctx.textBaseline = 'top';
    ctx.fillText('1.0', barX + 12, pad);
    ctx.textBaseline = 'bottom';
    ctx.fillText('0.0', barX + 12, pad + barH);
  }, [attentionNodes, currentStep, graph]);

  if (modelType !== 'Transformer' && !attentionNodes.length) {
    return (
      <div className="flex items-center justify-center h-full text-sm" style={{ color: 'var(--text-muted)' }}>
        Select Transformer model to see attention weights.
      </div>
    );
  }

  return (
    <div className="flex flex-col h-full gap-2">
      <div className="flex items-center justify-between">
        <span className="text-xs font-medium uppercase tracking-wide" style={{ color: 'var(--text-muted)' }}>
          Attention Weight Matrix
        </span>
        <span className="badge-orange text-xs">
          {attentionNodes.length || 4} heads · step {(currentStep?.step ?? 0) + 1}
        </span>
      </div>
      <div className="flex-1 relative rounded-xl overflow-hidden border" style={{ borderColor: 'var(--border)' }}>
        <canvas ref={canvasRef} className="w-full h-full" width={480} height={380} />
      </div>
      <p className="text-xs text-center" style={{ color: 'var(--text-faint)' }}>
        Row = query token · Column = key token · Color intensity = attention weight
      </p>
    </div>
  );
}
