import { useEffect, useRef, useState, useCallback } from 'react';
import { RefreshCw } from 'lucide-react';
import { useNetworkStore } from '../store/networkStore';

// Same generators as RealTraining.tsx — no server needed
function makeDataset(name: string, noise: number): [number[][], number[]] {
  const n = noise / 100;
  const N = 120;
  const X: number[][] = [], y: number[] = [];

  if (name === 'Circle') {
    for (let i = 0; i < N; i++) {
      const angle = Math.random() * 2 * Math.PI;
      const r = Math.random() < 0.5 ? 0.45 : 0.9;
      X.push([r * Math.cos(angle) + (Math.random() - .5) * n * 2, r * Math.sin(angle) + (Math.random() - .5) * n * 2]);
      y.push(r > 0.65 ? 1 : 0);
    }
  } else if (name === 'Spiral') {
    for (let c = 0; c < 2; c++) {
      for (let i = 0; i < N / 2; i++) {
        const t = (i / (N / 2)) * 3;
        const r = t + (Math.random() - .5) * n;
        const a = t * 2 + c * Math.PI;
        X.push([r * Math.cos(a), r * Math.sin(a)]);
        y.push(c);
      }
    }
  } else if (name === 'XOR') {
    for (let i = 0; i < N; i++) {
      const a = Math.random() > .5 ? 1 : -1, b = Math.random() > .5 ? 1 : -1;
      X.push([a + (Math.random() - .5) * n * 2, b + (Math.random() - .5) * n * 2]);
      y.push(a * b > 0 ? 1 : 0);
    }
  } else {
    for (let i = 0; i < N; i++) {
      const c = i < N / 2 ? 0 : 1;
      X.push([(c ? 1 : -1) + (Math.random() - .5) * (n * 2 + .5), (Math.random() - .5) * (n * 2 + .5)]);
      y.push(c);
    }
  }
  return [X, y];
}

export function DatasetPreview() {
  const { trainingConfig } = useNetworkStore();
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [data, setData] = useState<[number[][], number[]] | null>(null);

  const refresh = useCallback(() => {
    setData(makeDataset(trainingConfig.dataset, trainingConfig.noise));
  }, [trainingConfig.dataset, trainingConfig.noise]);

  useEffect(() => { refresh(); }, [refresh]);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas || !data) return;
    const ctx = canvas.getContext('2d')!;
    const [X, y] = data;
    const W = canvas.width, H = canvas.height;
    ctx.clearRect(0, 0, W, H);

    const xs = X.map(p => p[0]), ys = X.map(p => p[1]);
    const xMin = Math.min(...xs) - .2, xMax = Math.max(...xs) + .2;
    const yMin = Math.min(...ys) - .2, yMax = Math.max(...ys) + .2;
    const toCanv = (px: number, py: number) => [
      ((px - xMin) / (xMax - xMin)) * W,
      H - ((py - yMin) / (yMax - yMin)) * H,
    ] as [number, number];

    // Grid
    ctx.strokeStyle = 'rgba(55,65,81,0.3)'; ctx.lineWidth = .5;
    for (let i = 0; i <= 4; i++) {
      const x = (i / 4) * W; ctx.beginPath(); ctx.moveTo(x, 0); ctx.lineTo(x, H); ctx.stroke();
      const y2 = (i / 4) * H; ctx.beginPath(); ctx.moveTo(0, y2); ctx.lineTo(W, y2); ctx.stroke();
    }

    // Points
    X.forEach((p, i) => {
      const [cx, cy] = toCanv(p[0], p[1]);
      ctx.beginPath();
      ctx.arc(cx, cy, 3.5, 0, Math.PI * 2);
      ctx.fillStyle = y[i] === 1 ? 'rgba(59,130,246,0.85)' : 'rgba(239,68,68,0.85)';
      ctx.fill();
      ctx.strokeStyle = 'rgba(255,255,255,0.3)'; ctx.lineWidth = .5; ctx.stroke();
    });
  }, [data]);

  return (
    <div className="rounded-xl border p-2" style={{ background: 'var(--bg-card)', borderColor: 'var(--border)' }}>
      <div className="flex items-center justify-between mb-1.5">
        <span className="text-xs font-medium uppercase tracking-wide" style={{ color: 'var(--text-muted)' }}>
          {trainingConfig.dataset} · noise {trainingConfig.noise}%
        </span>
        <button onClick={refresh} className="p-1 rounded hover:opacity-70 transition-opacity" title="Resample">
          <RefreshCw size={11} style={{ color: 'var(--text-faint)' }} />
        </button>
      </div>
      <canvas ref={canvasRef} width={240} height={140} className="w-full rounded-lg" />
      <div className="flex gap-3 mt-1.5 text-xs justify-center" style={{ color: 'var(--text-faint)' }}>
        <span className="flex items-center gap-1"><span className="w-2 h-2 rounded-full bg-blue-500 inline-block" />Class 1</span>
        <span className="flex items-center gap-1"><span className="w-2 h-2 rounded-full bg-red-500 inline-block" />Class 0</span>
      </div>
    </div>
  );
}
