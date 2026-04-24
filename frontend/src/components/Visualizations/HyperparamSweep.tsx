import { useState, useCallback, useRef, useEffect } from 'react';
import * as tf from '@tensorflow/tfjs';
import { Play, Square } from 'lucide-react';
import { useNetworkStore } from '../../store/networkStore';

const SWEEP_LRS = [0.0001, 0.001, 0.01, 0.05, 0.1];
const COLORS    = ['#3b82f6', '#10b981', '#f59e0b', '#ef4444', '#a855f7'];

function makeDataset(name: string, noise: number): [number[][], number[]] {
  const n = noise / 100, N = 150;
  const X: number[][] = [], y: number[] = [];
  if (name === 'Circle') {
    for (let i = 0; i < N; i++) {
      const a = Math.random() * 2 * Math.PI, r = Math.random() < .5 ? .45 : .9;
      X.push([r*Math.cos(a)+(Math.random()-.5)*n*2, r*Math.sin(a)+(Math.random()-.5)*n*2]);
      y.push(r > .65 ? 1 : 0);
    }
  } else if (name === 'Spiral') {
    for (let c = 0; c < 2; c++) for (let i = 0; i < N/2; i++) {
      const t = (i/(N/2))*3, r = t+(Math.random()-.5)*n, a = t*2+c*Math.PI;
      X.push([r*Math.cos(a), r*Math.sin(a)]); y.push(c);
    }
  } else {
    for (let i = 0; i < N; i++) {
      const a = Math.random()>.5?1:-1, b = Math.random()>.5?1:-1;
      X.push([a+(Math.random()-.5)*n*2, b+(Math.random()-.5)*n*2]); y.push(a*b>0?1:0);
    }
  }
  return [X, y];
}

function buildModel(neurons: number[], activations: string[], lr: number) {
  const m = tf.sequential();
  m.add(tf.layers.dense({ units: neurons[0]??16, inputShape:[2], activation:'relu' }));
  for (let i = 1; i < neurons.length; i++) {
    m.add(tf.layers.dense({ units: neurons[i], activation: (activations[i]??'relu').toLowerCase() as any }));
  }
  m.add(tf.layers.dense({ units:1, activation:'sigmoid' }));
  m.compile({ optimizer: tf.train.adam(lr), loss:'binaryCrossentropy', metrics:['accuracy'] });
  return m;
}

interface RunResult { lr: number; losses: number[]; accs: number[]; color: string }

export function HyperparamSweep() {
  const { networkConfig, trainingConfig } = useNetworkStore();
  const [results, setResults] = useState<RunResult[]>([]);
  const [running, setRunning] = useState(false);
  const [progress, setProgress] = useState(0);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const stopRef = useRef(false);
  const EPOCHS = 40;

  const run = useCallback(async () => {
    stopRef.current = false;
    setResults([]); setRunning(true); setProgress(0);
    const neurons = networkConfig.neurons.slice(0, networkConfig.n_layers);
    const activations = networkConfig.activations.slice(0, networkConfig.n_layers);
    const [X, y] = makeDataset(trainingConfig.dataset, trainingConfig.noise);
    const xT = tf.tensor2d(X), yT = tf.tensor1d(y, 'float32');
    const collected: RunResult[] = [];

    for (let ri = 0; ri < SWEEP_LRS.length; ri++) {
      if (stopRef.current) break;
      const lr = SWEEP_LRS[ri];
      const model = buildModel(neurons, activations, lr);
      const losses: number[] = [], accs: number[] = [];

      for (let ep = 0; ep < EPOCHS; ep++) {
        if (stopRef.current) break;
        const h = await model.fit(xT, yT, { epochs:1, batchSize:32, shuffle:true, verbose:0 });
        losses.push(h.history['loss'][0] as number);
        accs.push((h.history['acc']?.[0] ?? h.history['accuracy']?.[0] ?? 0) as number);
        setProgress(((ri * EPOCHS + ep + 1) / (SWEEP_LRS.length * EPOCHS)) * 100);
        await new Promise(r => setTimeout(r, 0));
      }
      collected.push({ lr, losses, accs, color: COLORS[ri] });
      setResults([...collected]);
      model.dispose();
    }
    xT.dispose(); yT.dispose();
    setRunning(false);
  }, [networkConfig, trainingConfig]);

  // Draw chart whenever results update
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas || !results.length) return;
    const ctx = canvas.getContext('2d')!;
    const W = canvas.width, H = canvas.height;
    ctx.clearRect(0, 0, W, H);

    const pad = { top: 24, right: 100, bottom: 36, left: 52 };
    const cw = W - pad.left - pad.right, ch = H - pad.top - pad.bottom;

    // Grid
    ctx.strokeStyle = 'rgba(55,65,81,0.4)'; ctx.lineWidth = .5;
    for (let i = 0; i <= 4; i++) {
      const y2 = pad.top + (ch / 4) * i;
      ctx.beginPath(); ctx.moveTo(pad.left, y2); ctx.lineTo(pad.left + cw, y2); ctx.stroke();
    }

    const maxLoss = Math.max(...results.flatMap(r => r.losses), 0.01);

    results.forEach(({ losses, accs, color, lr }) => {
      // Loss (solid)
      ctx.strokeStyle = color; ctx.lineWidth = 1.8; ctx.setLineDash([]);
      ctx.beginPath();
      losses.forEach((l, i) => {
        const x2 = pad.left + (i / (EPOCHS - 1)) * cw;
        const y2 = pad.top + ch - (l / maxLoss) * ch;
        i === 0 ? ctx.moveTo(x2, y2) : ctx.lineTo(x2, y2);
      });
      ctx.stroke();

      // Accuracy (dashed)
      ctx.strokeStyle = color; ctx.lineWidth = 1; ctx.setLineDash([4, 3]);
      ctx.beginPath();
      accs.forEach((a, i) => {
        const x2 = pad.left + (i / (EPOCHS - 1)) * cw;
        const y2 = pad.top + ch - a * ch;
        i === 0 ? ctx.moveTo(x2, y2) : ctx.lineTo(x2, y2);
      });
      ctx.stroke();
      ctx.setLineDash([]);

      // Right label
      if (losses.length) {
        const lastLoss = losses.at(-1)!;
        const labelY = pad.top + ch - (lastLoss / maxLoss) * ch;
        ctx.fillStyle = color; ctx.font = 'bold 9px Inter'; ctx.textAlign = 'left'; ctx.textBaseline = 'middle';
        ctx.fillText(`lr=${lr}`, pad.left + cw + 4, labelY);
      }
    });

    // Axis labels
    ctx.fillStyle = 'rgba(156,163,175,0.7)'; ctx.font = '10px Inter'; ctx.textAlign = 'center'; ctx.textBaseline = 'bottom';
    ctx.fillText('Epoch', pad.left + cw / 2, H - 2);
    ctx.save(); ctx.translate(12, pad.top + ch / 2); ctx.rotate(-Math.PI/2);
    ctx.fillText('Loss (solid) / Acc (dashed)', 0, 0); ctx.restore();
  }, [results]);

  return (
    <div className="flex flex-col h-full gap-3">
      <div className="flex items-center justify-between">
        <div>
          <span className="text-sm font-semibold" style={{ color: 'var(--text-primary)' }}>Learning Rate Sweep</span>
          <span className="text-xs ml-2" style={{ color: 'var(--text-faint)' }}>
            {SWEEP_LRS.map((lr, i) => (
              <span key={lr} style={{ color: COLORS[i] }} className="mr-2">lr={lr}</span>
            ))}
          </span>
        </div>
        <div className="flex gap-2">
          <button onClick={run} disabled={running} className="btn-primary flex items-center gap-1.5 text-sm px-4">
            <Play size={13} /> Sweep {SWEEP_LRS.length} LRs
          </button>
          {running && (
            <button onClick={() => { stopRef.current = true; }} className="btn-secondary flex items-center gap-1.5 text-sm">
              <Square size={13} /> Stop
            </button>
          )}
        </div>
      </div>

      {running && (
        <div className="h-1.5 rounded-full overflow-hidden" style={{ background: 'var(--border)' }}>
          <div className="h-full rounded-full transition-all duration-200" style={{ width: `${progress}%`, background: 'var(--accent)' }} />
        </div>
      )}

      <div className="flex-1 relative rounded-xl border overflow-hidden" style={{ borderColor: 'var(--border)' }}>
        {!results.length && (
          <div className="absolute inset-0 flex items-center justify-center text-sm" style={{ color: 'var(--text-muted)' }}>
            Click "Sweep" to train with {SWEEP_LRS.length} learning rates simultaneously
          </div>
        )}
        <canvas ref={canvasRef} className="w-full h-full" width={900} height={420} />
      </div>

      {results.length > 0 && (
        <div className="grid grid-cols-5 gap-2">
          {results.map(({ lr, losses, accs, color }) => (
            <div key={lr} className="rounded-lg p-2 text-center border text-xs" style={{ background: color + '11', borderColor: color + '44' }}>
              <div className="font-bold" style={{ color }}>lr={lr}</div>
              <div style={{ color: 'var(--text-muted)' }}>loss {losses.at(-1)?.toFixed(3)}</div>
              <div style={{ color: 'var(--text-faint)' }}>{((accs.at(-1)??0)*100).toFixed(0)}% acc</div>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
