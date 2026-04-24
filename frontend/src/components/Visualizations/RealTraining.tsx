import { useState, useCallback, useRef, useEffect } from 'react';
import * as tf from '@tensorflow/tfjs';
import { Play, Square, Zap } from 'lucide-react';
import { useNetworkStore } from '../../store/networkStore';

/* ── Dataset generators (same distributions as Python backend) ─────── */
function makeCircleData(n = 200, noise = 0.1): [number[][], number[]] {
  const X: number[][] = [];
  const y: number[] = [];
  for (let i = 0; i < n; i++) {
    const angle = Math.random() * 2 * Math.PI;
    const r = Math.random() < 0.5 ? 0.5 : 1.0;
    const nx = r * Math.cos(angle) + (Math.random() - 0.5) * noise * 2;
    const ny = r * Math.sin(angle) + (Math.random() - 0.5) * noise * 2;
    X.push([nx, ny]);
    y.push(r > 0.75 ? 1 : 0);
  }
  return [X, y];
}

function makeSpiralData(n = 200, noise = 0.1): [number[][], number[]] {
  const X: number[][] = [];
  const y: number[] = [];
  for (let cls = 0; cls < 2; cls++) {
    for (let i = 0; i < n / 2; i++) {
      const t = (i / (n / 2)) * 3;
      const r = t + (Math.random() - 0.5) * noise;
      const angle = t * 2 + cls * Math.PI;
      X.push([r * Math.cos(angle), r * Math.sin(angle)]);
      y.push(cls);
    }
  }
  return [X, y];
}

function makeGaussianData(n = 200, noise = 0.5): [number[][], number[]] {
  const X: number[][] = [];
  const y: number[] = [];
  for (let i = 0; i < n; i++) {
    const cls = i < n / 2 ? 0 : 1;
    const cx = cls === 0 ? -1 : 1;
    X.push([cx + (Math.random() - 0.5) * noise * 2, (Math.random() - 0.5) * noise * 2]);
    y.push(cls);
  }
  return [X, y];
}

function makeXorData(n = 200, noise = 0.1): [number[][], number[]] {
  const X: number[][] = [];
  const y: number[] = [];
  for (let i = 0; i < n; i++) {
    const a = Math.random() > 0.5 ? 1 : -1;
    const b = Math.random() > 0.5 ? 1 : -1;
    X.push([
      a + (Math.random() - 0.5) * noise * 2,
      b + (Math.random() - 0.5) * noise * 2,
    ]);
    y.push(a * b > 0 ? 1 : 0);
  }
  return [X, y];
}

function getDataset(name: string, noise: number): [number[][], number[]] {
  const n = noise / 100;
  switch (name) {
    case 'Spiral':   return makeSpiralData(200, n);
    case 'Gaussian': return makeGaussianData(200, n * 2 + 0.3);
    case 'XOR':      return makeXorData(200, n);
    default:         return makeCircleData(200, n);
  }
}

/* ── Build a TF model matching the network config ──────────────────── */
function buildTFModel(neurons: number[], activations: string[], lr: number): tf.Sequential {
  const model = tf.sequential();
  model.add(tf.layers.dense({ units: neurons[0] ?? 16, inputShape: [2], activation: 'relu' }));
  for (let i = 1; i < neurons.length; i++) {
    const act = (activations[i] ?? 'relu').toLowerCase().replace('leakyrelu', 'leaky_relu');
    model.add(tf.layers.dense({ units: neurons[i], activation: act as any }));
  }
  model.add(tf.layers.dense({ units: 1, activation: 'sigmoid' }));
  model.compile({
    optimizer: tf.train.adam(lr),
    loss: 'binaryCrossentropy',
    metrics: ['accuracy'],
  });
  return model;
}

interface EpochResult {
  epoch: number;
  loss: number;
  acc: number;
}

export function RealTraining() {
  const { networkConfig, trainingConfig } = useNetworkStore();
  const [results, setResults] = useState<EpochResult[]>([]);
  const [running, setRunning] = useState(false);
  const [currentEpoch, setCurrentEpoch] = useState(0);
  const stopRef = useRef(false);
  const modelRef = useRef<tf.Sequential | null>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);

  const totalEpochs = Math.max(trainingConfig.epochs * 5, 30);

  const run = useCallback(async () => {
    stopRef.current = false;
    setResults([]);
    setRunning(true);
    setCurrentEpoch(0);

    const neurons = networkConfig.neurons.slice(0, networkConfig.n_layers);
    const activations = networkConfig.activations.slice(0, networkConfig.n_layers);
    const [X, y] = getDataset(trainingConfig.dataset, trainingConfig.noise);

    const xTensor = tf.tensor2d(X);
    const yTensor = tf.tensor1d(y, 'float32');

    modelRef.current?.dispose();
    const model = buildTFModel(neurons, activations, trainingConfig.learning_rate);
    modelRef.current = model;

    const collected: EpochResult[] = [];

    for (let ep = 0; ep < totalEpochs; ep++) {
      if (stopRef.current) break;

      const history = await model.fit(xTensor, yTensor, {
        epochs: 1,
        batchSize: trainingConfig.batch_size,
        shuffle: true,
        verbose: 0,
      });

      const loss = history.history['loss'][0] as number;
      const acc  = history.history['acc']?.[0] as number ?? history.history['accuracy']?.[0] as number ?? 0;

      collected.push({ epoch: ep + 1, loss, acc });
      setResults([...collected]);
      setCurrentEpoch(ep + 1);

      // Yield to browser between epochs
      await new Promise((r) => setTimeout(r, 0));
    }

    xTensor.dispose();
    yTensor.dispose();
    setRunning(false);
  }, [networkConfig, trainingConfig, totalEpochs]);

  const stop = useCallback(() => {
    stopRef.current = true;
  }, []);

  // Draw live chart on canvas
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas || !results.length) return;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    const w = canvas.width;
    const h = canvas.height;
    ctx.clearRect(0, 0, w, h);

    const pad = { top: 20, right: 60, bottom: 35, left: 50 };
    const cw = w - pad.left - pad.right;
    const ch = h - pad.top - pad.bottom;

    // Background
    ctx.fillStyle = 'rgba(0,0,0,0)';
    ctx.fillRect(0, 0, w, h);

    // Grid
    ctx.strokeStyle = 'rgba(55,65,81,0.5)';
    ctx.lineWidth = 0.5;
    for (let i = 0; i <= 5; i++) {
      const y2 = pad.top + (ch / 5) * i;
      ctx.beginPath(); ctx.moveTo(pad.left, y2); ctx.lineTo(pad.left + cw, y2); ctx.stroke();
    }

    const xOf = (i: number) => pad.left + (i / (totalEpochs - 1)) * cw;
    const maxLoss = Math.max(...results.map((r) => r.loss), 0.01);

    // Loss curve (red)
    ctx.strokeStyle = '#ef4444';
    ctx.lineWidth = 2;
    ctx.beginPath();
    results.forEach((r, i) => {
      const x2 = xOf(r.epoch - 1);
      const y2 = pad.top + ch - (r.loss / maxLoss) * ch;
      i === 0 ? ctx.moveTo(x2, y2) : ctx.lineTo(x2, y2);
    });
    ctx.stroke();

    // Fill under loss
    ctx.fillStyle = 'rgba(239,68,68,0.08)';
    ctx.beginPath();
    results.forEach((r, i) => {
      const x2 = xOf(r.epoch - 1);
      const y2 = pad.top + ch - (r.loss / maxLoss) * ch;
      i === 0 ? ctx.moveTo(x2, y2) : ctx.lineTo(x2, y2);
    });
    ctx.lineTo(xOf(results.at(-1)!.epoch - 1), pad.top + ch);
    ctx.lineTo(xOf(0), pad.top + ch);
    ctx.closePath();
    ctx.fill();

    // Accuracy curve (green) — right axis
    ctx.strokeStyle = '#10b981';
    ctx.lineWidth = 2;
    ctx.beginPath();
    results.forEach((r, i) => {
      const x2 = xOf(r.epoch - 1);
      const y2 = pad.top + ch - r.acc * ch;
      i === 0 ? ctx.moveTo(x2, y2) : ctx.lineTo(x2, y2);
    });
    ctx.stroke();

    // Axis labels
    ctx.fillStyle = '#6b7280';
    ctx.font = '10px Inter, sans-serif';
    ctx.textAlign = 'center';
    ctx.fillText('Epoch', pad.left + cw / 2, h - 4);
    ctx.save(); ctx.translate(12, pad.top + ch / 2);
    ctx.rotate(-Math.PI / 2);
    ctx.fillStyle = '#ef4444';
    ctx.fillText('Loss', 0, 0);
    ctx.restore();
    ctx.save(); ctx.translate(w - 10, pad.top + ch / 2);
    ctx.rotate(-Math.PI / 2);
    ctx.fillStyle = '#10b981';
    ctx.fillText('Accuracy', 0, 0);
    ctx.restore();

    // Last values
    const last = results.at(-1)!;
    ctx.fillStyle = '#ef4444';
    ctx.font = 'bold 10px Inter, sans-serif';
    ctx.textAlign = 'left';
    ctx.fillText(`${last.loss.toFixed(4)}`, pad.left + cw + 2, pad.top + ch - (last.loss / maxLoss) * ch + 4);
    ctx.fillStyle = '#10b981';
    ctx.fillText(`${(last.acc * 100).toFixed(1)}%`, pad.left + cw + 2, pad.top + ch - last.acc * ch + 4);
  }, [results, totalEpochs]);

  const last = results.at(-1);

  return (
    <div className="flex flex-col h-full gap-3">
      {/* Stats row */}
      <div className="grid grid-cols-4 gap-2">
        {[
          { label: 'Epoch', value: `${currentEpoch}/${totalEpochs}`, color: 'text-blue-400' },
          { label: 'Loss',  value: last ? last.loss.toFixed(4) : '—',            color: 'text-red-400' },
          { label: 'Acc',   value: last ? `${(last.acc * 100).toFixed(1)}%` : '—', color: 'text-green-400' },
          { label: 'Status', value: running ? 'Training…' : results.length ? 'Done' : 'Ready', color: running ? 'text-yellow-400' : 'text-gray-400' },
        ].map((s) => (
          <div key={s.label} className="rounded-lg p-2.5 text-center border" style={{ background: 'var(--bg-card)', borderColor: 'var(--border)' }}>
            <div className={`text-base font-bold ${s.color}`}>{s.value}</div>
            <div className="text-xs mt-0.5" style={{ color: 'var(--text-faint)' }}>{s.label}</div>
          </div>
        ))}
      </div>

      {/* Live chart */}
      <div className="flex-1 relative rounded-xl overflow-hidden border" style={{ borderColor: 'var(--border)' }}>
        {!results.length && (
          <div className="absolute inset-0 flex flex-col items-center justify-center gap-2" style={{ color: 'var(--text-muted)' }}>
            <Zap size={28} style={{ color: 'var(--accent)' }} />
            <p className="text-sm">Click "Run Real Training" to train in-browser with TensorFlow.js</p>
          </div>
        )}
        <canvas
          ref={canvasRef}
          className="w-full h-full"
          width={900}
          height={400}
        />
      </div>

      {/* Progress bar */}
      {running && (
        <div className="h-1.5 rounded-full overflow-hidden" style={{ background: 'var(--border)' }}>
          <div
            className="h-full rounded-full transition-all duration-300"
            style={{ width: `${(currentEpoch / totalEpochs) * 100}%`, background: 'var(--accent)' }}
          />
        </div>
      )}

      {/* Controls */}
      <div className="flex gap-2">
        <button
          onClick={run}
          disabled={running}
          className="btn-primary flex items-center gap-2 flex-1"
        >
          <Play size={14} />
          Run Real Training ({totalEpochs} epochs, TF.js)
        </button>
        {running && (
          <button onClick={stop} className="btn-secondary flex items-center gap-1.5 px-4">
            <Square size={14} />
            Stop
          </button>
        )}
      </div>

      <p className="text-xs text-center" style={{ color: 'var(--text-faint)' }}>
        Training runs entirely in your browser via TensorFlow.js — no server needed
      </p>
    </div>
  );
}
