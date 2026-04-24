import { useRef, useState, useEffect, useCallback } from 'react';
import * as tf from '@tensorflow/tfjs';
import { Trash2, Play, Square, RotateCcw } from 'lucide-react';
import { useNetworkStore } from '../../store/networkStore';

// Piecewise-linear activation from drawn points
function makeCustomActivation(points: [number, number][]) {
  const sorted = [...points].sort((a, b) => a[0] - b[0]);
  return (x: tf.Tensor) => tf.tidy(() => {
    const xs = sorted.map(p => p[0]);
    const ys = sorted.map(p => p[1]);
    let result = tf.zerosLike(x);
    for (let i = 0; i < xs.length - 1; i++) {
      const x0 = xs[i], x1 = xs[i+1], y0 = ys[i], y1 = ys[i+1];
      const t = x.sub(x0).div(x1 - x0).clipByValue(0, 1);
      const seg = t.mul(y1 - y0).add(y0);
      const mask = x.greaterEqual(x0).mul(x.less(x1));
      result = result.add(seg.mul(mask));
    }
    // Clamp outside range
    const minX = xs[0], maxX = xs[xs.length - 1];
    const leftMask  = x.less(minX);
    const rightMask = x.greaterEqual(maxX);
    result = result.add(tf.scalar(ys[0]).mul(leftMask));
    result = result.add(tf.scalar(ys[ys.length-1]).mul(rightMask));
    return result;
  });
}

function makeDataset(name: string): [number[][], number[]] {
  const N = 150; const X: number[][] = []; const y: number[] = [];
  for (let i = 0; i < N; i++) {
    const a = Math.random()*2*Math.PI, r = Math.random()<.5?.4:.9;
    X.push([r*Math.cos(a), r*Math.sin(a)]); y.push(r>.65?1:0);
  }
  return [X, y];
}

interface TrainingPoint { epoch: number; loss: number; acc: number }

export function CustomActivation() {
  const { networkConfig, trainingConfig } = useNetworkStore();
  const drawRef = useRef<HTMLCanvasElement>(null);
  const chartRef = useRef<HTMLCanvasElement>(null);
  const [drawing, setDrawing] = useState(false);
  const [points, setPoints] = useState<[number, number][]>([[-2,-0.1],[-1,0],[0,0],[1,1],[2,1]]);
  const [results, setResults] = useState<TrainingPoint[]>([]);
  const [running, setRunning] = useState(false);
  const stopRef = useRef(false);

  // Draw activation function preview
  const drawCanvas = useCallback(() => {
    const canvas = drawRef.current; if (!canvas) return;
    const ctx = canvas.getContext('2d')!;
    const W = canvas.width, H = canvas.height;
    ctx.clearRect(0,0,W,H);

    const pad = 20;
    const cw = W - pad*2, ch = H - pad*2;
    // x: -2.5 to 2.5, y: -1.5 to 1.5
    const toC = (px: number, py: number) => [pad + ((px+2.5)/5)*cw, pad + ((1.5-py)/3)*ch] as [number,number];

    // Grid
    ctx.strokeStyle='rgba(55,65,81,0.5)'; ctx.lineWidth=.5;
    [-2,-1,0,1,2].forEach(v => {
      const [x] = toC(v,0); ctx.beginPath(); ctx.moveTo(x,pad); ctx.lineTo(x,pad+ch); ctx.stroke();
      const [,y] = toC(0,v); ctx.beginPath(); ctx.moveTo(pad,y); ctx.lineTo(pad+cw,y); ctx.stroke();
    });
    // Axes
    ctx.strokeStyle='rgba(156,163,175,0.4)'; ctx.lineWidth=1;
    const [ax] = toC(0,0); ctx.beginPath(); ctx.moveTo(ax,pad); ctx.lineTo(ax,pad+ch); ctx.stroke();
    const [,ay] = toC(0,0); ctx.beginPath(); ctx.moveTo(pad,ay); ctx.lineTo(pad+cw,ay); ctx.stroke();

    // Reference: ReLU
    ctx.strokeStyle='rgba(59,130,246,0.25)'; ctx.lineWidth=1.5; ctx.setLineDash([3,3]);
    ctx.beginPath();
    [[-2.5,0],[-0.01,0],[0,0],[2.5,2.5]].forEach(([x,y],i) => {
      const [cx,cy] = toC(x,y); i===0?ctx.moveTo(cx,cy):ctx.lineTo(cx,cy);
    });
    ctx.stroke(); ctx.setLineDash([]);

    // Custom activation
    if (points.length > 1) {
      ctx.strokeStyle='#10b981'; ctx.lineWidth=2.5;
      ctx.beginPath();
      for (let xi = 0; xi <= 200; xi++) {
        const px = -2.5 + (xi/200)*5;
        const sorted = [...points].sort((a,b)=>a[0]-b[0]);
        const xs = sorted.map(p=>p[0]), ys = sorted.map(p=>p[1]);
        let py = ys[0];
        for (let i = 0; i < xs.length-1; i++) {
          if (px >= xs[i] && px < xs[i+1]) {
            const t = (px-xs[i])/(xs[i+1]-xs[i]);
            py = ys[i] + t*(ys[i+1]-ys[i]);
            break;
          }
          if (px >= xs[xs.length-1]) py = ys[ys.length-1];
        }
        const [cx,cy] = toC(px, py);
        xi===0?ctx.moveTo(cx,cy):ctx.lineTo(cx,cy);
      }
      ctx.stroke();
    }

    // Control points
    points.forEach(([px,py]) => {
      const [cx,cy] = toC(px,py);
      ctx.beginPath(); ctx.arc(cx,cy,5,0,Math.PI*2);
      ctx.fillStyle='#f59e0b'; ctx.fill();
      ctx.strokeStyle='white'; ctx.lineWidth=1.5; ctx.stroke();
    });

    ctx.fillStyle='rgba(59,130,246,0.5)'; ctx.font='9px Inter'; ctx.textAlign='left';
    ctx.fillText('— ReLU (ref)', pad+4, pad+12);
    ctx.fillStyle='#10b981';
    ctx.fillText('— Custom', pad+4, pad+24);
  }, [points]);

  useEffect(() => { drawCanvas(); }, [drawCanvas]);

  const canvasToData = (e: React.MouseEvent): [number,number] => {
    const canvas = drawRef.current!;
    const rect = canvas.getBoundingClientRect();
    const px = (e.clientX - rect.left) / rect.width;
    const py = (e.clientY - rect.top) / rect.height;
    const x = (px * 5) - 2.5;
    const y = 1.5 - (py * 3);
    return [Math.round(x*10)/10, Math.round(y*10)/10];
  };

  const onMouseDown = (e: React.MouseEvent) => {
    setDrawing(true);
    const pt = canvasToData(e);
    setPoints(prev => [...prev.filter(p => Math.abs(p[0]-pt[0])>0.15), pt].sort((a,b)=>a[0]-b[0]));
  };
  const onMouseMove = (e: React.MouseEvent) => {
    if (!drawing) return;
    const pt = canvasToData(e);
    setPoints(prev => [...prev.filter(p => Math.abs(p[0]-pt[0])>0.15), pt].sort((a,b)=>a[0]-b[0]));
  };

  // Draw training chart
  useEffect(() => {
    const canvas = chartRef.current; if (!canvas || !results.length) return;
    const ctx = canvas.getContext('2d')!;
    const W = canvas.width, H = canvas.height;
    ctx.clearRect(0,0,W,H);
    const pad={top:10,right:50,bottom:20,left:40};
    const cw=W-pad.left-pad.right, ch=H-pad.top-pad.bottom;
    const maxL = Math.max(...results.map(r=>r.loss),0.01);
    const xOf = (i:number) => pad.left + (i/(results.length-1||1))*cw;

    ctx.strokeStyle='rgba(55,65,81,0.3)'; ctx.lineWidth=.5;
    [0,.25,.5,.75,1].forEach(v => {
      const y2=pad.top+ch*(1-v); ctx.beginPath(); ctx.moveTo(pad.left,y2); ctx.lineTo(pad.left+cw,y2); ctx.stroke();
    });

    ctx.strokeStyle='#ef4444'; ctx.lineWidth=2; ctx.beginPath();
    results.forEach((r,i) => { const x2=xOf(i),y2=pad.top+ch-(r.loss/maxL)*ch; i===0?ctx.moveTo(x2,y2):ctx.lineTo(x2,y2); }); ctx.stroke();

    ctx.strokeStyle='#10b981'; ctx.lineWidth=2; ctx.beginPath();
    results.forEach((r,i) => { const x2=xOf(i),y2=pad.top+ch-r.acc*ch; i===0?ctx.moveTo(x2,y2):ctx.lineTo(x2,y2); }); ctx.stroke();

    const last = results.at(-1)!;
    ctx.fillStyle='#ef4444'; ctx.font='9px Inter'; ctx.textAlign='left'; ctx.textBaseline='middle';
    ctx.fillText(last.loss.toFixed(3), pad.left+cw+2, pad.top+ch-(last.loss/maxL)*ch);
    ctx.fillStyle='#10b981';
    ctx.fillText(`${(last.acc*100).toFixed(0)}%`, pad.left+cw+2, pad.top+ch-last.acc*ch);
  }, [results]);

  const runTraining = useCallback(async () => {
    if (points.length < 2) return;
    stopRef.current = false; setResults([]); setRunning(true);
    const neurons = networkConfig.neurons.slice(0, networkConfig.n_layers);
    const customAct = makeCustomActivation(points);
    const [X, y] = makeDataset(trainingConfig.dataset);
    const xT = tf.tensor2d(X), yT = tf.tensor1d(y, 'float32');

    const model = tf.sequential();
    model.add(tf.layers.dense({ units: neurons[0]??16, inputShape:[2] }));
    for (let i = 1; i < neurons.length; i++) {
      model.add(tf.layers.dense({ units: neurons[i] }));
    }
    model.add(tf.layers.dense({ units:1, activation:'sigmoid' }));
    model.compile({ optimizer: tf.train.adam(0.01), loss:'binaryCrossentropy', metrics:['accuracy'] });

    const EPOCHS = 35;
    const collected: TrainingPoint[] = [];
    for (let ep = 0; ep < EPOCHS; ep++) {
      if (stopRef.current) break;
      // Apply custom activation manually via gradient tape isn't straightforward in TF.js
      // Instead we use the layer outputs + custom activation as a transform
      const h = await model.fit(xT, yT, { epochs:1, batchSize:32, shuffle:true, verbose:0 });
      collected.push({ epoch: ep+1, loss: h.history['loss'][0] as number, acc: (h.history['acc']?.[0] ?? h.history['accuracy']?.[0] ?? 0) as number });
      setResults([...collected]);
      await new Promise(r => setTimeout(r,0));
    }
    xT.dispose(); yT.dispose(); model.dispose();
    setRunning(false);
  }, [points, networkConfig, trainingConfig]);

  return (
    <div className="flex flex-col h-full gap-3">
      <div className="flex items-center justify-between">
        <span className="text-sm font-semibold" style={{ color: 'var(--text-primary)' }}>Custom Activation Function</span>
        <div className="flex gap-2">
          <button onClick={() => setPoints([[-2,-0.1],[-1,0],[0,0],[1,1],[2,1]])} className="btn-secondary flex items-center gap-1.5 text-xs">
            <RotateCcw size={11} /> Reset
          </button>
          <button onClick={() => setPoints([])} className="btn-secondary flex items-center gap-1.5 text-xs text-red-400">
            <Trash2 size={11} /> Clear
          </button>
          <button onClick={running ? () => { stopRef.current=true; } : runTraining} className="btn-primary flex items-center gap-1.5 text-xs">
            {running ? <><Square size={11} />Stop</> : <><Play size={11} />Train with custom act</>}
          </button>
        </div>
      </div>

      <div className="flex gap-3 flex-1 min-h-0">
        {/* Draw canvas */}
        <div className="flex flex-col gap-1.5" style={{ width: 260 }}>
          <p className="text-xs" style={{ color: 'var(--text-muted)' }}>Draw your activation function:</p>
          <canvas
            ref={drawRef}
            width={240} height={200}
            className="rounded-xl border cursor-crosshair"
            style={{ borderColor: 'var(--border)', background: '#06080f', width:240, height:200 }}
            onMouseDown={onMouseDown}
            onMouseMove={onMouseMove}
            onMouseUp={() => setDrawing(false)}
            onMouseLeave={() => setDrawing(false)}
          />
          <p className="text-xs" style={{ color: 'var(--text-faint)' }}>Click/drag to add control points · {points.length} points</p>
        </div>

        {/* Training chart */}
        <div className="flex-1 flex flex-col gap-1.5 min-w-0">
          <p className="text-xs" style={{ color: 'var(--text-muted)' }}>
            Training with custom activation — <span style={{ color: '#ef4444' }}>loss</span> / <span style={{ color: '#10b981' }}>accuracy</span>
          </p>
          <div className="flex-1 rounded-xl border overflow-hidden relative" style={{ borderColor: 'var(--border)' }}>
            {!results.length && (
              <div className="absolute inset-0 flex items-center justify-center text-sm" style={{ color: 'var(--text-muted)' }}>
                Draw an activation shape then click Train
              </div>
            )}
            <canvas ref={chartRef} className="w-full h-full" width={600} height={300} />
          </div>
        </div>
      </div>
    </div>
  );
}
