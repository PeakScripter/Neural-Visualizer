import { useState, useCallback } from 'react';
import { Loader2, Zap, GitCompare } from 'lucide-react';
import { Network3DView } from './Network3DView';
import * as api from '../../api/client';
import type { NetworkGraph, NetworkConfig, ModelType, ActivationType } from '../../types';

const DEFAULTS: NetworkConfig = {
  model_type: 'ANN',
  n_layers: 3,
  neurons: [32, 32, 32, 32, 32],
  activations: ['ReLU', 'ReLU', 'ReLU', 'ReLU', 'ReLU'],
  loss_fn: 'Binary Cross Entropy',
  reg_type: 'None',
  reg_rate: 0.01,
  input_nodes: 2,
  output_nodes: 1,
};

const MODEL_TYPES: ModelType[] = ['ANN', 'CNN', 'RNN', 'LSTM', 'GAN', 'Transformer', 'Diffuser'];
const ACT_TYPES: ActivationType[] = ['ReLU', 'Sigmoid', 'Tanh', 'LeakyReLU', 'ELU'];

function MiniConfig({
  cfg,
  onChange,
  onBuild,
  loading,
  label,
  color,
}: {
  cfg: NetworkConfig;
  onChange: (c: Partial<NetworkConfig>) => void;
  onBuild: () => void;
  loading: boolean;
  label: string;
  color: string;
}) {
  return (
    <div className="flex flex-col gap-2 p-3 rounded-xl border" style={{ borderColor: color + '55', background: color + '08' }}>
      <div className="flex items-center gap-2">
        <div className="w-2.5 h-2.5 rounded-full" style={{ background: color }} />
        <span className="text-sm font-semibold" style={{ color }}>{label}</span>
      </div>

      <div className="grid grid-cols-2 gap-2">
        <div>
          <label className="label-base">Model</label>
          <select
            value={cfg.model_type}
            onChange={(e) => onChange({ model_type: e.target.value as ModelType })}
            className="select-base text-xs py-1"
          >
            {MODEL_TYPES.map((m) => <option key={m}>{m}</option>)}
          </select>
        </div>
        <div>
          <label className="label-base">Layers</label>
          <input
            type="number" min={1} max={5}
            value={cfg.n_layers}
            onChange={(e) => onChange({ n_layers: +e.target.value })}
            className="input-base text-xs py-1"
          />
        </div>
        <div>
          <label className="label-base">Neurons/layer</label>
          <input
            type="number" min={4} max={128}
            value={cfg.neurons[0]}
            onChange={(e) => {
              const n = +e.target.value;
              onChange({ neurons: Array(5).fill(n) });
            }}
            className="input-base text-xs py-1"
          />
        </div>
        <div>
          <label className="label-base">Activation</label>
          <select
            value={cfg.activations[0]}
            onChange={(e) => {
              const a = e.target.value as ActivationType;
              onChange({ activations: Array(5).fill(a) });
            }}
            className="select-base text-xs py-1"
          >
            {ACT_TYPES.map((a) => <option key={a}>{a}</option>)}
          </select>
        </div>
      </div>

      <button
        onClick={onBuild}
        disabled={loading}
        className="btn-primary flex items-center justify-center gap-1.5 text-xs py-1.5"
        style={{ background: color }}
      >
        {loading ? <Loader2 size={12} className="animate-spin" /> : <Zap size={12} />}
        Build
      </button>
    </div>
  );
}

export function CompareView() {
  const [cfgA, setCfgA] = useState<NetworkConfig>({ ...DEFAULTS, model_type: 'ANN' });
  const [cfgB, setCfgB] = useState<NetworkConfig>({ ...DEFAULTS, model_type: 'Transformer', n_layers: 2 });
  const [graphA, setGraphA] = useState<NetworkGraph>({ nodes: [], edges: [] });
  const [graphB, setGraphB] = useState<NetworkGraph>({ nodes: [], edges: [] });
  const [loadingA, setLoadingA] = useState(false);
  const [loadingB, setLoadingB] = useState(false);

  const buildA = useCallback(async () => {
    setLoadingA(true);
    try {
      const cfg = { ...cfgA, neurons: cfgA.neurons.slice(0, cfgA.n_layers), activations: cfgA.activations.slice(0, cfgA.n_layers) };
      const g = await api.buildNetwork(cfg);
      setGraphA(g);
    } finally { setLoadingA(false); }
  }, [cfgA]);

  const buildB = useCallback(async () => {
    setLoadingB(true);
    try {
      const cfg = { ...cfgB, neurons: cfgB.neurons.slice(0, cfgB.n_layers), activations: cfgB.activations.slice(0, cfgB.n_layers) };
      const g = await api.buildNetwork(cfg);
      setGraphB(g);
    } finally { setLoadingB(false); }
  }, [cfgB]);

  const statsOf = (g: NetworkGraph) => ({
    nodes: g.nodes.length,
    edges: g.edges.length,
    params: g.edges.length + g.nodes.length,
  });

  const sA = statsOf(graphA);
  const sB = statsOf(graphB);

  return (
    <div className="flex flex-col h-full gap-3">
      <div className="flex items-center gap-2">
        <GitCompare size={16} style={{ color: 'var(--accent)' }} />
        <span className="text-sm font-semibold" style={{ color: 'var(--text-primary)' }}>Side-by-Side Architecture Compare</span>
      </div>

      {/* Config row */}
      <div className="grid grid-cols-2 gap-3">
        <MiniConfig cfg={cfgA} onChange={(c) => setCfgA((p) => ({ ...p, ...c }))} onBuild={buildA} loading={loadingA} label="Network A" color="#3b82f6" />
        <MiniConfig cfg={cfgB} onChange={(c) => setCfgB((p) => ({ ...p, ...c }))} onBuild={buildB} loading={loadingB} label="Network B" color="#a855f7" />
      </div>

      {/* Stat comparison bar */}
      {(graphA.nodes.length > 0 || graphB.nodes.length > 0) && (
        <div className="grid grid-cols-3 gap-2 text-xs text-center">
          {(['nodes', 'edges', 'params'] as const).map((k) => (
            <div key={k} className="rounded-lg p-2 border" style={{ background: 'var(--bg-card)', borderColor: 'var(--border)' }}>
              <div className="flex justify-between mb-1">
                <span style={{ color: '#3b82f6' }}>{sA[k]}</span>
                <span style={{ color: 'var(--text-faint)' }}>{k}</span>
                <span style={{ color: '#a855f7' }}>{sB[k]}</span>
              </div>
              <div className="h-1.5 rounded-full overflow-hidden" style={{ background: 'var(--border)' }}>
                <div
                  className="h-full rounded-full"
                  style={{
                    width: `${(sA[k] / Math.max(sA[k] + sB[k], 1)) * 100}%`,
                    background: 'linear-gradient(90deg, #3b82f6, #a855f7)',
                  }}
                />
              </div>
            </div>
          ))}
        </div>
      )}

      {/* 3D side-by-side */}
      <div className="flex-1 grid grid-cols-2 gap-3 min-h-0">
        <div className="rounded-xl overflow-hidden border relative" style={{ borderColor: '#3b82f655' }}>
          <div className="absolute top-2 left-2 z-10 text-xs px-2 py-0.5 rounded-md" style={{ background: 'rgba(59,130,246,0.2)', color: '#93c5fd', border: '1px solid #3b82f655' }}>
            A · {cfgA.model_type}
          </div>
          <Network3DView graph={graphA} />
        </div>
        <div className="rounded-xl overflow-hidden border relative" style={{ borderColor: '#a855f755' }}>
          <div className="absolute top-2 left-2 z-10 text-xs px-2 py-0.5 rounded-md" style={{ background: 'rgba(168,85,247,0.2)', color: '#c4b5fd', border: '1px solid #a855f755' }}>
            B · {cfgB.model_type}
          </div>
          <Network3DView graph={graphB} />
        </div>
      </div>
    </div>
  );
}
