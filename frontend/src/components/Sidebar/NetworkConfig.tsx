import { useState } from 'react';
import { Network, ChevronDown, ChevronUp, Layers } from 'lucide-react';
import { useNetworkStore, MODEL_LOSSES } from '../../store/networkStore';
import type { ModelType, ActivationType, RegType } from '../../types';

const MODEL_TYPES: { value: ModelType; label: string; color: string }[] = [
  { value: 'ANN', label: 'ANN', color: 'blue' },
  { value: 'CNN', label: 'CNN', color: 'green' },
  { value: 'RNN', label: 'RNN', color: 'purple' },
  { value: 'LSTM', label: 'LSTM', color: 'pink' },
  { value: 'GAN', label: 'GAN', color: 'orange' },
  { value: 'Transformer', label: 'Transformer', color: 'teal' },
  { value: 'Diffuser', label: 'Diffuser', color: 'red' },
];

const ACTIVATIONS: ActivationType[] = ['ReLU', 'Sigmoid', 'Tanh', 'LeakyReLU', 'ELU'];
const REG_TYPES: RegType[] = ['None', 'L1', 'L2', 'L1L2'];
const ALL_NEURON_OPTIONS = [4, 8, 16, 32, 64, 96, 128];

function maxNeuronsForLayers(n: number) {
  return Math.min(128, Math.floor(192 / n));
}

export function NetworkConfig() {
  const { networkConfig, setNetworkConfig } = useNetworkStore();
  const [expanded, setExpanded] = useState(true);

  const losses = MODEL_LOSSES[networkConfig.model_type];
  const maxNeurons = maxNeuronsForLayers(networkConfig.n_layers);
  const neuronOptions = ALL_NEURON_OPTIONS.filter((n) => n <= maxNeurons);

  const updateNeurons = (layerIdx: number, value: number) => {
    const neurons = [...networkConfig.neurons];
    neurons[layerIdx] = value;
    setNetworkConfig({ neurons });
  };

  const updateActivation = (layerIdx: number, value: ActivationType) => {
    const activations = [...networkConfig.activations];
    activations[layerIdx] = value;
    setNetworkConfig({ activations });
  };

  const handleModelChange = (model: ModelType) => {
    const newLosses = MODEL_LOSSES[model];
    setNetworkConfig({ model_type: model, loss_fn: newLosses[0] });
  };

  return (
    <div className="card" data-tour="network-config-panel">
      <div
        className="card-header cursor-pointer select-none"
        onClick={() => setExpanded(!expanded)}
      >
        <Network size={14} className="text-blue-400" />
        <span className="text-sm font-semibold text-gray-200 flex-1">Network Configuration</span>
        {expanded ? <ChevronUp size={14} className="text-gray-500" /> : <ChevronDown size={14} className="text-gray-500" />}
      </div>

      {expanded && (
        <div className="p-4 space-y-4">
          {/* Model type grid */}
          <div data-tour="model-type">
            <label className="label-base">Model Type</label>
            <div className="grid grid-cols-4 gap-1">
              {MODEL_TYPES.map((m) => (
                <button
                  key={m.value}
                  onClick={() => handleModelChange(m.value)}
                  className={`py-1.5 px-2 rounded-md text-xs font-medium transition-all duration-150 ${
                    networkConfig.model_type === m.value
                      ? 'bg-blue-600 text-white shadow-lg'
                      : 'bg-gray-800 text-gray-400 hover:bg-gray-700 hover:text-gray-200'
                  }`}
                >
                  {m.label}
                </button>
              ))}
            </div>
          </div>

          {/* Loss function */}
          <div data-tour="loss-fn">
            <label className="label-base">Loss Function</label>
            <select
              className="select-base"
              value={networkConfig.loss_fn}
              onChange={(e) => setNetworkConfig({ loss_fn: e.target.value })}
            >
              {losses.map((l) => (
                <option key={l} value={l}>{l}</option>
              ))}
            </select>
          </div>

          {/* Input / Output nodes */}
          <div className="grid grid-cols-2 gap-2" data-tour="input-output">
            <div>
              <div className="flex justify-between items-center mb-1.5">
                <label className="label-base mb-0">Input Nodes</label>
                <span className="text-sm font-bold" style={{ color: 'var(--accent)' }}>{networkConfig.input_nodes}</span>
              </div>
              <input
                type="range"
                min={1} max={16} step={1}
                value={networkConfig.input_nodes}
                onChange={(e) => setNetworkConfig({ input_nodes: +e.target.value })}
                className="w-full"
              />
              <div className="flex justify-between text-xs mt-0.5" style={{ color: 'var(--text-faint)' }}>
                <span>1</span><span>16</span>
              </div>
            </div>
            <div>
              <div className="flex justify-between items-center mb-1.5">
                <label className="label-base mb-0">Output Nodes</label>
                <span className="text-sm font-bold" style={{ color: 'var(--accent)' }}>{networkConfig.output_nodes}</span>
              </div>
              <input
                type="range"
                min={1} max={10} step={1}
                value={networkConfig.output_nodes}
                onChange={(e) => setNetworkConfig({ output_nodes: +e.target.value })}
                className="w-full"
              />
              <div className="flex justify-between text-xs mt-0.5" style={{ color: 'var(--text-faint)' }}>
                <span>1</span><span>10</span>
              </div>
            </div>
          </div>

          {/* Number of layers */}
          <div data-tour="hidden-layers">
            <div className="flex justify-between items-center mb-1.5">
              <label className="label-base mb-0">Hidden Layers</label>
              <span className="text-sm font-bold" style={{ color: 'var(--accent)' }}>{networkConfig.n_layers}</span>
            </div>
            <input
              type="range"
              min={1} max={5} step={1}
              value={networkConfig.n_layers}
              onChange={(e) => {
                const n = +e.target.value;
                const newMax = maxNeuronsForLayers(n);
                const neurons = networkConfig.neurons.map((v) =>
                  v > newMax ? newMax : v
                );
                setNetworkConfig({ n_layers: n, neurons });
              }}
              className="w-full"
            />
            <div className="flex justify-between text-xs mt-1" style={{ color: 'var(--text-faint)' }}>
              {[1, 2, 3, 4, 5].map((n) => <span key={n}>{n}</span>)}
            </div>
            <p className="text-xs mt-1.5" style={{ color: 'var(--text-faint)' }}>
              Max {maxNeurons} neurons/layer at {networkConfig.n_layers} {networkConfig.n_layers === 1 ? 'layer' : 'layers'} · budget: ~{networkConfig.n_layers * maxNeurons} nodes
            </p>
          </div>

          {/* Layer controls */}
          <div data-tour="layer-config">
            <div className="flex items-center gap-2 mb-2">
              <Layers size={12} className="text-gray-500" />
              <label className="label-base mb-0">Layer Configuration</label>
            </div>
            <div className="space-y-3">
              {Array.from({ length: networkConfig.n_layers }).map((_, i) => (
                <div key={i} className="bg-gray-800/60 rounded-lg p-3 border border-gray-700/50">
                  <div className="text-xs font-medium mb-2" style={{ color: 'var(--text-muted)' }}>
                    Layer {i + 1}
                    <span className="ml-2" style={{ color: 'var(--accent)' }}>
                      {networkConfig.neurons[i] ?? 32} neurons
                    </span>
                  </div>
                  <div className="grid grid-cols-2 gap-2">
                    <div>
                      <label className="label-base">Neurons</label>
                      <select
                        className="select-base text-xs py-1.5"
                        value={Math.min(networkConfig.neurons[i] ?? 32, maxNeurons)}
                        onChange={(e) => updateNeurons(i, +e.target.value)}
                      >
                        {neuronOptions.map((n) => (
                          <option key={n} value={n}>{n}</option>
                        ))}
                      </select>
                    </div>
                    <div>
                      <label className="label-base">Activation</label>
                      <select
                        className="select-base text-xs py-1.5"
                        value={networkConfig.activations[i] ?? 'ReLU'}
                        onChange={(e) => updateActivation(i, e.target.value as ActivationType)}
                      >
                        {ACTIVATIONS.map((a) => (
                          <option key={a} value={a}>{a}</option>
                        ))}
                      </select>
                    </div>
                  </div>
                </div>
              ))}
            </div>
          </div>

          {/* Regularization */}
          <div className="grid grid-cols-2 gap-2" data-tour="regularization">
            <div>
              <label className="label-base">Regularization</label>
              <select
                className="select-base text-xs py-1.5"
                value={networkConfig.reg_type}
                onChange={(e) => setNetworkConfig({ reg_type: e.target.value as RegType })}
              >
                {REG_TYPES.map((r) => <option key={r} value={r}>{r}</option>)}
              </select>
            </div>
            {networkConfig.reg_type !== 'None' && (
              <div>
                <div className="flex justify-between items-center mb-1">
                  <label className="label-base mb-0">Rate</label>
                  <span className="text-xs text-blue-400">{networkConfig.reg_rate}</span>
                </div>
                <input
                  type="range" min={0.0001} max={0.1} step={0.0001}
                  value={networkConfig.reg_rate}
                  onChange={(e) => setNetworkConfig({ reg_rate: +e.target.value })}
                />
              </div>
            )}
          </div>
        </div>
      )}
    </div>
  );
}
