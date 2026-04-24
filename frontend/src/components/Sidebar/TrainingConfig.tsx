import { useState } from 'react';
import { Settings2, ChevronDown, ChevronUp } from 'lucide-react';
import { useNetworkStore } from '../../store/networkStore';
import type { DatasetType, ProblemType, RegType } from '../../types';

const DATASETS: DatasetType[] = ['Circle', 'Gaussian', 'XOR', 'Spiral'];
const BATCH_SIZES = [1, 8, 16, 32, 64, 128];
const LR_VALUES = [0.0001, 0.001, 0.01, 0.1];
const EPOCH_MARKS = [1, 2, 5, 10, 20, 50];

function formatLR(v: number) {
  if (v < 0.001) return '1e-4';
  if (v < 0.01) return '1e-3';
  if (v < 0.1) return '1e-2';
  return '1e-1';
}

export function TrainingConfig() {
  const { trainingConfig, setTrainingConfig } = useNetworkStore();
  const [expanded, setExpanded] = useState(true);

  return (
    <div className="card" data-tour="training-config-panel">
      <div
        className="card-header cursor-pointer select-none"
        onClick={() => setExpanded(!expanded)}
      >
        <Settings2 size={14} className="text-purple-400" />
        <span className="text-sm font-semibold text-gray-200 flex-1">Training Parameters</span>
        {expanded ? <ChevronUp size={14} className="text-gray-500" /> : <ChevronDown size={14} className="text-gray-500" />}
      </div>

      {expanded && (
        <div className="p-4 space-y-4">
          {/* Problem type */}
          <div>
            <label className="label-base">Problem Type</label>
            <div className="grid grid-cols-2 gap-1">
              {(['Classification', 'Regression'] as ProblemType[]).map((p) => (
                <button
                  key={p}
                  onClick={() => setTrainingConfig({ problem_type: p })}
                  className={`py-1.5 rounded-md text-xs font-medium transition-all ${
                    trainingConfig.problem_type === p
                      ? 'bg-purple-600 text-white'
                      : 'bg-gray-800 text-gray-400 hover:bg-gray-700'
                  }`}
                >
                  {p}
                </button>
              ))}
            </div>
          </div>

          {/* Dataset */}
          <div data-tour="dataset">
            <label className="label-base">Dataset</label>
            <div className="grid grid-cols-2 gap-1">
              {DATASETS.map((d) => (
                <button
                  key={d}
                  onClick={() => setTrainingConfig({ dataset: d })}
                  className={`py-1.5 rounded-md text-xs font-medium transition-all ${
                    trainingConfig.dataset === d
                      ? 'bg-teal-600 text-white'
                      : 'bg-gray-800 text-gray-400 hover:bg-gray-700'
                  }`}
                >
                  {d}
                </button>
              ))}
            </div>
          </div>

          {/* Noise */}
          <div>
            <div className="flex justify-between items-center mb-1.5">
              <label className="label-base mb-0">Noise Level</label>
              <span className="text-teal-400 text-sm font-bold">{trainingConfig.noise}%</span>
            </div>
            <input
              type="range" min={0} max={20} step={5}
              value={trainingConfig.noise}
              onChange={(e) => setTrainingConfig({ noise: +e.target.value })}
            />
            <div className="flex justify-between text-xs text-gray-600 mt-1">
              {[0, 5, 10, 15, 20].map((n) => <span key={n}>{n}%</span>)}
            </div>
          </div>

          {/* Batch size */}
          <div>
            <label className="label-base">Batch Size</label>
            <div className="grid grid-cols-3 gap-1">
              {BATCH_SIZES.map((b) => (
                <button
                  key={b}
                  onClick={() => setTrainingConfig({ batch_size: b })}
                  className={`py-1 rounded text-xs font-medium transition-all ${
                    trainingConfig.batch_size === b
                      ? 'bg-blue-600 text-white'
                      : 'bg-gray-800 text-gray-400 hover:bg-gray-700'
                  }`}
                >
                  {b === 1 ? 'SGD' : b}
                </button>
              ))}
            </div>
          </div>

          {/* Learning rate */}
          <div data-tour="lr-batch-epochs">
            <label className="label-base">Learning Rate</label>
            <div className="grid grid-cols-4 gap-1">
              {LR_VALUES.map((lr) => (
                <button
                  key={lr}
                  onClick={() => setTrainingConfig({ learning_rate: lr })}
                  className={`py-1 rounded text-xs font-medium transition-all ${
                    trainingConfig.learning_rate === lr
                      ? 'bg-orange-600 text-white'
                      : 'bg-gray-800 text-gray-400 hover:bg-gray-700'
                  }`}
                >
                  {formatLR(lr)}
                </button>
              ))}
            </div>
          </div>

          {/* Epochs */}
          <div>
            <label className="label-base">Epochs</label>
            <div className="grid grid-cols-3 gap-1">
              {EPOCH_MARKS.map((e) => (
                <button
                  key={e}
                  onClick={() => setTrainingConfig({ epochs: e })}
                  className={`py-1 rounded text-xs font-medium transition-all ${
                    trainingConfig.epochs === e
                      ? 'bg-green-600 text-white'
                      : 'bg-gray-800 text-gray-400 hover:bg-gray-700'
                  }`}
                >
                  {e}
                </button>
              ))}
            </div>
          </div>

          {/* Regularization */}
          <div>
            <label className="label-base">Regularization</label>
            <select
              className="select-base text-xs py-1.5"
              value={trainingConfig.reg_type}
              onChange={(e) => setTrainingConfig({ reg_type: e.target.value as RegType })}
            >
              {['None', 'L1', 'L2', 'L1L2'].map((r) => (
                <option key={r} value={r}>{r}</option>
              ))}
            </select>
          </div>
        </div>
      )}
    </div>
  );
}
