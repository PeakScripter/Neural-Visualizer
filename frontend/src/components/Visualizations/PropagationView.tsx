import { useState, useCallback } from 'react';
import { ChevronLeft, ChevronRight, Play, Pause, RotateCcw } from 'lucide-react';
import { useNetworkStore } from '../../store/networkStore';
import { NetworkGraph } from './NetworkGraph';
import type { PropStep } from '../../types';

interface NetworkViewProps {
  activeNodeIds?: Set<number>;
  activeEdgeIds?: Set<number>;
  mode?: 'architecture' | 'forward' | 'backward';
}

interface Props {
  steps: PropStep[];
  mode: 'forward' | 'backward';
  NetworkViewComponent?: (props: NetworkViewProps) => React.ReactElement | null;
}

const STEP_COLORS: Record<string, string> = {
  input: '#3b82f6',
  hidden: '#10b981',
  output: '#ef4444',
  conv: '#8b5cf6',
  fc: '#6366f1',
  rnn: '#f59e0b',
  lstm: '#ec4899',
  generator: '#14b8a6',
  discriminator: '#f97316',
  attention: '#fb923c',
  embedding: '#818cf8',
  feedforward: '#34d399',
  encoder: '#2dd4bf',
  decoder: '#fb7185',
  bottleneck: '#c084fc',
};

export function PropagationView({ steps, mode, NetworkViewComponent }: Props) {
  const { graph, propStep, setPropStep } = useNetworkStore();
  const [isPlaying, setIsPlaying] = useState(false);
  const [intervalId, setIntervalId] = useState<ReturnType<typeof setInterval> | null>(null);

  const currentStep = steps[propStep];
  const activeNodeIds = currentStep ? new Set(currentStep.active_nodes) : undefined;
  const activeEdgeIds = currentStep ? new Set(currentStep.active_edges) : undefined;

  const play = useCallback(() => {
    setIsPlaying(true);
    let current = propStep;
    const id = setInterval(() => {
      current += 1;
      if (current >= steps.length) {
        clearInterval(id);
        setIsPlaying(false);
        setPropStep(0);
      } else {
        setPropStep(current);
      }
    }, 800);
    setIntervalId(id);
  }, [steps.length, setPropStep, propStep]);

  const pause = useCallback(() => {
    if (intervalId) clearInterval(intervalId);
    setIsPlaying(false);
  }, [intervalId]);

  const reset = useCallback(() => {
    if (intervalId) clearInterval(intervalId);
    setIsPlaying(false);
    setPropStep(0);
  }, [intervalId, setPropStep]);

  if (!steps.length) {
    return (
      <div className="flex items-center justify-center h-full text-gray-500 text-sm">
        Build the network first to see propagation.
      </div>
    );
  }

  return (
    <div className="flex flex-col h-full gap-3">
      {/* Step info bar */}
      <div className={`flex items-center gap-3 p-3 rounded-lg bg-gray-800/60 border border-gray-700`}>
        <div
          className="w-3 h-3 rounded-full flex-shrink-0"
          style={{ backgroundColor: STEP_COLORS[currentStep?.layer_type ?? 'hidden'] ?? '#6b7280' }}
        />
        <div className="flex-1 min-w-0">
          <p className="text-sm font-medium text-gray-200 truncate">{currentStep?.label}</p>
          <p className="text-xs text-gray-500">
            Step {propStep + 1} of {steps.length} ·{' '}
            {currentStep?.active_nodes.length} active nodes ·{' '}
            {currentStep?.active_edges.length} active connections
          </p>
        </div>
        <div className={`badge ${mode === 'forward' ? 'badge-blue' : 'badge-orange'}`}>
          {mode === 'forward' ? '→ Forward' : '← Backward'}
        </div>
      </div>

      {/* Graph */}
      <div className="flex-1 min-h-0 rounded-xl overflow-hidden border" style={{ background: 'var(--bg-card)', borderColor: 'var(--border)' }}>
        {NetworkViewComponent
          ? <NetworkViewComponent activeNodeIds={activeNodeIds} activeEdgeIds={activeEdgeIds} mode={mode} />
          : <NetworkGraph graph={graph} activeNodeIds={activeNodeIds} activeEdgeIds={activeEdgeIds} gradients={currentStep?.gradients} mode={mode} />
        }
      </div>

      {/* Step slider */}
      <div className="space-y-2">
        <input
          type="range"
          min={0} max={steps.length - 1} step={1}
          value={propStep}
          onChange={(e) => { pause(); setPropStep(+e.target.value); }}
          className="w-full"
        />
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-1">
            <button onClick={reset} className="btn-secondary p-1.5" title="Reset">
              <RotateCcw size={14} />
            </button>
            <button
              onClick={() => { pause(); setPropStep(Math.max(0, propStep - 1)); }}
              className="btn-secondary p-1.5"
              disabled={propStep === 0}
            >
              <ChevronLeft size={14} />
            </button>
            <button onClick={isPlaying ? pause : play} className="btn-primary px-4 py-1.5 flex items-center gap-1.5 text-sm">
              {isPlaying ? <Pause size={13} /> : <Play size={13} />}
              {isPlaying ? 'Pause' : 'Play'}
            </button>
            <button
              onClick={() => { pause(); setPropStep(Math.min(steps.length - 1, propStep + 1)); }}
              className="btn-secondary p-1.5"
              disabled={propStep === steps.length - 1}
            >
              <ChevronRight size={14} />
            </button>
          </div>
          <div className="text-xs text-gray-500">
            {propStep + 1} / {steps.length}
          </div>
        </div>
      </div>

      {/* Step list mini-nav */}
      <div className="flex gap-1.5 overflow-x-auto pb-1">
        {steps.map((_s, i) => (
          <button
            key={i}
            onClick={() => { pause(); setPropStep(i); }}
            className={`flex-shrink-0 h-1.5 rounded-full transition-all duration-200 ${
              i === propStep
                ? `w-6 ${mode === 'forward' ? 'bg-blue-500' : 'bg-orange-500'}`
                : i < propStep
                ? 'w-3 bg-gray-600'
                : 'w-3 bg-gray-800'
            }`}
          />
        ))}
      </div>
    </div>
  );
}
