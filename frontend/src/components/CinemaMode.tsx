import { useState, useEffect, useCallback, useRef } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { X, ChevronLeft, ChevronRight, Play, Pause, Film } from 'lucide-react';
import { useNetworkStore } from '../store/networkStore';
import { NetworkGraph } from './Visualizations/NetworkGraph';

const LAYER_NARRATIVES: Record<string, { title: string; body: string; color: string }> = {
  input:         { title: 'Input Layer',         color: '#3b82f6', body: 'Raw features enter the network. Each neuron holds one input dimension.' },
  hidden:        { title: 'Hidden Layer',        color: '#10b981', body: 'Neurons apply a weighted sum + activation, learning abstract representations.' },
  output:        { title: 'Output Layer',        color: '#ef4444', body: 'Final prediction: probability for classification, value for regression.' },
  conv:          { title: 'Conv Layer',          color: '#8b5cf6', body: 'Sliding filters detect local patterns — edges, textures — with shared weights.' },
  rnn:           { title: 'Recurrent Layer',     color: '#f59e0b', body: 'Hidden state loops back each time step, giving the network memory.' },
  lstm:          { title: 'LSTM Layer',          color: '#ec4899', body: 'Input, forget, output gates precisely control what information persists.' },
  attention:     { title: 'Attention Head',      color: '#fb923c', body: 'Each token attends to every other, learning which context matters most.' },
  embedding:     { title: 'Embedding Layer',     color: '#818cf8', body: 'Discrete tokens lifted into continuous space — similar tokens cluster.' },
  feedforward:   { title: 'Feed-Forward Block',  color: '#34d399', body: 'Two-layer MLP after attention, processing each position independently.' },
  encoder:       { title: 'Encoder',             color: '#2dd4bf', body: 'Compresses input into a compact latent code, discarding noise.' },
  decoder:       { title: 'Decoder',             color: '#fb7185', body: 'Reconstructs output from the latent code.' },
  bottleneck:    { title: 'Bottleneck',          color: '#c084fc', body: 'The tightest point — only essential features survive compression.' },
  generator:     { title: 'Generator',           color: '#14b8a6', body: 'Maps noise to synthetic samples that fool the discriminator.' },
  discriminator: { title: 'Discriminator',       color: '#f97316', body: 'Classifies real vs generated samples, training the generator.' },
};

interface Props {
  onClose: () => void;
}

export function CinemaMode({ onClose }: Props) {
  const { graph, forwardSteps, setPropStep, networkConfig } = useNetworkStore();
  const [stepIdx, setStepIdx] = useState(0);
  const [playing, setPlaying] = useState(true);
  const intervalRef = useRef<ReturnType<typeof setInterval> | null>(null);

  const totalSteps = forwardSteps.length;
  const currentStep = forwardSteps[stepIdx] ?? null;
  const layerType = currentStep?.layer_type ?? graph.nodes[0]?.layer_type ?? 'hidden';
  const nar = LAYER_NARRATIVES[layerType] ?? { title: layerType, color: '#6b7280', body: '' };

  const activeNodeIds = currentStep ? new Set(currentStep.active_nodes) : undefined;
  const activeEdgeIds = currentStep ? new Set(currentStep.active_edges) : undefined;

  const advance = useCallback(() => {
    setStepIdx((i) => {
      if (i >= totalSteps - 1) { setPlaying(false); return i; }
      return i + 1;
    });
  }, [totalSteps]);

  useEffect(() => {
    if (!playing) { if (intervalRef.current) clearInterval(intervalRef.current); return; }
    intervalRef.current = setInterval(advance, 1800);
    return () => { if (intervalRef.current) clearInterval(intervalRef.current); };
  }, [playing, advance]);

  useEffect(() => { setPropStep(stepIdx); }, [stepIdx, setPropStep]);

  // Keyboard nav
  useEffect(() => {
    const handler = (e: KeyboardEvent) => {
      if (e.key === 'Escape') onClose();
      if (e.key === 'ArrowRight') { setPlaying(false); setStepIdx((i) => Math.min(totalSteps - 1, i + 1)); }
      if (e.key === 'ArrowLeft')  { setPlaying(false); setStepIdx((i) => Math.max(0, i - 1)); }
      if (e.key === ' ') { e.preventDefault(); setPlaying((p) => !p); }
    };
    window.addEventListener('keydown', handler);
    return () => window.removeEventListener('keydown', handler);
  }, [onClose, totalSteps]);

  return (
    <div className="cinema-overlay flex flex-col" onClick={onClose}>

      {/* ── Top bar ── */}
      <div
        className="flex items-center justify-between px-6 py-3 flex-shrink-0"
        style={{ borderBottom: '1px solid rgba(255,255,255,0.06)' }}
        onClick={(e) => e.stopPropagation()}
      >
        <div className="flex items-center gap-2 text-sm" style={{ color: nar.color }}>
          <Film size={14} />
          <span className="font-semibold tracking-widest uppercase text-xs">Cinema Mode</span>
          <span className="ml-3 text-xs" style={{ color: 'rgba(156,163,175,0.6)' }}>
            {networkConfig.model_type} · {graph.nodes.length} nodes · {graph.edges.length} edges
          </span>
        </div>
        <div className="flex items-center gap-3 text-xs" style={{ color: 'rgba(156,163,175,0.5)' }}>
          <span>← → navigate · Space play/pause · Esc close</span>
          <button
            onClick={onClose}
            className="p-1.5 rounded-lg transition-opacity hover:opacity-100 opacity-50"
            style={{ background: 'rgba(255,255,255,0.08)' }}
          >
            <X size={15} color="white" />
          </button>
        </div>
      </div>

      {/* ── Network — takes up all remaining space ── */}
      <div className="flex-1 min-h-0 relative" onClick={(e) => e.stopPropagation()}>
        <NetworkGraph
          graph={graph}
          activeNodeIds={activeNodeIds}
          activeEdgeIds={activeEdgeIds}
          mode="forward"
        />

        {/* Colour-coded step label floating top-left of network */}
        <AnimatePresence mode="wait">
          <motion.div
            key={layerType}
            initial={{ opacity: 0, x: -10 }}
            animate={{ opacity: 1, x: 0 }}
            exit={{ opacity: 0 }}
            transition={{ duration: 0.25 }}
            className="absolute top-4 left-4 flex items-center gap-2 px-3 py-1.5 rounded-full text-xs font-bold uppercase tracking-wider"
            style={{
              background: nar.color + '22',
              border: `1px solid ${nar.color}55`,
              color: nar.color,
            }}
          >
            <span
              className="w-2 h-2 rounded-full"
              style={{ background: nar.color, boxShadow: `0 0 6px ${nar.color}` }}
            />
            {nar.title}
          </motion.div>
        </AnimatePresence>

        {/* Active node / edge count */}
        {currentStep && (
          <div
            className="absolute top-4 right-4 flex gap-3 text-xs"
            style={{ color: 'rgba(156,163,175,0.7)' }}
          >
            <span><b style={{ color: nar.color }}>{currentStep.active_nodes.length}</b> active neurons</span>
            <span><b style={{ color: nar.color }}>{currentStep.active_edges.length}</b> connections</span>
          </div>
        )}
      </div>

      {/* ── Bottom narration + controls ── */}
      <div
        className="flex-shrink-0 px-6 py-4"
        style={{ borderTop: `1px solid ${nar.color}22`, background: 'rgba(0,0,0,0.6)', backdropFilter: 'blur(12px)' }}
        onClick={(e) => e.stopPropagation()}
      >
        {/* Progress bar */}
        <div className="w-full h-0.5 rounded-full mb-3 overflow-hidden" style={{ background: 'rgba(255,255,255,0.08)' }}>
          <motion.div
            className="h-full rounded-full"
            style={{ background: nar.color }}
            animate={{ width: `${totalSteps ? ((stepIdx + 1) / totalSteps) * 100 : 0}%` }}
            transition={{ duration: 0.3 }}
          />
        </div>

        <div className="flex items-center gap-6">
          {/* Narration text */}
          <div className="flex-1 min-w-0">
            <AnimatePresence mode="wait">
              <motion.p
                key={nar.body}
                initial={{ opacity: 0, y: 4 }}
                animate={{ opacity: 1, y: 0 }}
                exit={{ opacity: 0, y: -4 }}
                transition={{ duration: 0.2 }}
                className="text-sm leading-relaxed"
                style={{ color: 'rgba(226,232,240,0.8)' }}
              >
                {nar.body}
              </motion.p>
            </AnimatePresence>
          </div>

          {/* Controls */}
          <div className="flex items-center gap-2 flex-shrink-0">
            <button
              disabled={stepIdx === 0}
              onClick={() => { setPlaying(false); setStepIdx((i) => Math.max(0, i - 1)); }}
              className="p-2 rounded-lg disabled:opacity-30 transition-opacity"
              style={{ background: 'rgba(255,255,255,0.08)' }}
            >
              <ChevronLeft size={16} color="white" />
            </button>

            <button
              onClick={() => setPlaying((p) => !p)}
              className="flex items-center gap-2 px-5 py-2 rounded-lg font-medium text-sm transition-all"
              style={{ background: nar.color, color: '#000' }}
            >
              {playing ? <Pause size={14} /> : <Play size={14} />}
              {playing ? 'Pause' : 'Play'}
            </button>

            <button
              disabled={stepIdx >= totalSteps - 1}
              onClick={() => { setPlaying(false); setStepIdx((i) => Math.min(totalSteps - 1, i + 1)); }}
              className="p-2 rounded-lg disabled:opacity-30 transition-opacity"
              style={{ background: 'rgba(255,255,255,0.08)' }}
            >
              <ChevronRight size={16} color="white" />
            </button>

            <span className="text-xs ml-2" style={{ color: 'rgba(156,163,175,0.5)' }}>
              {stepIdx + 1} / {totalSteps}
            </span>
          </div>
        </div>
      </div>
    </div>
  );
}
