import { useState, useEffect, useCallback, useRef } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { X, ChevronRight, ChevronLeft, BookOpen, Lightbulb } from 'lucide-react';
import { useNetworkStore } from '../store/networkStore';
import type { TabId } from '../types';

interface Step {
  title: string;
  description: string;
  hint?: string;
  tab?: TabId;
  tourId?: string;          // matches data-tour="..." on the target DOM element
  scrollTarget?: string;    // if different element should be scrolled into view
}

const STEPS: Step[] = [
  {
    title: 'Welcome to Neural Visualizer',
    description:
      'This tool lets you build, configure, train, and explore neural networks interactively. ' +
      'We\'ll walk through every part of the interface — starting with the sidebar controls on the left.',
    hint: 'Press → or click Next to begin. Press Esc to close at any time.',
  },

  // ── Sidebar: Network Config ───────────────────────────────────────────────
  {
    title: 'Network Configuration Panel',
    description:
      'The entire left sidebar is your control centre. ' +
      'The top card — Network Configuration — defines the shape of your neural network. ' +
      'Click any section header to collapse / expand it.',
    tourId: 'network-config-panel',
  },
  {
    title: 'Model Type',
    description:
      'Seven architectures are available. ANN: classic fully-connected layers — the simplest starting point. ' +
      'CNN: convolutional layers for image-like data. RNN / LSTM: sequential / time-series models with recurrent connections. ' +
      'GAN: a generator + discriminator pair trained adversarially. ' +
      'Transformer: multi-head attention encoder. Diffuser: noise-denoising U-Net.',
    hint: 'Each model type changes the architecture, loss options, and code export.',
    tourId: 'model-type',
  },
  {
    title: 'Loss Function',
    description:
      'The loss function measures how wrong the network\'s predictions are. ' +
      'Binary Cross Entropy: for binary classification (output is a probability). ' +
      'Categorical Cross Entropy: multi-class classification. ' +
      'MSE: regression tasks. ' +
      'Wasserstein / Least Squares: for GAN training stability. ' +
      'The list updates automatically when you change model type.',
    tourId: 'loss-fn',
  },
  {
    title: 'Input & Output Nodes',
    description:
      'Input Nodes (1–16) sets your data\'s feature count — use 2 for the built-in 2D datasets (XOR, Circle, Spiral). ' +
      'Increase it if you have higher-dimensional data. ' +
      'Output Nodes (1–10) defines the prediction head — 1 for binary classification, more for multi-class or regression. ' +
      'Both values feed directly into the generated code.',
    tourId: 'input-output',
  },
  {
    title: 'Hidden Layers',
    description:
      'The slider sets network depth (1–5 hidden layers). ' +
      'More layers let the network learn more abstract features but can overfit or suffer vanishing gradients. ' +
      'The budget hint below the slider shows the maximum neurons per layer at the current depth to keep the graph performant.',
    tourId: 'hidden-layers',
  },
  {
    title: 'Layer Configuration',
    description:
      'One card per hidden layer. Neurons: how wide the layer is — more neurons = more capacity to fit complex patterns. ' +
      'Activation: how each neuron fires. ReLU (default) is fast and avoids vanishing gradients. ' +
      'Sigmoid/Tanh output bounded ranges. LeakyReLU avoids "dead neuron" problems. ' +
      'ELU/SELU self-normalise and can converge faster.',
    tourId: 'layer-config',
  },
  {
    title: 'Regularization',
    description:
      'Regularization penalises large weights to reduce overfitting. ' +
      'L1 (Lasso): drives many weights to exactly zero — useful for feature selection. ' +
      'L2 (Ridge): shrinks all weights smoothly — the most common choice. ' +
      'L1L2 (Elastic Net): combines both. ' +
      'A Rate slider appears once you pick a type — start between 0.001 and 0.01.',
    tourId: 'regularization',
  },

  // ── Sidebar: Training Config ──────────────────────────────────────────────
  {
    title: 'Training Parameters Panel',
    description:
      'The second sidebar card controls the training process. ' +
      'Problem Type switches the network\'s objective: Classification (predict a discrete class) or Regression (predict a continuous value). ' +
      'Dataset picks which built-in toy data to train on.',
    tourId: 'training-config-panel',
  },
  {
    title: 'Dataset & Noise',
    description:
      'Four datasets are built in. Circle: two concentric rings — linearly non-separable. ' +
      'Gaussian: two blob clusters — linearly separable. ' +
      'XOR: the classic non-linear problem no single layer can solve. ' +
      'Spiral: interleaved spirals — the hardest, requires deep/wide networks. ' +
      'Noise adds random label flips (0–20%) to simulate messy real-world data.',
    tourId: 'dataset',
  },
  {
    title: 'Learning Rate, Batch Size & Epochs',
    description:
      'Learning Rate: size of each gradient descent step. 1e-2 (0.01) is a safe default; 1e-1 often overshoots; 1e-4 is slow but stable. ' +
      'Batch Size: samples processed per weight update. SGD (1) is noisy but fast-iterating; 32–64 balances speed and stability. ' +
      'Epochs: full passes through the dataset. Increase for harder tasks.',
    tourId: 'lr-batch-epochs',
  },

  // ── Build ─────────────────────────────────────────────────────────────────
  {
    title: 'Build & Train Buttons',
    description:
      '"Build Network" sends your config to the FastAPI backend, which computes the graph, propagation steps, and loss landscape — all tabs update instantly. ' +
      '"Simulate Training" runs the training loop and produces decision boundary and training curve data. ' +
      'Both buttons must succeed before the visualisation tabs are populated.',
    tourId: 'build-btn',
  },

  // ── Tabs ──────────────────────────────────────────────────────────────────
  {
    title: 'Architecture View',
    description:
      'Renders your network as a zoomable 2D graph with colour-coded nodes and weight-scaled edges. ' +
      'Click a node to inspect its type, value, activation, and bias. ' +
      'Toggle to 3D (top-right of the content area) for an interactive WebGL scene — drag to orbit, scroll to zoom.',
    tab: 'architecture',
    tourId: 'tab-architecture',
  },
  {
    title: '3D / 2D Toggle',
    description:
      'When you\'re on the Architecture, Forward, Backprop, or Pruning tabs this button appears top-right. ' +
      '3D mode renders a WebGL starfield with each layer at a different X position. ' +
      'Click any sphere to see the node\'s details in a floating tooltip.',
    tab: 'architecture',
    tourId: 'toggle-3d',
  },
  {
    title: 'Forward & Backpropagation',
    description:
      'Forward: step through data flowing layer by layer. Active nodes glow; signal particles travel along active edges. ' +
      'Backprop: reverse pass — watch gradients flow backward, showing which weights are updated most. ' +
      'Use the step slider to scrub through each propagation stage.',
    tab: 'forward',
    tourId: 'tab-forward',
  },
  {
    title: 'Weights & Layer Activations',
    description:
      'Weights tab: bar histograms of weight distributions per layer — a healthy initialisation is roughly bell-shaped; exploding/vanishing weights signal problems. ' +
      'Activations tab: heatmap of each layer\'s output values — large dark patches indicate dead ReLUs or vanishing activations.',
    tab: 'weights',
    tourId: 'tab-weights',
  },
  {
    title: 'Training Curves & Decision Boundary',
    description:
      'Training tab: loss and accuracy plotted over epochs after "Simulate Training" — look for smooth convergence. ' +
      'Decision tab: a colour grid showing what class the model predicts at every (x, y) point — reveals whether the learned boundary is linear, curved, or complex.',
    tab: 'training',
    tourId: 'tab-training',
  },
  {
    title: 'Live Training (TF.js)',
    description:
      'Runs a real TensorFlow.js model entirely in your browser — no backend call needed. ' +
      'The loss and accuracy curves update live every epoch. ' +
      'LR Sweep trains five models in parallel with different learning rates so you can compare convergence speeds on the same chart.',
    tab: 'live-train',
    tourId: 'tab-live-train',
  },
  {
    title: 'Custom Activation Function',
    description:
      'Draw any curve on the canvas to define your own activation function. ' +
      'Click or drag to place control points; the green line interpolates between them. ' +
      'Compare it against the ReLU reference (blue dashes). ' +
      'Hit "Train with custom act" to run a TF.js model using your hand-drawn function.',
    tab: 'custom-act',
    tourId: 'tab-custom-act',
  },
  {
    title: 'Export & Template Code',
    description:
      'Export generates PyTorch or Keras code that exactly matches your current configuration — ready to copy and run. ' +
      'Switch to the Template tab to write your own code scaffold using {{variables}} like {{n_layers}}, {{input_nodes}}, and {{loss_fn_code}} that auto-fill from your settings in real time.',
    tab: 'export',
    tourId: 'tab-export',
  },
  {
    title: 'You\'re all set!',
    description:
      'Experiment freely — swap model types, adjust depth and width, watch every tab react. ' +
      'Click the Tour button in the header any time to revisit this guide.',
    hint: 'Happy exploring! 🚀',
    tourId: 'tour-btn',
  },
];

interface Props {
  onClose: () => void;
}

export function Tutorial({ onClose }: Props) {
  const [step, setStep] = useState(0);
  const { setActiveTab } = useNetworkStore();
  const cardRef = useRef<HTMLDivElement>(null);
  const [highlightRect, setHighlightRect] = useState<DOMRect | null>(null);

  const isLast = step === STEPS.length - 1;
  const current = STEPS[step];

  // ── Measure the target element ────────────────────────────────────────────
  const measureTarget = useCallback(() => {
    const tid = STEPS[step]?.tourId;
    if (!tid) { setHighlightRect(null); return; }
    const el = document.querySelector<HTMLElement>(`[data-tour="${tid}"]`);
    if (!el) { setHighlightRect(null); return; }
    el.scrollIntoView({ behavior: 'instant', block: 'nearest', inline: 'nearest' });
    setHighlightRect(el.getBoundingClientRect());
  }, [step]);

  useEffect(() => {
    measureTarget();
    window.addEventListener('resize', measureTarget);
    // Re-measure when sidebar scrolls
    document.querySelector('aside')?.addEventListener('scroll', measureTarget);
    return () => {
      window.removeEventListener('resize', measureTarget);
      document.querySelector('aside')?.removeEventListener('scroll', measureTarget);
    };
  }, [measureTarget]);

  // ── Navigation ────────────────────────────────────────────────────────────
  const goTo = useCallback((s: number) => {
    const target = STEPS[s];
    if (target?.tab) setActiveTab(target.tab);
    setStep(s);
  }, [setActiveTab]);

  const goNext = useCallback(() => {
    if (isLast) { onClose(); return; }
    goTo(step + 1);
  }, [isLast, step, goTo, onClose]);

  const goPrev = useCallback(() => {
    if (step > 0) goTo(step - 1);
  }, [step, goTo]);

  useEffect(() => {
    const handler = (e: KeyboardEvent) => {
      if (e.key === 'ArrowRight' || e.key === 'Enter') goNext();
      else if (e.key === 'ArrowLeft') goPrev();
      else if (e.key === 'Escape') onClose();
    };
    window.addEventListener('keydown', handler);
    return () => window.removeEventListener('keydown', handler);
  }, [goNext, goPrev, onClose]);

  // ── Arrow badge direction (relative to card in bottom-right corner) ───────
  const arrowBadge = highlightRect
    ? highlightRect.top < 120
      ? { label: '↑ above', top: highlightRect.bottom + 6, left: highlightRect.left + highlightRect.width / 2 - 36 }
      : { label: '← left',  top: highlightRect.top  + highlightRect.height / 2 - 14, left: highlightRect.right + 6 }
    : null;

  // Clamp badge inside viewport
  if (arrowBadge) {
    arrowBadge.left = Math.max(4, Math.min(window.innerWidth - 100, arrowBadge.left));
    arrowBadge.top  = Math.max(4, Math.min(window.innerHeight - 40, arrowBadge.top));
  }

  return (
    <>
      {/* ── Pulsing highlight box ──────────────────────────────────────────── */}
      <AnimatePresence>
        {highlightRect && (
          <motion.div
            key={`hl-${step}`}
            initial={{ opacity: 0, scale: 0.94 }}
            animate={{
              opacity: 1,
              scale: 1,
              boxShadow: [
                '0 0 0 3px rgba(139,92,246,0.3), 0 0 16px rgba(139,92,246,0.2)',
                '0 0 0 6px rgba(139,92,246,0.5), 0 0 32px rgba(139,92,246,0.4)',
                '0 0 0 3px rgba(139,92,246,0.3), 0 0 16px rgba(139,92,246,0.2)',
              ],
            }}
            exit={{ opacity: 0 }}
            transition={{ duration: 0.25, boxShadow: { duration: 1.6, repeat: Infinity, ease: 'easeInOut' } }}
            className="fixed pointer-events-none z-[99]"
            style={{
              top:    highlightRect.top    - 5,
              left:   highlightRect.left   - 5,
              width:  highlightRect.width  + 10,
              height: highlightRect.height + 10,
              border: '2px solid #8b5cf6',
              borderRadius: 10,
            }}
          />
        )}
      </AnimatePresence>

      {/* ── Arrow badge ───────────────────────────────────────────────────── */}
      <AnimatePresence>
        {arrowBadge && (
          <motion.div
            key={`badge-${step}`}
            initial={{ opacity: 0, scale: 0.8 }}
            animate={{ opacity: 1, scale: 1 }}
            exit={{ opacity: 0, scale: 0.8 }}
            transition={{ duration: 0.2 }}
            className="fixed pointer-events-none z-[99] text-xs font-semibold px-2.5 py-1 rounded-full"
            style={{
              top:  arrowBadge.top,
              left: arrowBadge.left,
              background: 'rgba(139,92,246,0.92)',
              color: '#fff',
              backdropFilter: 'blur(4px)',
              border: '1px solid rgba(255,255,255,0.25)',
              whiteSpace: 'nowrap',
              boxShadow: '0 2px 12px rgba(139,92,246,0.5)',
            }}
          >
            {arrowBadge.label}
          </motion.div>
        )}
      </AnimatePresence>

      {/* ── Floating tutorial card ─────────────────────────────────────────── */}
      <AnimatePresence mode="wait">
        <motion.div
          key={step}
          ref={cardRef}
          initial={{ opacity: 0, y: 18, scale: 0.96 }}
          animate={{ opacity: 1, y: 0,  scale: 1 }}
          exit={{ opacity: 0, y: -12, scale: 0.96 }}
          transition={{ duration: 0.18 }}
          className="fixed bottom-6 right-6 z-[100] w-[400px] rounded-2xl border shadow-2xl"
          style={{
            background: 'var(--bg-card)',
            borderColor: 'var(--accent)',
            boxShadow: '0 0 40px rgba(139,92,246,0.2), 0 20px 60px rgba(0,0,0,0.55)',
          }}
        >
          {/* Header row */}
          <div className="flex items-center justify-between px-4 pt-4 pb-2">
            <div className="flex items-center gap-2">
              <BookOpen size={14} style={{ color: 'var(--accent)' }} />
              <span className="text-xs font-semibold" style={{ color: 'var(--accent)' }}>
                Tour · Step {step + 1} / {STEPS.length}
              </span>
            </div>
            <button
              onClick={onClose}
              className="flex items-center gap-1.5 text-xs px-2.5 py-1 rounded-lg border transition-all hover:opacity-80"
              style={{ borderColor: 'var(--border-soft)', color: 'var(--text-faint)' }}
            >
              <X size={11} /> Skip Tutorial
            </button>
          </div>

          {/* Progress bar */}
          <div className="px-4 mb-3">
            <div className="w-full h-1 rounded-full" style={{ background: 'var(--border)' }}>
              <div
                className="h-1 rounded-full transition-all duration-300"
                style={{ width: `${((step + 1) / STEPS.length) * 100}%`, background: 'var(--accent)' }}
              />
            </div>
          </div>

          {/* Body */}
          <div className="px-4 pb-2 space-y-2">
            <h3 className="text-sm font-bold" style={{ color: 'var(--text-primary)' }}>
              {current.title}
            </h3>
            <p className="text-xs leading-relaxed" style={{ color: 'var(--text-muted)' }}>
              {current.description}
            </p>
            {current.hint && (
              <div
                className="flex items-start gap-2 p-2.5 rounded-lg"
                style={{ background: 'rgba(139,92,246,0.08)', border: '1px solid rgba(139,92,246,0.2)' }}
              >
                <Lightbulb size={12} style={{ color: '#a78bfa', flexShrink: 0, marginTop: 1 }} />
                <span className="text-xs" style={{ color: '#a78bfa' }}>{current.hint}</span>
              </div>
            )}
          </div>

          {/* Navigation */}
          <div
            className="flex items-center justify-between px-4 py-3 mt-1 border-t"
            style={{ borderColor: 'var(--border)' }}
          >
            <button
              onClick={goPrev}
              disabled={step === 0}
              className="flex items-center gap-1.5 text-xs px-3 py-1.5 rounded-lg border transition-all disabled:opacity-30"
              style={{ borderColor: 'var(--border-soft)', color: 'var(--text-muted)' }}
            >
              <ChevronLeft size={13} /> Back
            </button>

            {/* Step dots */}
            <div className="flex gap-1 flex-wrap justify-center" style={{ maxWidth: 200 }}>
              {STEPS.map((_, i) => (
                <button
                  key={i}
                  onClick={() => goTo(i)}
                  className="rounded-full transition-all duration-150"
                  style={{
                    width:  i === step ? 14 : 6,
                    height: 6,
                    background: i === step ? 'var(--accent)' : 'var(--border)',
                  }}
                />
              ))}
            </div>

            <button
              onClick={goNext}
              className="flex items-center gap-1.5 text-xs px-3 py-1.5 rounded-lg font-medium transition-all"
              style={{ background: 'var(--accent)', color: '#fff' }}
            >
              {isLast ? 'Done' : 'Next'}
              {!isLast && <ChevronRight size={13} />}
            </button>
          </div>
        </motion.div>
      </AnimatePresence>
    </>
  );
}
