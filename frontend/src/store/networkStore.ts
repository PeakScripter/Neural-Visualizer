import { create } from 'zustand';
import type {
  NetworkConfig,
  TrainingConfig,
  NetworkGraph,
  PropStep,
  TrainingResult,
  DecisionBoundaryData,
  LossLandscapeData,
  TabId,
  ModelType,
} from '../types';
import type { ParsedDataset } from '../components/DatasetUpload';

interface NetworkState {
  // Config
  networkConfig: NetworkConfig;
  trainingConfig: TrainingConfig;

  // Data
  graph: NetworkGraph;
  forwardSteps: PropStep[];
  backwardSteps: PropStep[];
  trainingResult: TrainingResult | null;
  decisionBoundary: DecisionBoundaryData | null;
  lossLandscape: LossLandscapeData | null;
  customDataset: ParsedDataset | null;

  // UI
  activeTab: TabId;
  isLoading: boolean;
  error: string | null;
  propStep: number;
  networkBuilt: boolean;
  view3D: boolean;

  // Actions
  setNetworkConfig: (cfg: Partial<NetworkConfig>) => void;
  setTrainingConfig: (cfg: Partial<TrainingConfig>) => void;
  setGraph: (g: NetworkGraph) => void;
  setForwardSteps: (s: PropStep[]) => void;
  setBackwardSteps: (s: PropStep[]) => void;
  setTrainingResult: (r: TrainingResult) => void;
  setDecisionBoundary: (d: DecisionBoundaryData) => void;
  setLossLandscape: (l: LossLandscapeData) => void;
  setActiveTab: (t: TabId) => void;
  setLoading: (v: boolean) => void;
  setError: (e: string | null) => void;
  setPropStep: (s: number) => void;
  setNetworkBuilt: (v: boolean) => void;
  setCustomDataset: (d: ParsedDataset | null) => void;
  setView3D: (v: boolean) => void;
}

const MODEL_LOSSES: Record<ModelType, string[]> = {
  ANN: ['Binary Cross Entropy', 'Mean Squared Error', 'Hinge Loss'],
  CNN: ['Binary Cross Entropy', 'Categorical Cross Entropy', 'Mean Squared Error'],
  RNN: ['Binary Cross Entropy', 'Mean Squared Error', 'Sequence Loss'],
  LSTM: ['Binary Cross Entropy', 'Mean Squared Error', 'Sequence Loss'],
  GAN: ['Binary Cross Entropy', 'Wasserstein Loss', 'Least Squares'],
  Transformer: ['Binary Cross Entropy', 'Cross Entropy', 'Masked Language Model Loss'],
  Diffuser: ['Mean Squared Error', 'L1 Loss', 'Huber Loss'],
};

export { MODEL_LOSSES };

export const useNetworkStore = create<NetworkState>((set) => ({
  networkConfig: {
    model_type: 'ANN',
    n_layers: 3,
    neurons: [32, 32, 32, 32, 32],
    activations: ['ReLU', 'ReLU', 'ReLU', 'ReLU', 'ReLU'],
    loss_fn: 'Binary Cross Entropy',
    reg_type: 'None',
    reg_rate: 0.01,
    input_nodes: 2,
    output_nodes: 1,
  },
  trainingConfig: {
    problem_type: 'Classification',
    dataset: 'Circle',
    noise: 10,
    batch_size: 32,
    learning_rate: 0.01,
    reg_type: 'None',
    reg_rate: 0.01,
    epochs: 5,
    loss_fn: 'Binary Cross Entropy',
  },
  graph: { nodes: [], edges: [] },
  forwardSteps: [],
  backwardSteps: [],
  trainingResult: null,
  decisionBoundary: null,
  lossLandscape: null,
  customDataset: null,
  activeTab: 'architecture',
  isLoading: false,
  error: null,
  propStep: 0,
  networkBuilt: false,
  view3D: false,

  setNetworkConfig: (cfg) =>
    set((s) => ({ networkConfig: { ...s.networkConfig, ...cfg } })),
  setTrainingConfig: (cfg) =>
    set((s) => ({ trainingConfig: { ...s.trainingConfig, ...cfg } })),
  setGraph: (g) => set({ graph: g }),
  setForwardSteps: (steps) => set({ forwardSteps: steps }),
  setBackwardSteps: (steps) => set({ backwardSteps: steps }),
  setTrainingResult: (r) => set({ trainingResult: r }),
  setDecisionBoundary: (d) => set({ decisionBoundary: d }),
  setLossLandscape: (l) => set({ lossLandscape: l }),
  setActiveTab: (t) => set({ activeTab: t }),
  setLoading: (v) => set({ isLoading: v }),
  setError: (e) => set({ error: e }),
  setPropStep: (s) => set({ propStep: s }),
  setNetworkBuilt: (v) => set({ networkBuilt: v }),
  setCustomDataset: (d) => set({ customDataset: d }),
  setView3D: (v) => set({ view3D: v }),
}));
