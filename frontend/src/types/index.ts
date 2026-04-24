export type ModelType = 'ANN' | 'CNN' | 'RNN' | 'LSTM' | 'GAN' | 'Transformer' | 'Diffuser';
export type ActivationType = 'ReLU' | 'Sigmoid' | 'Tanh' | 'LeakyReLU' | 'ELU' | 'SELU';
export type DatasetType = 'Circle' | 'Gaussian' | 'XOR' | 'Spiral';
export type ProblemType = 'Classification' | 'Regression';
export type RegType = 'None' | 'L1' | 'L2' | 'L1L2' | 'Gradient Penalty' | 'Spectral Norm' | 'Gradient Clipping';

export interface NetworkNode {
  id: number;
  x: number;
  y: number;
  name: string;
  layer: number;
  layer_type: string;
  value: number;
  z_val?: number;
  activation?: string | null;
  weights?: number[];
  bias?: number;
  time_step?: number;
}

export interface NetworkEdge {
  source: number;
  target: number;
  weight: number;
  layer: number;
  type?: string;
}

export interface NetworkGraph {
  nodes: NetworkNode[];
  edges: NetworkEdge[];
}

export interface NetworkConfig {
  model_type: ModelType;
  n_layers: number;
  neurons: number[];
  activations: ActivationType[];
  loss_fn: string;
  reg_type: RegType;
  reg_rate: number;
  input_nodes: number;
  output_nodes: number;
}

export interface TrainingConfig {
  problem_type: ProblemType;
  dataset: DatasetType;
  noise: number;
  batch_size: number;
  learning_rate: number;
  reg_type: RegType;
  reg_rate: number;
  epochs: number;
  loss_fn: string;
}

export interface PropStep {
  step: number;
  label: string;
  active_nodes: number[];
  active_edges: number[];
  layer_type: string;
  gradients?: Record<string, number>;
}

export interface TrainingResult {
  loss_history: number[];
  accuracy_history: number[];
  epochs: number[];
}

export interface DecisionBoundaryData {
  xx: number[][];
  yy: number[][];
  zz: number[][];
  X: number[][];
  y: number[];
}

export interface LossLandscapeData {
  w1: number[][];
  w2: number[][];
  loss: number[][];
}

export type TabId = 'architecture' | 'forward' | 'backward' | 'decision' | 'loss' | 'training' | 'live-train' | 'compare' | 'attention' | 'weights' | 'layer-act' | 'sweep' | 'pruning' | 'custom-act' | 'export';
