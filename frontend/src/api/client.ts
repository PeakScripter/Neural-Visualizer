import axios from 'axios';
import type {
  NetworkConfig,
  TrainingConfig,
  NetworkGraph,
  PropStep,
  TrainingResult,
  DecisionBoundaryData,
  LossLandscapeData,
} from '../types';

const BASE = import.meta.env.VITE_API_BASE_URL || 'http://localhost:8000';
const api = axios.create({ baseURL: BASE, timeout: 30000 });

export const buildNetwork = (config: NetworkConfig): Promise<NetworkGraph> =>
  api.post('/api/build-network', config).then(r => r.data);

export const getForwardProp = (config: NetworkConfig): Promise<{ steps: PropStep[]; graph: NetworkGraph }> =>
  api.post('/api/forward-propagation', config).then(r => r.data);

export const getBackwardProp = (config: NetworkConfig): Promise<{ steps: PropStep[]; graph: NetworkGraph }> =>
  api.post('/api/backward-propagation', config).then(r => r.data);

export const getDecisionBoundary = (
  network: NetworkConfig,
  training: TrainingConfig
): Promise<DecisionBoundaryData> =>
  api.post('/api/decision-boundary', { network, training }).then(r => r.data);

export const getLossLandscape = (config: NetworkConfig): Promise<LossLandscapeData> =>
  api.post('/api/loss-landscape', config).then(r => r.data);

export const simulateTraining = (
  network: NetworkConfig,
  training: TrainingConfig
): Promise<TrainingResult> =>
  api.post('/api/simulate-training', { network, training }).then(r => r.data);

export const getDatasetPreview = (training: TrainingConfig): Promise<{ X: number[][]; y: number[] }> =>
  api.post('/api/dataset', training).then(r => r.data);
