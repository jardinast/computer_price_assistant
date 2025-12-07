import axios from 'axios';
import type {
  PredictionRequest,
  PredictionResponse,
  MarketStatistics,
  ClusterData,
  SimilarLaptop,
  ChatMessage,
} from '../types';

const API_BASE = '/api';

const api = axios.create({
  baseURL: API_BASE,
  headers: {
    'Content-Type': 'application/json',
  },
});

// Health check
export const checkHealth = async () => {
  const response = await api.get('/health');
  return response.data;
};

// Prediction endpoints
export const predictPrice = async (data: PredictionRequest): Promise<PredictionResponse> => {
  const response = await api.post('/predict', data);
  return response.data;
};

export const getFormOptions = async () => {
  const response = await api.get('/form-options');
  return response.data;
};

// Statistics endpoints
export const getStatistics = async (): Promise<MarketStatistics> => {
  const response = await api.get('/statistics');
  return response.data;
};

export const getStatsByCategory = async (category: 'brand' | 'cpu' | 'gpu') => {
  const response = await api.get(`/statistics/by-category?category=${category}`);
  return response.data;
};

// Clustering endpoints
export const getClusters = async (): Promise<ClusterData> => {
  const response = await api.get('/clusters');
  return response.data;
};

// Similar laptops endpoints
export const findSimilarLaptops = async (params: {
  price_target: number;
  ram_gb: number;
  ssd_gb: number;
  screen_size: number;
  brand?: string;
  cpu_brand?: string;
  num_results?: number;
}): Promise<{ laptops: SimilarLaptop[]; count: number }> => {
  const response = await api.post('/similar', params);
  return response.data;
};

// Chat endpoints
export const sendChatMessage = async (
  message: string,
  apiKey: string,
  history: ChatMessage[]
): Promise<{ response: string; conversation_history: ChatMessage[] }> => {
  const response = await api.post('/chat', {
    message,
    api_key: apiKey,
    conversation_history: history,
  });
  return response.data;
};

export const getChatGreeting = async (): Promise<{ greeting: string }> => {
  const response = await api.get('/chat/greeting');
  return response.data;
};

export default api;
