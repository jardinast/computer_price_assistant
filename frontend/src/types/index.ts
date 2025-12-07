// API Types

export interface PredictionRequest {
  brand?: string;
  cpu_brand?: string;
  cpu_family: string;
  ram_gb: number;
  ssd_gb: number;
  gpu_brand?: string;
  gpu_series?: string;
  gpu_is_integrated: boolean;
  screen_size?: number;
  refresh_rate?: number;
  weight_kg?: number;
  use_case: string;
}

export interface PredictionResponse {
  predicted_price: number;
  price_min: number;
  price_max: number;
  confidence: 'high' | 'medium' | 'low';
  top_features: FeatureImportance[];
  explanation: string;
}

export interface FeatureImportance {
  feature: string;
  readable_name: string;
  importance: number;
  importance_pct: number;
}

export interface MarketStatistics {
  total_listings: number;
  price_stats: {
    mean: number;
    median: number;
    min: number;
    max: number;
    std: number;
    q25: number;
    q75: number;
  };
  brand_distribution: Record<string, number>;
  cpu_distribution: Record<string, number>;
  gpu_distribution: Record<string, number>;
  ram_distribution: Record<string, number>;
  storage_distribution: Record<string, number>;
  screen_distribution: Record<string, number>;
}

export interface ClusterProfile {
  cluster_id: number;
  count: number;
  percentage: number;
  avg_price: number;
  avg_ram: number;
  avg_storage: number;
  avg_screen: number;
  top_brand: string;
  segment_name: string;
  description: string;
}

export interface ClusterData {
  profiles: ClusterProfile[];
  pca_data: { x: number; y: number; cluster: number }[];
  total_clustered: number;
}

export interface SimilarLaptop {
  rank: number;
  title: string;
  price: number;
  price_min: number;
  price_max: number;
  brand: string;
  ram_gb: number | null;
  ssd_gb: number | null;
  screen_size: number | null;
  cpu_brand: string;
  gpu_brand: string;
  distance: number;
}

export interface ChatMessage {
  role: 'user' | 'assistant';
  content: string;
}
