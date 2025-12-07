import { useState, useEffect } from 'react';
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, PieChart, Pie, Cell, ScatterChart, Scatter, Legend } from 'recharts';
import { Loader2, TrendingUp, Package, Cpu, Monitor } from 'lucide-react';
import { getStatistics, getClusters, getStatsByCategory } from '../services/api';
import type { MarketStatistics, ClusterData } from '../types';

const COLORS = ['#4F8BF9', '#10B981', '#F59E0B', '#EF4444', '#8B5CF6', '#EC4899'];

export default function MarketOverview() {
  const [activeTab, setActiveTab] = useState<'statistics' | 'segmentation'>('statistics');
  const [stats, setStats] = useState<MarketStatistics | null>(null);
  const [clusters, setClusters] = useState<ClusterData | null>(null);
  const [categoryData, setCategoryData] = useState<any[]>([]);
  const [selectedCategory, setSelectedCategory] = useState<'brand' | 'cpu' | 'gpu'>('brand');
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    loadData();
  }, []);

  useEffect(() => {
    if (activeTab === 'statistics') {
      loadCategoryData();
    }
  }, [selectedCategory, activeTab]);

  const loadData = async () => {
    try {
      setLoading(true);
      const [statsData, clusterData] = await Promise.all([
        getStatistics(),
        getClusters(),
      ]);
      setStats(statsData);
      setClusters(clusterData);
    } catch (err) {
      setError('Failed to load data');
      console.error(err);
    } finally {
      setLoading(false);
    }
  };

  const loadCategoryData = async () => {
    try {
      const data = await getStatsByCategory(selectedCategory);
      setCategoryData(data.slice(0, 10));
    } catch (err) {
      console.error(err);
    }
  };

  if (loading) {
    return (
      <div className="flex items-center justify-center h-96">
        <Loader2 className="h-8 w-8 animate-spin text-primary-500" />
      </div>
    );
  }

  if (error) {
    return (
      <div className="text-center py-12">
        <p className="text-red-500">{error}</p>
        <button onClick={loadData} className="btn-primary mt-4">
          Retry
        </button>
      </div>
    );
  }

  const brandData = stats?.brand_distribution
    ? Object.entries(stats.brand_distribution).map(([name, value]) => ({ name, value }))
    : [];

  const cpuData = stats?.cpu_distribution
    ? Object.entries(stats.cpu_distribution).map(([name, value]) => ({ name, value }))
    : [];

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-3xl font-bold text-gray-900">Market Overview</h1>
        <p className="text-gray-600 mt-1">
          Explore laptop market statistics and segmentation
        </p>
      </div>

      {/* Tabs */}
      <div className="flex gap-2 border-b border-gray-200">
        <button
          onClick={() => setActiveTab('statistics')}
          className={`px-4 py-2 font-medium border-b-2 transition-colors ${
            activeTab === 'statistics'
              ? 'border-primary-500 text-primary-600'
              : 'border-transparent text-gray-500 hover:text-gray-700'
          }`}
        >
          <TrendingUp className="h-4 w-4 inline mr-2" />
          Statistics
        </button>
        <button
          onClick={() => setActiveTab('segmentation')}
          className={`px-4 py-2 font-medium border-b-2 transition-colors ${
            activeTab === 'segmentation'
              ? 'border-primary-500 text-primary-600'
              : 'border-transparent text-gray-500 hover:text-gray-700'
          }`}
        >
          <Package className="h-4 w-4 inline mr-2" />
          Segmentation
        </button>
      </div>

      {activeTab === 'statistics' && stats && (
        <div className="space-y-6">
          {/* Key Metrics */}
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
            <div className="card">
              <div className="text-sm text-gray-500">Total Listings</div>
              <div className="text-2xl font-bold text-gray-900">{stats.total_listings.toLocaleString()}</div>
            </div>
            <div className="card">
              <div className="text-sm text-gray-500">Average Price</div>
              <div className="text-2xl font-bold text-gray-900">€{stats.price_stats.mean.toFixed(0)}</div>
            </div>
            <div className="card">
              <div className="text-sm text-gray-500">Median Price</div>
              <div className="text-2xl font-bold text-gray-900">€{stats.price_stats.median.toFixed(0)}</div>
            </div>
            <div className="card">
              <div className="text-sm text-gray-500">Price Range</div>
              <div className="text-2xl font-bold text-gray-900">
                €{stats.price_stats.min.toFixed(0)} - €{stats.price_stats.max.toFixed(0)}
              </div>
            </div>
          </div>

          {/* Charts Row */}
          <div className="grid md:grid-cols-2 gap-6">
            {/* Brand Distribution */}
            <div className="card">
              <h3 className="font-semibold text-gray-900 mb-4">Top Brands</h3>
              <ResponsiveContainer width="100%" height={300}>
                <BarChart data={brandData} layout="vertical">
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis type="number" />
                  <YAxis dataKey="name" type="category" width={80} />
                  <Tooltip />
                  <Bar dataKey="value" fill="#4F8BF9" />
                </BarChart>
              </ResponsiveContainer>
            </div>

            {/* CPU Distribution */}
            <div className="card">
              <h3 className="font-semibold text-gray-900 mb-4">CPU Distribution</h3>
              <ResponsiveContainer width="100%" height={300}>
                <PieChart>
                  <Pie
                    data={cpuData}
                    cx="50%"
                    cy="50%"
                    labelLine={false}
                    label={({ name, percent }) => `${name} ${(percent * 100).toFixed(0)}%`}
                    outerRadius={100}
                    fill="#8884d8"
                    dataKey="value"
                  >
                    {cpuData.map((_, index) => (
                      <Cell key={`cell-${index}`} fill={COLORS[index % COLORS.length]} />
                    ))}
                  </Pie>
                  <Tooltip />
                </PieChart>
              </ResponsiveContainer>
            </div>
          </div>

          {/* Price by Category */}
          <div className="card">
            <div className="flex items-center justify-between mb-4">
              <h3 className="font-semibold text-gray-900">Price by Category</h3>
              <select
                value={selectedCategory}
                onChange={(e) => setSelectedCategory(e.target.value as any)}
                className="input w-40"
              >
                <option value="brand">By Brand</option>
                <option value="cpu">By CPU</option>
                <option value="gpu">By GPU</option>
              </select>
            </div>
            <ResponsiveContainer width="100%" height={350}>
              <BarChart data={categoryData}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey={selectedCategory === 'brand' ? '_brand' : selectedCategory === 'cpu' ? '_cpu_brand' : '_gpu_brand'} />
                <YAxis />
                <Tooltip formatter={(value: number) => `€${value.toFixed(0)}`} />
                <Legend />
                <Bar dataKey="Mean Price" fill="#4F8BF9" name="Mean Price" />
                <Bar dataKey="Median Price" fill="#10B981" name="Median Price" />
              </BarChart>
            </ResponsiveContainer>
          </div>
        </div>
      )}

      {activeTab === 'segmentation' && clusters && (
        <div className="space-y-6">
          {/* Segment Cards */}
          <div className="grid md:grid-cols-5 gap-4">
            {clusters.profiles.map((profile, index) => (
              <div
                key={profile.cluster_id}
                className="card text-center"
                style={{ borderTop: `4px solid ${COLORS[index % COLORS.length]}` }}
              >
                <div className="font-semibold text-gray-900">{profile.segment_name}</div>
                <div className="text-2xl font-bold text-primary-600 my-2">
                  €{profile.avg_price.toFixed(0)}
                </div>
                <div className="text-sm text-gray-500">
                  {profile.count.toLocaleString()} laptops ({profile.percentage.toFixed(1)}%)
                </div>
                <div className="text-xs text-gray-400 mt-2">{profile.description}</div>
              </div>
            ))}
          </div>

          {/* Cluster Visualization */}
          <div className="card">
            <h3 className="font-semibold text-gray-900 mb-4">Cluster Visualization (PCA)</h3>
            <ResponsiveContainer width="100%" height={400}>
              <ScatterChart>
                <CartesianGrid />
                <XAxis type="number" dataKey="x" name="PC1" />
                <YAxis type="number" dataKey="y" name="PC2" />
                <Tooltip cursor={{ strokeDasharray: '3 3' }} />
                <Legend />
                {clusters.profiles.map((profile, index) => (
                  <Scatter
                    key={profile.cluster_id}
                    name={profile.segment_name}
                    data={clusters.pca_data.filter((d) => d.cluster === profile.cluster_id)}
                    fill={COLORS[index % COLORS.length]}
                    opacity={0.6}
                  />
                ))}
              </ScatterChart>
            </ResponsiveContainer>
          </div>

          {/* Segment Details */}
          <div className="card">
            <h3 className="font-semibold text-gray-900 mb-4">Segment Details</h3>
            <div className="overflow-x-auto">
              <table className="w-full">
                <thead>
                  <tr className="border-b border-gray-200">
                    <th className="text-left py-3 px-4">Segment</th>
                    <th className="text-right py-3 px-4">Avg Price</th>
                    <th className="text-right py-3 px-4">Avg RAM</th>
                    <th className="text-right py-3 px-4">Avg Storage</th>
                    <th className="text-right py-3 px-4">Avg Screen</th>
                    <th className="text-left py-3 px-4">Top Brand</th>
                    <th className="text-right py-3 px-4">Count</th>
                  </tr>
                </thead>
                <tbody>
                  {clusters.profiles.map((profile) => (
                    <tr key={profile.cluster_id} className="border-b border-gray-100">
                      <td className="py-3 px-4 font-medium">{profile.segment_name}</td>
                      <td className="py-3 px-4 text-right">€{profile.avg_price.toFixed(0)}</td>
                      <td className="py-3 px-4 text-right">{profile.avg_ram.toFixed(0)} GB</td>
                      <td className="py-3 px-4 text-right">{profile.avg_storage.toFixed(0)} GB</td>
                      <td className="py-3 px-4 text-right">{profile.avg_screen.toFixed(1)}"</td>
                      <td className="py-3 px-4">{profile.top_brand}</td>
                      <td className="py-3 px-4 text-right">{profile.count.toLocaleString()}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
