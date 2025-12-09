import { useState, useEffect } from 'react'
import { motion } from 'framer-motion'
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  PieChart,
  Pie,
  Cell,
  ScatterChart,
  Scatter,
  Legend,
} from 'recharts'
import {
  TrendingUp,
  Package,
  DollarSign,
  HardDrive,
  Monitor,
  Cpu,
  Layers,
  RefreshCw,
  AlertCircle,
} from 'lucide-react'
import axios from 'axios'
import { API_BASE } from '../config'
import FeedbackWidget from '../components/FeedbackWidget'

const COLORS = ['#0ea5e9', '#8b5cf6', '#d946ef', '#06b6d4', '#f59e0b', '#10b981']

// Cluster colors matching CSS badge colors
const LABEL_COLORS = {
  'Ultra Premium': '#f43f5e',  // rose-500
  'High-End': '#f59e0b',       // amber-500
  'Performance': '#06b6d4',    // cyan-500
  'Mid-Range': '#3b82f6',      // blue-500
  'Budget': '#22c55e',         // green-500
}

function StatCard({ title, value, subtitle, icon: Icon, color = 'primary' }) {
  const colorClasses = {
    primary: 'from-primary-500/20 to-primary-600/10 border-primary-500/20 text-primary-400',
    accent: 'from-accent-500/20 to-accent-600/10 border-accent-500/20 text-accent-400',
    emerald: 'from-emerald-500/20 to-emerald-600/10 border-emerald-500/20 text-emerald-400',
    amber: 'from-amber-500/20 to-amber-600/10 border-amber-500/20 text-amber-400',
  }

  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      className="glass-card p-6"
    >
      <div className="flex items-start justify-between">
        <div>
          <p className="text-surface-400 text-sm mb-1">{title}</p>
          <p className="font-display text-3xl font-bold text-white stat-number">{value}</p>
          {subtitle && <p className="text-surface-500 text-sm mt-1">{subtitle}</p>}
        </div>
        <div className={`p-3 rounded-xl bg-gradient-to-br ${colorClasses[color]} border`}>
          <Icon className="w-5 h-5" />
        </div>
      </div>
    </motion.div>
  )
}

function CustomTooltip({ active, payload, label }) {
  if (active && payload && payload.length) {
    return (
      <div className="glass-card p-3 border border-surface-600">
        <p className="text-surface-300 text-sm font-medium">{label}</p>
        {payload.map((entry, index) => (
          <p key={index} className="text-white font-mono" style={{ color: entry.color }}>
            {entry.name}: {typeof entry.value === 'number' ? entry.value.toLocaleString() : entry.value}
          </p>
        ))}
      </div>
    )
  }
  return null
}

export default function Dashboard() {
  const [statistics, setStatistics] = useState(null)
  const [clustering, setClustering] = useState(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState(null)
  const [activeTab, setActiveTab] = useState('overview')

  useEffect(() => {
    fetchData()
  }, [])

  const fetchData = async () => {
    setLoading(true)
    setError(null)
    try {
      const [statsRes, clusterRes] = await Promise.all([
        axios.get(`${API_BASE}/descriptive/statistics`),
        axios.get(`${API_BASE}/descriptive/clustering`),
      ])
      setStatistics(statsRes.data)
      setClustering(clusterRes.data)
    } catch (err) {
      setError(err.message || 'Failed to fetch data')
    } finally {
      setLoading(false)
    }
  }

  if (loading) {
    return (
      <div className="flex items-center justify-center min-h-[60vh]">
        <div className="text-center">
          <div className="spinner mx-auto mb-4" />
          <p className="text-surface-400">Loading market analytics...</p>
        </div>
      </div>
    )
  }

  if (error) {
    return (
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-12">
        <div className="glass-card p-8 text-center">
          <AlertCircle className="w-12 h-12 text-red-400 mx-auto mb-4" />
          <h2 className="text-xl font-semibold text-white mb-2">Failed to Load Data</h2>
          <p className="text-surface-400 mb-4">{error}</p>
          <button onClick={fetchData} className="btn-secondary inline-flex items-center gap-2">
            <RefreshCw className="w-4 h-4" />
            Retry
          </button>
        </div>
      </div>
    )
  }

  const stats = statistics
  const priceStats = stats?.price_statistics || {}

  // Prepare chart data
  const brandData = Object.entries(stats?.brand_distribution || {})
    .filter(([name]) => name.toLowerCase() !== 'unknown')
    .slice(0, 10)
    .map(([name, value]) => ({ name, value }))

  const typeData = Object.entries(stats?.product_type_distribution || {})
    .slice(0, 8)
    .map(([name, value]) => ({ name: name.replace('Portátil ', ''), value }))

  const priceByBrandData = Object.entries(stats?.price_by_brand || {})
    .filter(([_, data]) => data.count > 20)
    .sort((a, b) => b[1].mean - a[1].mean)
    .slice(0, 10)
    .map(([name, data]) => ({
      name,
      avg: Math.round(data.mean),
      min: Math.round(data.min),
      max: Math.round(data.max),
    }))

  const ramData = Object.entries(stats?.ram_distribution || {})
    .filter(([k, _]) => [4, 8, 16, 32, 64].includes(parseInt(k)))
    .map(([name, value]) => ({ name: `${name} GB`, value }))

  // Cluster data
  const clusterStats = clustering?.cluster_stats || []
  const scatterData = clustering?.scatter_data || []

  return (
    <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
      {/* Page Header */}
      <div className="mb-8">
        <h1 className="font-display text-3xl font-bold text-white mb-2">
          Market <span className="gradient-text">Overview</span>
        </h1>
        <p className="text-surface-400">
          Statistical analysis and clustering of {stats?.total_listings?.toLocaleString() || '13,000+'} computer listings
        </p>
      </div>

      {/* Tabs */}
      <div className="flex gap-2 mb-8 p-1 bg-surface-800/50 rounded-xl w-fit">
        <button
          onClick={() => setActiveTab('overview')}
          className={`tab-button ${activeTab === 'overview' ? 'active' : ''}`}
        >
          <Layers className="w-4 h-4 inline mr-2" />
          Statistics
        </button>
        <button
          onClick={() => setActiveTab('clustering')}
          className={`tab-button ${activeTab === 'clustering' ? 'active' : ''}`}
        >
          <TrendingUp className="w-4 h-4 inline mr-2" />
          Segmentation
        </button>
      </div>

      {activeTab === 'overview' && (
        <>
          {/* Key Stats Grid */}
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4 mb-8">
            <StatCard
              title="Total Listings"
              value={stats?.total_listings?.toLocaleString() || '-'}
              subtitle="Computer offerings"
              icon={Package}
              color="primary"
            />
            <StatCard
              title="Average Price"
              value={`€${Math.round(priceStats.mean || 0).toLocaleString()}`}
              subtitle={`Median: €${Math.round(priceStats['50%'] || 0).toLocaleString()}`}
              icon={DollarSign}
              color="emerald"
            />
            <StatCard
              title="Price Range"
              value={`€${Math.round(priceStats.min || 0)} - €${Math.round(priceStats.max || 0)}`}
              subtitle="Min to Max"
              icon={TrendingUp}
              color="accent"
            />
            <StatCard
              title="Brands"
              value={Object.keys(stats?.brand_distribution || {}).length}
              subtitle="Unique manufacturers"
              icon={Cpu}
              color="amber"
            />
          </div>

          {/* Charts Grid */}
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6 mb-8">
            {/* Brand Distribution */}
            <motion.div
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: 0.1 }}
              className="glass-card p-6"
            >
              <h3 className="font-display text-lg font-semibold text-white mb-4">
                Brand Distribution
              </h3>
              <div className="h-72">
                <ResponsiveContainer width="100%" height="100%">
                  <BarChart data={brandData} layout="vertical" margin={{ left: 20 }}>
                    <CartesianGrid strokeDasharray="3 3" stroke="#334155" />
                    <XAxis type="number" stroke="#64748b" />
                    <YAxis dataKey="name" type="category" stroke="#64748b" width={80} interval={0} />
                    <Tooltip content={<CustomTooltip />} />
                    <Bar dataKey="value" fill="#0ea5e9" radius={[0, 4, 4, 0]} />
                  </BarChart>
                </ResponsiveContainer>
              </div>
            </motion.div>

            {/* Product Type Distribution */}
            <motion.div
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: 0.2 }}
              className="glass-card p-6"
            >
              <h3 className="font-display text-lg font-semibold text-white mb-4">
                Product Categories
              </h3>
              <div className="h-72">
                <ResponsiveContainer width="100%" height="100%">
                  <PieChart>
                    <Pie
                      data={typeData}
                      cx="50%"
                      cy="50%"
                      innerRadius={60}
                      outerRadius={100}
                      paddingAngle={2}
                      dataKey="value"
                    >
                      {typeData.map((entry, index) => (
                        <Cell key={`cell-${index}`} fill={COLORS[index % COLORS.length]} />
                      ))}
                    </Pie>
                    <Tooltip content={<CustomTooltip />} />
                    <Legend
                      formatter={(value) => <span className="text-surface-300">{value}</span>}
                    />
                  </PieChart>
                </ResponsiveContainer>
              </div>
            </motion.div>

            {/* Price by Brand */}
            <motion.div
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: 0.3 }}
              className="glass-card p-6"
            >
              <h3 className="font-display text-lg font-semibold text-white mb-4">
                Average Price by Brand
              </h3>
              <div className="h-72">
                <ResponsiveContainer width="100%" height="100%">
                  <BarChart data={priceByBrandData} layout="vertical" margin={{ left: 20 }}>
                    <CartesianGrid strokeDasharray="3 3" stroke="#334155" />
                    <XAxis type="number" stroke="#64748b" tickFormatter={(v) => `€${v}`} />
                    <YAxis dataKey="name" type="category" stroke="#64748b" width={80} interval={0} />
                    <Tooltip content={<CustomTooltip />} />
                    <Bar dataKey="avg" fill="#8b5cf6" radius={[0, 4, 4, 0]} name="Avg Price" />
                  </BarChart>
                </ResponsiveContainer>
              </div>
            </motion.div>

            {/* RAM Distribution */}
            <motion.div
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: 0.4 }}
              className="glass-card p-6"
            >
              <h3 className="font-display text-lg font-semibold text-white mb-4">
                RAM Configuration
              </h3>
              <div className="h-72">
                <ResponsiveContainer width="100%" height="100%">
                  <BarChart data={ramData}>
                    <CartesianGrid strokeDasharray="3 3" stroke="#334155" />
                    <XAxis dataKey="name" stroke="#64748b" />
                    <YAxis stroke="#64748b" />
                    <Tooltip content={<CustomTooltip />} />
                    <Bar dataKey="value" fill="#10b981" radius={[4, 4, 0, 0]} name="Count" />
                  </BarChart>
                </ResponsiveContainer>
              </div>
            </motion.div>
          </div>

          {/* Feedback Widget for Statistics */}
          <div className="mt-8">
            <FeedbackWidget
              key="statistics-feedback"
              feature="dashboard-statistics"
              context={{
                tab: 'statistics',
                total_listings: statistics?.total_listings,
              }}
            />
          </div>
        </>
      )}

      {activeTab === 'clustering' && (
        <>
          {/* Cluster Summary Cards */}
          <div className="grid grid-cols-1 md:grid-cols-3 lg:grid-cols-5 gap-4 mb-8">
            {clusterStats.map((cluster, idx) => (
              <motion.div
                key={cluster.cluster_id}
                initial={{ opacity: 0, scale: 0.95 }}
                animate={{ opacity: 1, scale: 1 }}
                transition={{ delay: idx * 0.1 }}
                className="glass-card-hover p-5"
              >
                <div className="flex items-center justify-between mb-3">
                  <span
                    className={`cluster-badge ${cluster.label.toLowerCase().replace(' ', '-')}`}
                  >
                    {cluster.label}
                  </span>
                  <span className="text-surface-500 text-sm">{cluster.count} items</span>
                </div>
                <div className="space-y-2">
                  <div>
                    <p className="text-surface-400 text-xs">Avg Price</p>
                    <p className="font-mono text-lg text-white">
                      €{Math.round(cluster.avg_price).toLocaleString()}
                    </p>
                  </div>
                  <div className="grid grid-cols-2 gap-2 text-sm">
                    <div>
                      <p className="text-surface-500 text-xs">RAM</p>
                      <p className="text-surface-200">{Math.round(cluster.avg_ram)} GB</p>
                    </div>
                    <div>
                      <p className="text-surface-500 text-xs">SSD</p>
                      <p className="text-surface-200">{Math.round(cluster.avg_ssd)} GB</p>
                    </div>
                  </div>
                </div>
              </motion.div>
            ))}
          </div>

          {/* Scatter Plot */}
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.3 }}
            className="glass-card p-6"
          >
            <h3 className="font-display text-lg font-semibold text-white mb-4">
              Price vs RAM Segmentation
            </h3>
            <p className="text-surface-400 text-sm mb-6">
              K-means clustering visualization showing how products are segmented based on price and specifications
            </p>
            <div className="h-96">
              <ResponsiveContainer width="100%" height="100%">
                <ScatterChart margin={{ top: 20, right: 20, bottom: 20, left: 20 }}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#334155" />
                  <XAxis
                    type="number"
                    dataKey="ram"
                    name="RAM (GB)"
                    unit=" GB"
                    stroke="#64748b"
                    domain={[0, 'auto']}
                  />
                  <YAxis
                    type="number"
                    dataKey="price"
                    name="Price"
                    stroke="#64748b"
                    tickFormatter={(v) => `€${v}`}
                  />
                  <Tooltip
                    content={({ active, payload }) => {
                      if (active && payload && payload.length) {
                        const data = payload[0].payload
                        return (
                          <div className="glass-card p-3 border border-surface-600">
                            <p className="text-surface-300 font-medium mb-1">{data.brand}</p>
                            <p className="text-white font-mono">€{data.price.toLocaleString()}</p>
                            <p className="text-surface-400 text-sm">
                              RAM: {data.ram}GB | SSD: {data.ssd}GB
                            </p>
                          </div>
                        )
                      }
                      return null
                    }}
                  />
                  <Legend
                    formatter={(value) => (
                      <span className="text-surface-300">{value}</span>
                    )}
                  />
                  {clusterStats.map((cluster) => (
                    <Scatter
                      key={cluster.cluster_id}
                      name={cluster.label}
                      data={scatterData.filter((d) => d.cluster === cluster.cluster_id)}
                      fill={LABEL_COLORS[cluster.label] || '#8b5cf6'}
                      opacity={0.7}
                    />
                  ))}
                </ScatterChart>
              </ResponsiveContainer>
            </div>
          </motion.div>

          {/* Cluster Details Table */}
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.4 }}
            className="glass-card p-6 mt-6"
          >
            <h3 className="font-display text-lg font-semibold text-white mb-4">
              Segment Analysis
            </h3>
            <div className="overflow-x-auto">
              <table className="w-full">
                <thead>
                  <tr className="border-b border-surface-700">
                    <th className="text-left py-3 px-4 text-surface-400 font-medium">Segment</th>
                    <th className="text-right py-3 px-4 text-surface-400 font-medium">Count</th>
                    <th className="text-right py-3 px-4 text-surface-400 font-medium">Avg Price</th>
                    <th className="text-right py-3 px-4 text-surface-400 font-medium">RAM</th>
                    <th className="text-right py-3 px-4 text-surface-400 font-medium">SSD</th>
                    <th className="text-left py-3 px-4 text-surface-400 font-medium">Top Brands</th>
                  </tr>
                </thead>
                <tbody>
                  {clusterStats.map((cluster, idx) => (
                    <tr key={cluster.cluster_id} className="border-b border-surface-800 hover:bg-surface-800/30">
                      <td className="py-3 px-4">
                        <span className={`cluster-badge ${cluster.label.toLowerCase().replace(' ', '-')}`}>
                          {cluster.label}
                        </span>
                      </td>
                      <td className="py-3 px-4 text-right text-surface-200">{cluster.count.toLocaleString()}</td>
                      <td className="py-3 px-4 text-right font-mono text-emerald-400">
                        €{Math.round(cluster.avg_price).toLocaleString()}
                      </td>
                      <td className="py-3 px-4 text-right text-surface-200">{Math.round(cluster.avg_ram)} GB</td>
                      <td className="py-3 px-4 text-right text-surface-200">{Math.round(cluster.avg_ssd)} GB</td>
                      <td className="py-3 px-4 text-surface-300">
                        {Object.keys(cluster.top_brands || {}).slice(0, 3).join(', ')}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </motion.div>

          {/* Feedback Widget for Segmentation */}
          <div className="mt-8">
            <FeedbackWidget
              key="segmentation-feedback"
              feature="dashboard-segmentation"
              context={{
                tab: 'segmentation',
                n_clusters: clustering?.n_clusters,
              }}
            />
          </div>
        </>
      )}
    </div>
  )
}


