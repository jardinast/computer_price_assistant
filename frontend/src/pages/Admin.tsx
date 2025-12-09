import { useState, useEffect } from 'react'
import { Lock, RefreshCw, ThumbsUp, ThumbsDown, BarChart3, Download } from 'lucide-react'
import axios from 'axios'
import { API_BASE } from '../config'

interface FeedbackEntry {
  timestamp: string
  feature: string
  rating: string
  sliders?: Record<string, number>
  comment?: string
  context?: Record<string, any>
}

interface FeedbackStats {
  total: number
  positive: number
  negative: number
  positive_rate: number
  by_feature: Record<string, { total: number; positive: number; negative: number }>
  avg_slider_ratings: Record<string, number>
}

export default function Admin() {
  const [adminKey, setAdminKey] = useState('')
  const [authenticated, setAuthenticated] = useState(false)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [stats, setStats] = useState<FeedbackStats | null>(null)
  const [feedback, setFeedback] = useState<FeedbackEntry[]>([])
  const [filter, setFilter] = useState<string>('')

  const fetchData = async () => {
    setLoading(true)
    setError(null)
    try {
      const [statsRes, feedbackRes] = await Promise.all([
        axios.get(`${API_BASE}/admin/feedback/stats`, { params: { admin_key: adminKey } }),
        axios.get(`${API_BASE}/admin/feedback`, { params: { admin_key: adminKey, limit: 100, feature: filter || undefined } }),
      ])
      setStats(statsRes.data)
      setFeedback(feedbackRes.data.feedback)
      setAuthenticated(true)
    } catch (err: any) {
      if (err.response?.status === 401) {
        setError('Invalid admin key')
        setAuthenticated(false)
      } else {
        setError(err.message || 'Failed to fetch data')
      }
    } finally {
      setLoading(false)
    }
  }

  const handleLogin = () => {
    if (adminKey.trim()) {
      fetchData()
    }
  }

  const handleKeyPress = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter') {
      handleLogin()
    }
  }

  const downloadFeedback = () => {
    const dataStr = JSON.stringify(feedback, null, 2)
    const blob = new Blob([dataStr], { type: 'application/json' })
    const url = URL.createObjectURL(blob)
    const a = document.createElement('a')
    a.href = url
    a.download = `feedback_${new Date().toISOString().split('T')[0]}.json`
    a.click()
    URL.revokeObjectURL(url)
  }

  useEffect(() => {
    if (authenticated && adminKey) {
      fetchData()
    }
  }, [filter])

  if (!authenticated) {
    return (
      <div className="max-w-md mx-auto mt-20">
        <div className="glass-card p-8 text-center">
          <Lock className="w-12 h-12 text-primary-400 mx-auto mb-4" />
          <h1 className="text-2xl font-bold text-white mb-2">Admin Access</h1>
          <p className="text-surface-400 mb-6">Enter admin key to view feedback data</p>

          <input
            type="password"
            value={adminKey}
            onChange={(e) => setAdminKey(e.target.value)}
            onKeyDown={handleKeyPress}
            placeholder="Admin key..."
            className="input-field w-full mb-4"
          />

          {error && (
            <p className="text-red-400 text-sm mb-4">{error}</p>
          )}

          <button
            onClick={handleLogin}
            disabled={loading || !adminKey.trim()}
            className="btn-primary w-full"
          >
            {loading ? 'Authenticating...' : 'Access Dashboard'}
          </button>

          <p className="text-surface-500 text-xs mt-4">
            Default key: admin123 (set ADMIN_KEY env var in Railway)
          </p>
        </div>
      </div>
    )
  }

  return (
    <div className="max-w-6xl mx-auto px-4 py-8">
      {/* Header */}
      <div className="flex items-center justify-between mb-8">
        <div>
          <h1 className="text-3xl font-bold text-white">Feedback Dashboard</h1>
          <p className="text-surface-400">Monitor user feedback across all features</p>
        </div>
        <div className="flex gap-2">
          <button onClick={fetchData} className="btn-secondary flex items-center gap-2">
            <RefreshCw className={`w-4 h-4 ${loading ? 'animate-spin' : ''}`} />
            Refresh
          </button>
          <button onClick={downloadFeedback} className="btn-secondary flex items-center gap-2">
            <Download className="w-4 h-4" />
            Export JSON
          </button>
        </div>
      </div>

      {/* Stats Cards */}
      {stats && (
        <div className="grid grid-cols-1 md:grid-cols-4 gap-4 mb-8">
          <div className="glass-card p-6">
            <div className="flex items-center gap-3">
              <BarChart3 className="w-8 h-8 text-primary-400" />
              <div>
                <p className="text-surface-400 text-sm">Total Feedback</p>
                <p className="text-2xl font-bold text-white">{stats.total}</p>
              </div>
            </div>
          </div>

          <div className="glass-card p-6">
            <div className="flex items-center gap-3">
              <ThumbsUp className="w-8 h-8 text-emerald-400" />
              <div>
                <p className="text-surface-400 text-sm">Positive</p>
                <p className="text-2xl font-bold text-emerald-400">{stats.positive}</p>
              </div>
            </div>
          </div>

          <div className="glass-card p-6">
            <div className="flex items-center gap-3">
              <ThumbsDown className="w-8 h-8 text-red-400" />
              <div>
                <p className="text-surface-400 text-sm">Negative</p>
                <p className="text-2xl font-bold text-red-400">{stats.negative}</p>
              </div>
            </div>
          </div>

          <div className="glass-card p-6">
            <div>
              <p className="text-surface-400 text-sm">Positive Rate</p>
              <p className="text-2xl font-bold text-white">{stats.positive_rate}%</p>
            </div>
          </div>
        </div>
      )}

      {/* Stats by Feature */}
      {stats && Object.keys(stats.by_feature).length > 0 && (
        <div className="glass-card p-6 mb-8">
          <h2 className="text-lg font-semibold text-white mb-4">By Feature</h2>
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
            {Object.entries(stats.by_feature).map(([feature, data]) => (
              <div key={feature} className="bg-surface-800/50 rounded-lg p-4">
                <p className="text-surface-300 font-medium capitalize mb-2">{feature}</p>
                <div className="flex items-center gap-4 text-sm">
                  <span className="text-emerald-400">👍 {data.positive}</span>
                  <span className="text-red-400">👎 {data.negative}</span>
                  <span className="text-surface-400">Total: {data.total}</span>
                </div>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Average Slider Ratings */}
      {stats && Object.keys(stats.avg_slider_ratings).length > 0 && (
        <div className="glass-card p-6 mb-8">
          <h2 className="text-lg font-semibold text-white mb-4">Average Ratings</h2>
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
            {Object.entries(stats.avg_slider_ratings).map(([key, value]) => (
              <div key={key} className="bg-surface-800/50 rounded-lg p-4">
                <p className="text-surface-400 text-sm capitalize mb-1">{key.replace('_', ' ')}</p>
                <p className="text-xl font-bold text-white">{value}/10</p>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Filter and Feedback List */}
      <div className="glass-card p-6">
        <div className="flex items-center justify-between mb-4">
          <h2 className="text-lg font-semibold text-white">Recent Feedback</h2>
          <select
            value={filter}
            onChange={(e) => setFilter(e.target.value)}
            className="select-field w-40"
          >
            <option value="">All Features</option>
            <option value="predictor">Predictor</option>
            <option value="similar">Similar Offers</option>
            <option value="chat">Chat</option>
            <option value="dashboard">Dashboard</option>
          </select>
        </div>

        {feedback.length === 0 ? (
          <p className="text-surface-400 text-center py-8">No feedback yet</p>
        ) : (
          <div className="space-y-3 max-h-96 overflow-y-auto">
            {feedback.map((entry, i) => (
              <div key={i} className="bg-surface-800/50 rounded-lg p-4">
                <div className="flex items-start justify-between mb-2">
                  <div className="flex items-center gap-3">
                    <span className={`text-xl ${entry.rating === 'positive' ? 'text-emerald-400' : 'text-red-400'}`}>
                      {entry.rating === 'positive' ? '👍' : '👎'}
                    </span>
                    <span className="px-2 py-1 bg-surface-700 rounded text-xs text-surface-300 capitalize">
                      {entry.feature}
                    </span>
                  </div>
                  <span className="text-surface-500 text-xs">
                    {new Date(entry.timestamp).toLocaleString()}
                  </span>
                </div>

                {entry.comment && (
                  <p className="text-surface-200 text-sm mb-2">"{entry.comment}"</p>
                )}

                {entry.sliders && Object.keys(entry.sliders).length > 0 && (
                  <div className="flex flex-wrap gap-2 mb-2">
                    {Object.entries(entry.sliders).map(([key, val]) => (
                      <span key={key} className="text-xs bg-surface-700 px-2 py-1 rounded text-surface-400">
                        {key}: {val}/10
                      </span>
                    ))}
                  </div>
                )}

                {entry.context && (
                  <details className="text-xs text-surface-500">
                    <summary className="cursor-pointer hover:text-surface-400">Context</summary>
                    <pre className="mt-1 p-2 bg-surface-900 rounded overflow-x-auto">
                      {JSON.stringify(entry.context, null, 2)}
                    </pre>
                  </details>
                )}
              </div>
            ))}
          </div>
        )}
      </div>
    </div>
  )
}

