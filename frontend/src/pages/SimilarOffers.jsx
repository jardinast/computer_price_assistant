import { useState, useEffect } from 'react'
import { motion } from 'framer-motion'
import {
  Search,
  Package,
  TrendingUp,
  TrendingDown,
  Minus,
  Monitor,
  HardDrive,
  Cpu,
  Filter,
  ArrowRight,
  AlertCircle,
  RefreshCw,
  Laptop,
  Zap,
} from 'lucide-react'
import axios from 'axios'

const API_BASE = '/api'

function OfferCard({ offer, predictedPrice, index }) {
  const priceDiff = offer.price_difference
  const diffPercent = predictedPrice ? ((offer.real_price - predictedPrice) / predictedPrice * 100) : 0

  const getPriceDiffColor = () => {
    if (!priceDiff) return 'text-surface-400'
    if (priceDiff < -50) return 'text-emerald-400'
    if (priceDiff < 0) return 'text-emerald-300'
    if (priceDiff > 100) return 'text-red-400'
    if (priceDiff > 0) return 'text-amber-400'
    return 'text-surface-400'
  }

  const getPriceDiffIcon = () => {
    if (!priceDiff) return Minus
    if (priceDiff < 0) return TrendingDown
    if (priceDiff > 0) return TrendingUp
    return Minus
  }

  const DiffIcon = getPriceDiffIcon()

  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ delay: index * 0.05 }}
      className="glass-card-hover p-5"
    >
      <div className="flex items-start justify-between mb-3">
        <div className="flex-1">
          <div className="flex items-center gap-2 mb-1">
            <span className="text-xs text-surface-500 font-mono">#{index + 1}</span>
            <span className="px-2 py-0.5 bg-surface-800 rounded text-xs text-surface-400">
              {offer.brand}
            </span>
            {offer.product_type && (
              <span className="px-2 py-0.5 bg-primary-500/10 text-primary-400 rounded text-xs">
                {offer.product_type.replace('Portátil ', '')}
              </span>
            )}
          </div>
          <h3 className="text-white font-medium leading-tight line-clamp-2">
            {offer.title}
          </h3>
        </div>
        <div className="text-right ml-4">
          <div className="price-tag">
            €{Math.round(offer.real_price).toLocaleString()}
          </div>
        </div>
      </div>

      {/* Specs */}
      <div className="grid grid-cols-3 gap-3 mb-3">
        {offer.ram_gb && (
          <div className="text-center p-2 bg-surface-800/50 rounded-lg">
            <p className="text-surface-500 text-xs mb-1">RAM</p>
            <p className="text-surface-200 font-medium">{offer.ram_gb} GB</p>
          </div>
        )}
        {offer.ssd_gb && (
          <div className="text-center p-2 bg-surface-800/50 rounded-lg">
            <p className="text-surface-500 text-xs mb-1">Storage</p>
            <p className="text-surface-200 font-medium">
              {offer.ssd_gb >= 1000 ? `${offer.ssd_gb / 1000} TB` : `${offer.ssd_gb} GB`}
            </p>
          </div>
        )}
        {offer.screen_inches && (
          <div className="text-center p-2 bg-surface-800/50 rounded-lg">
            <p className="text-surface-500 text-xs mb-1">Screen</p>
            <p className="text-surface-200 font-medium">{offer.screen_inches}"</p>
          </div>
        )}
      </div>

      {/* Price Comparison */}
      {predictedPrice && priceDiff !== null && (
        <div className={`flex items-center justify-between p-3 rounded-lg ${
          priceDiff < 0 ? 'bg-emerald-500/10' : priceDiff > 50 ? 'bg-red-500/10' : 'bg-amber-500/10'
        }`}>
          <span className="text-surface-400 text-sm">vs. Predicted</span>
          <div className={`flex items-center gap-1 font-mono ${getPriceDiffColor()}`}>
            <DiffIcon className="w-4 h-4" />
            <span>
              {priceDiff > 0 ? '+' : ''}€{Math.round(priceDiff).toLocaleString()}
              <span className="text-xs ml-1">({diffPercent > 0 ? '+' : ''}{diffPercent.toFixed(1)}%)</span>
            </span>
          </div>
        </div>
      )}

      {/* Similarity Score */}
      <div className="mt-3 pt-3 border-t border-surface-800">
        <div className="flex items-center justify-between text-sm">
          <span className="text-surface-500">Similarity Score</span>
          <div className="flex items-center gap-2">
            <div className="w-24 h-1.5 bg-surface-700 rounded-full overflow-hidden">
              <div
                className="h-full bg-gradient-to-r from-primary-500 to-accent-500 rounded-full"
                style={{ width: `${Math.max(0, 100 - offer.distance * 100)}%` }}
              />
            </div>
            <span className="text-surface-300 font-mono text-xs">
              {(100 - offer.distance * 100).toFixed(0)}%
            </span>
          </div>
        </div>
      </div>
    </motion.div>
  )
}

export default function SimilarOffers() {
  const [options, setOptions] = useState(null)
  const [loading, setLoading] = useState(true)
  const [searching, setSearching] = useState(false)
  const [results, setResults] = useState(null)
  const [error, setError] = useState(null)

  // Search form state
  const [useCase, setUseCase] = useState('general')
  const [brand, setBrand] = useState('')
  const [ramGb, setRamGb] = useState(16)
  const [ssdGb, setSsdGb] = useState(512)
  const [screenSize, setScreenSize] = useState(15.6)
  const [cpuFamily, setCpuFamily] = useState('core i5')
  const [gpuIntegrated, setGpuIntegrated] = useState(true)
  const [numResults, setNumResults] = useState(10)

  useEffect(() => {
    fetchOptions()
  }, [])

  const fetchOptions = async () => {
    try {
      const res = await axios.get(`${API_BASE}/predictive/options`)
      setOptions(res.data)
    } catch (err) {
      setError(err.message)
    } finally {
      setLoading(false)
    }
  }

  const handleSearch = async () => {
    setSearching(true)
    setError(null)

    const inputs = {
      use_case: useCase,
      _brand: brand || null,
      _ram_gb: ramGb,
      _ssd_gb: ssdGb,
      _tamano_pantalla_pulgadas: screenSize,
      cpu_family: cpuFamily,
      gpu_is_integrated: gpuIntegrated,
      k: numResults,
    }

    try {
      const res = await axios.post(`${API_BASE}/prescriptive/similar`, inputs)
      setResults(res.data)
    } catch (err) {
      setError(err.response?.data?.error || err.message)
    } finally {
      setSearching(false)
    }
  }

  const handleReset = () => {
    setUseCase('general')
    setBrand('')
    setRamGb(16)
    setSsdGb(512)
    setScreenSize(15.6)
    setCpuFamily('core i5')
    setGpuIntegrated(true)
    setNumResults(10)
    setResults(null)
    setError(null)
  }

  if (loading) {
    return (
      <div className="flex items-center justify-center min-h-[60vh]">
        <div className="text-center">
          <div className="spinner mx-auto mb-4" />
          <p className="text-surface-400">Loading options...</p>
        </div>
      </div>
    )
  }

  return (
    <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
      {/* Page Header */}
      <div className="mb-8">
        <h1 className="font-display text-3xl font-bold text-white mb-2">
          Find <span className="gradient-text">Similar Offers</span>
        </h1>
        <p className="text-surface-400">
          Enter your desired specifications to find the k-best matching computer offers
        </p>
      </div>

      {/* Search Form */}
      <div className="glass-card p-6 mb-8">
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4 mb-6">
          {/* RAM */}
          <div>
            <label className="block text-surface-300 text-sm mb-2">
              <HardDrive className="w-4 h-4 inline mr-1" />
              RAM
            </label>
            <select
              value={ramGb}
              onChange={(e) => setRamGb(parseInt(e.target.value))}
              className="select-field"
            >
              {[4, 8, 16, 32, 64].map((gb) => (
                <option key={gb} value={gb}>{gb} GB</option>
              ))}
            </select>
          </div>

          {/* SSD */}
          <div>
            <label className="block text-surface-300 text-sm mb-2">
              <HardDrive className="w-4 h-4 inline mr-1" />
              Storage
            </label>
            <select
              value={ssdGb}
              onChange={(e) => setSsdGb(parseInt(e.target.value))}
              className="select-field"
            >
              {[128, 256, 512, 1000, 2000].map((gb) => (
                <option key={gb} value={gb}>{gb >= 1000 ? `${gb / 1000} TB` : `${gb} GB`}</option>
              ))}
            </select>
          </div>

          {/* Screen Size */}
          <div>
            <label className="block text-surface-300 text-sm mb-2">
              <Monitor className="w-4 h-4 inline mr-1" />
              Screen Size
            </label>
            <select
              value={screenSize}
              onChange={(e) => setScreenSize(parseFloat(e.target.value))}
              className="select-field"
            >
              {[13, 14, 15, 15.6, 16, 17].map((size) => (
                <option key={size} value={size}>{size}"</option>
              ))}
            </select>
          </div>

          {/* Brand Filter */}
          <div>
            <label className="block text-surface-300 text-sm mb-2">
              <Laptop className="w-4 h-4 inline mr-1" />
              Brand (Optional)
            </label>
            <select
              value={brand}
              onChange={(e) => setBrand(e.target.value)}
              className="select-field"
            >
              <option value="">Any Brand</option>
              {(options?.brands || []).map((b) => (
                <option key={b} value={b}>{b}</option>
              ))}
            </select>
          </div>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mb-6">
          {/* CPU Family */}
          <div>
            <label className="block text-surface-300 text-sm mb-2">
              <Cpu className="w-4 h-4 inline mr-1" />
              CPU Family
            </label>
            <select
              value={cpuFamily}
              onChange={(e) => setCpuFamily(e.target.value)}
              className="select-field"
            >
              {['Core i3', 'Core i5', 'Core i7', 'Core i9', 'Ryzen 5', 'Ryzen 7', 'Ryzen 9', 'M1', 'M2', 'M3'].map((fam) => (
                <option key={fam} value={fam.toLowerCase()}>{fam}</option>
              ))}
            </select>
          </div>

          {/* GPU Type */}
          <div>
            <label className="block text-surface-300 text-sm mb-2">
              <Zap className="w-4 h-4 inline mr-1" />
              Graphics
            </label>
            <select
              value={gpuIntegrated ? 'integrated' : 'dedicated'}
              onChange={(e) => setGpuIntegrated(e.target.value === 'integrated')}
              className="select-field"
            >
              <option value="integrated">Integrated GPU</option>
              <option value="dedicated">Dedicated GPU</option>
            </select>
          </div>

          {/* Number of Results */}
          <div>
            <label className="block text-surface-300 text-sm mb-2">
              <Filter className="w-4 h-4 inline mr-1" />
              Number of Results
            </label>
            <select
              value={numResults}
              onChange={(e) => setNumResults(parseInt(e.target.value))}
              className="select-field"
            >
              {[5, 10, 15, 20, 25].map((n) => (
                <option key={n} value={n}>{n} offers</option>
              ))}
            </select>
          </div>
        </div>

        {/* Action Buttons */}
        <div className="flex gap-3">
          <button
            onClick={handleSearch}
            disabled={searching}
            className="btn-primary flex items-center gap-2"
          >
            {searching ? (
              <>
                <div className="spinner w-5 h-5" />
                Searching...
              </>
            ) : (
              <>
                <Search className="w-5 h-5" />
                Find Similar Offers
                <ArrowRight className="w-4 h-4" />
              </>
            )}
          </button>
          <button onClick={handleReset} className="btn-secondary px-4">
            <RefreshCw className="w-5 h-5" />
          </button>
        </div>
      </div>

      {/* Error Display */}
      {error && (
        <div className="glass-card p-4 mb-8 border-red-500/30 bg-red-500/5">
          <div className="flex items-start gap-3">
            <AlertCircle className="w-5 h-5 text-red-400 flex-shrink-0 mt-0.5" />
            <div>
              <p className="text-red-400 font-medium">Search Error</p>
              <p className="text-surface-400 text-sm">{error}</p>
            </div>
          </div>
        </div>
      )}

      {/* Results */}
      {results && (
        <motion.div
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
        >
          {/* Summary Card */}
          <div className="glass-card p-6 mb-6">
            <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
              {/* Predicted Price */}
              <div className="text-center">
                <p className="text-surface-400 text-sm mb-2">Your Predicted Price</p>
                <div className="price-tag-large inline-flex">
                  €{results.predicted_price ? Math.round(results.predicted_price).toLocaleString() : 'N/A'}
                </div>
              </div>

              {/* Target Specs */}
              <div className="text-center">
                <p className="text-surface-400 text-sm mb-2">Target Specifications</p>
                <div className="flex items-center justify-center gap-4">
                  <span className="px-3 py-1 bg-surface-800 rounded-lg text-surface-200">
                    {results.target_features?.ram_gb} GB RAM
                  </span>
                  <span className="px-3 py-1 bg-surface-800 rounded-lg text-surface-200">
                    {results.target_features?.ssd_gb >= 1000 
                      ? `${results.target_features.ssd_gb / 1000} TB` 
                      : `${results.target_features?.ssd_gb} GB`} SSD
                  </span>
                  <span className="px-3 py-1 bg-surface-800 rounded-lg text-surface-200">
                    {results.target_features?.screen_inches}"
                  </span>
                </div>
              </div>

              {/* Results Count */}
              <div className="text-center">
                <p className="text-surface-400 text-sm mb-2">Matching Offers</p>
                <p className="font-display text-3xl font-bold text-white">
                  {results.similar_offers?.length || 0}
                </p>
              </div>
            </div>
          </div>

          {/* Offers Grid */}
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            {results.similar_offers?.map((offer, index) => (
              <OfferCard
                key={index}
                offer={offer}
                predictedPrice={results.predicted_price}
                index={index}
              />
            ))}
          </div>

          {/* Empty State */}
          {results.similar_offers?.length === 0 && (
            <div className="glass-card p-12 text-center">
              <Package className="w-16 h-16 text-surface-600 mx-auto mb-4" />
              <h3 className="text-xl font-medium text-white mb-2">No Matching Offers Found</h3>
              <p className="text-surface-400">
                Try adjusting your search criteria to find more results
              </p>
            </div>
          )}
        </motion.div>
      )}

      {/* Initial State */}
      {!results && !searching && (
        <div className="glass-card p-12 text-center">
          <Search className="w-16 h-16 text-surface-600 mx-auto mb-4" />
          <h3 className="text-xl font-medium text-white mb-2">Enter Your Desired Specifications</h3>
          <p className="text-surface-400 max-w-md mx-auto">
            Configure RAM, storage, screen size, and other features above, then click "Find Similar Offers" to discover matching computers at various price points
          </p>
        </div>
      )}
    </div>
  )
}


