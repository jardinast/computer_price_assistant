import { useState, useEffect } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  Cell,
} from 'recharts'
import {
  Calculator,
  Cpu,
  HardDrive,
  Monitor,
  Laptop,
  Zap,
  Gauge,
  ChevronDown,
  ChevronUp,
  Lightbulb,
  TrendingUp,
  AlertCircle,
  Check,
  X,
  RefreshCw,
  Sliders,
} from 'lucide-react'
import axios from 'axios'
import { API_BASE } from '../config'
import FeedbackWidget from '../components/FeedbackWidget'

const USE_CASE_INFO = {
  general: { label: 'General Use', icon: Laptop, color: 'primary', description: 'Everyday tasks, browsing, office work' },
  gaming: { label: 'Gaming', icon: Zap, color: 'accent', description: 'High-performance gaming laptops' },
  work: { label: 'Professional', icon: Monitor, color: 'emerald', description: 'Business and productivity' },
  creative: { label: 'Creative', icon: TrendingUp, color: 'amber', description: 'Video editing, design, 3D work' },
  student: { label: 'Student', icon: Lightbulb, color: 'cyan', description: 'Budget-friendly for students' },
}

function FormSection({ title, children, icon: Icon, defaultOpen = true }) {
  const [isOpen, setIsOpen] = useState(defaultOpen)

  return (
    <div className="glass-card overflow-hidden">
      <button
        onClick={() => setIsOpen(!isOpen)}
        className="w-full flex items-center justify-between p-4 hover:bg-surface-800/30 transition-colors"
      >
        <div className="flex items-center gap-3">
          <div className="p-2 rounded-lg bg-primary-500/10 text-primary-400">
            <Icon className="w-5 h-5" />
          </div>
          <span className="font-medium text-white">{title}</span>
        </div>
        {isOpen ? (
          <ChevronUp className="w-5 h-5 text-surface-400" />
        ) : (
          <ChevronDown className="w-5 h-5 text-surface-400" />
        )}
      </button>
      <AnimatePresence>
        {isOpen && (
          <motion.div
            initial={{ height: 0, opacity: 0 }}
            animate={{ height: 'auto', opacity: 1 }}
            exit={{ height: 0, opacity: 0 }}
            transition={{ duration: 0.2 }}
          >
            <div className="p-4 pt-0 space-y-4 border-t border-surface-800">
              {children}
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  )
}

function SliderInput({ label, value, onChange, min, max, step = 1, unit = '', options = null }) {
  const displayValue = options ? options.find(o => o.value === value)?.label || value : value

  return (
    <div>
      <div className="flex items-center justify-between mb-2">
        <label className="text-surface-300 text-sm">{label}</label>
        <span className="font-mono text-primary-400">{displayValue}{unit}</span>
      </div>
      {options ? (
        <input
          type="range"
          min={0}
          max={options.length - 1}
          value={options.findIndex(o => o.value === value)}
          onChange={(e) => onChange(options[parseInt(e.target.value)].value)}
          className="w-full"
        />
      ) : (
        <input
          type="range"
          min={min}
          max={max}
          step={step}
          value={value}
          onChange={(e) => onChange(parseFloat(e.target.value))}
          className="w-full"
        />
      )}
      <div className="flex justify-between text-xs text-surface-500 mt-1">
        <span>{options ? options[0].label : `${min}${unit}`}</span>
        <span>{options ? options[options.length - 1].label : `${max}${unit}`}</span>
      </div>
    </div>
  )
}

function SelectInput({ label, value, onChange, options, placeholder = 'Select...' }) {
  return (
    <div>
      <label className="block text-surface-300 text-sm mb-2">{label}</label>
      <select
        value={value}
        onChange={(e) => onChange(e.target.value)}
        className="select-field"
      >
        <option value="">{placeholder}</option>
        {options.map((opt) => (
          <option key={typeof opt === 'string' ? opt : opt.value} value={typeof opt === 'string' ? opt : opt.value}>
            {typeof opt === 'string' ? opt : opt.label}
          </option>
        ))}
      </select>
    </div>
  )
}

function CheckboxInput({ label, checked, onChange, description }) {
  return (
    <label className="flex items-start gap-3 cursor-pointer group">
      <input
        type="checkbox"
        checked={checked}
        onChange={(e) => onChange(e.target.checked)}
        className="checkbox-custom mt-0.5"
      />
      <div>
        <span className="text-surface-200 group-hover:text-white transition-colors">{label}</span>
        {description && <p className="text-surface-500 text-sm">{description}</p>}
      </div>
    </label>
  )
}

// Individual Price Breakdown (SHAP values) - shows € contribution for THIS prediction
function ShapBreakdownChart({ features, basePrice, compact = false }) {
  const truncateName = (name, maxLength = 16) => {
    if (name.length <= maxLength) return name
    return name.substring(0, maxLength - 1) + '…'
  }

  // Limit to top 7 features for compact view
  const limitedFeatures = compact ? features.slice(0, 7) : features
  const data = limitedFeatures.map((f) => ({
    name: truncateName(f.readable_name),
    fullName: f.readable_name,
    value: f.shap_value,
    isPositive: f.shap_value >= 0,
  }))

  return (
    <div className={compact ? "h-56" : "h-64"}>
      <ResponsiveContainer width="100%" height="100%">
        <BarChart data={data} layout="vertical" margin={{ left: 15, right: 35 }}>
          <CartesianGrid strokeDasharray="3 3" stroke="#334155" horizontal={false} />
          <XAxis 
            type="number" 
            stroke="#64748b" 
            tickFormatter={(v) => `${v >= 0 ? '+' : ''}€${Math.round(v)}`}
            tick={{ fontSize: 11 }}
          />
          <YAxis dataKey="name" type="category" stroke="#64748b" width={100} tick={{ fontSize: 11 }} interval={0} />
          <Tooltip
            content={({ active, payload }) => {
              if (active && payload && payload.length) {
                const val = payload[0].value
                return (
                  <div className="glass-card p-2 border border-surface-600">
                    <p className="text-white font-medium text-sm">{payload[0].payload.fullName}</p>
                    <p className={`font-mono text-sm ${val >= 0 ? 'text-emerald-400' : 'text-red-400'}`}>
                      {val >= 0 ? '+' : ''}€{Math.round(val)} to price
                    </p>
                  </div>
                )
              }
              return null
            }}
          />
          <Bar 
            dataKey="value" 
            radius={[0, 4, 4, 0]}
            fill="#10b981"
          >
            {data.map((entry, index) => (
              <Cell key={`cell-${index}`} fill={entry.isPositive ? '#10b981' : '#ef4444'} />
            ))}
          </Bar>
        </BarChart>
      </ResponsiveContainer>
    </div>
  )
}

// Global Feature Importance - shows what the model generally values
function FeatureImportanceChart({ features, compact = false }) {
  // Truncate long feature names to prevent overlap
  const truncateName = (name, maxLength = 16) => {
    if (name.length <= maxLength) return name
    return name.substring(0, maxLength - 1) + '…'
  }
  
  // Limit to top 7 features for compact view
  const limitedFeatures = compact ? features.slice(0, 7) : features
  const data = limitedFeatures.map((f) => ({
    name: truncateName(f.readable_name),
    fullName: f.readable_name,
    importance: f.importance_pct,
  }))

  return (
    <div className={compact ? "h-56" : "h-64"}>
      <ResponsiveContainer width="100%" height="100%">
        <BarChart data={data} layout="vertical" margin={{ left: 15, right: 20 }}>
          <CartesianGrid strokeDasharray="3 3" stroke="#334155" horizontal={false} />
          <XAxis type="number" stroke="#64748b" tickFormatter={(v) => `${v.toFixed(0)}%`} tick={{ fontSize: 11 }} />
          <YAxis dataKey="name" type="category" stroke="#64748b" width={100} tick={{ fontSize: 11 }} interval={0} />
          <Tooltip
            content={({ active, payload }) => {
              if (active && payload && payload.length) {
                return (
                  <div className="glass-card p-2 border border-surface-600">
                    <p className="text-white font-medium text-sm">{payload[0].payload.fullName}</p>
                    <p className="text-primary-400 font-mono text-sm">{payload[0].value.toFixed(1)}% importance</p>
                  </div>
                )
              }
              return null
            }}
          />
          <Bar dataKey="importance" fill="url(#importanceGradient)" radius={[0, 4, 4, 0]} />
          <defs>
            <linearGradient id="importanceGradient" x1="0" y1="0" x2="1" y2="0">
              <stop offset="0%" stopColor="#0ea5e9" />
              <stop offset="100%" stopColor="#d946ef" />
            </linearGradient>
          </defs>
        </BarChart>
      </ResponsiveContainer>
    </div>
  )
}

export default function Predictor() {
  const [mode, setMode] = useState('simple')
  const [options, setOptions] = useState(null)
  const [loading, setLoading] = useState(true)
  const [predicting, setPredicting] = useState(false)
  const [prediction, setPrediction] = useState(null)
  const [error, setError] = useState(null)
  const [showConfig, setShowConfig] = useState(true) // Toggle config panel visibility

  // Form state
  const [useCase, setUseCase] = useState('general')
  const [brand, setBrand] = useState('')
  const [cpuBrand, setCpuBrand] = useState('Intel')
  const [cpuFamily, setCpuFamily] = useState('Core i5')
  const [gpuType, setGpuType] = useState('Integrated')
  const [gpuSeries, setGpuSeries] = useState('')
  const [gpuIntegrated, setGpuIntegrated] = useState(true)
  const [ramGb, setRamGb] = useState(16)
  const [ssdGb, setSsdGb] = useState(512)
  const [screenSize, setScreenSize] = useState(15.6)
  const [refreshRate, setRefreshRate] = useState(60)
  const [weightKg, setWeightKg] = useState(1.8)
  const [cpuCores, setCpuCores] = useState(6)
  const [gpuMemoryGb, setGpuMemoryGb] = useState(0)
  const [resolutionPixels, setResolutionPixels] = useState(2073600)
  const [hasWifi, setHasWifi] = useState(true)
  const [hasBluetooth, setHasBluetooth] = useState(true)
  const [hasWebcam, setHasWebcam] = useState(true)

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

  const handlePredict = async () => {
    setPredicting(true)
    setError(null)

    const inputs = {
      use_case: useCase,
      _brand: brand || undefined,
      cpu_family: cpuFamily.toLowerCase(),
      cpu_brand: cpuBrand.toLowerCase(),
      gpu_series: gpuSeries || (gpuIntegrated ? 'integrated' : ''),
      gpu_is_integrated: gpuIntegrated,
      _ram_gb: ramGb,
      _ssd_gb: ssdGb,
    }

    if (mode === 'advanced') {
      inputs._tamano_pantalla_pulgadas = screenSize
      inputs._tasa_refresco_hz = refreshRate
      inputs._peso_kg = weightKg
      inputs._cpu_cores = cpuCores
      inputs._gpu_memory_gb = gpuMemoryGb
      inputs._resolucion_pixeles = resolutionPixels
      inputs._tiene_wifi = hasWifi ? 1 : 0
      inputs._tiene_bluetooth = hasBluetooth ? 1 : 0
      inputs._tiene_webcam = hasWebcam ? 1 : 0
    }

    try {
      const res = await axios.post(`${API_BASE}/predictive/predict`, inputs)
      setPrediction(res.data)
      setShowConfig(false) // Collapse config panel after successful prediction
    } catch (err) {
      setError(err.response?.data?.error || err.message)
    } finally {
      setPredicting(false)
    }
  }

  const handleReset = () => {
    setUseCase('general')
    setBrand('')
    setCpuBrand('Intel')
    setCpuFamily('Core i5')
    setGpuType('Integrated')
    setGpuSeries('')
    setGpuIntegrated(true)
    setRamGb(16)
    setSsdGb(512)
    setScreenSize(15.6)
    setRefreshRate(60)
    setWeightKg(1.8)
    setCpuCores(6)
    setGpuMemoryGb(0)
    setResolutionPixels(2073600)
    setHasWifi(true)
    setHasBluetooth(true)
    setHasWebcam(true)
    setMode('simple')
    setPrediction(null)
    setError(null)
  }

  const ramOptions = [
    { value: 4, label: '4 GB' },
    { value: 8, label: '8 GB' },
    { value: 16, label: '16 GB' },
    { value: 32, label: '32 GB' },
    { value: 64, label: '64 GB' },
  ]

  const ssdOptions = [
    { value: 128, label: '128 GB' },
    { value: 256, label: '256 GB' },
    { value: 512, label: '512 GB' },
    { value: 1000, label: '1 TB' },
    { value: 2000, label: '2 TB' },
  ]

  const refreshOptions = [
    { value: 60, label: '60 Hz' },
    { value: 90, label: '90 Hz' },
    { value: 120, label: '120 Hz' },
    { value: 144, label: '144 Hz' },
    { value: 165, label: '165 Hz' },
    { value: 240, label: '240 Hz' },
  ]

  const cpuCoreOptions = [
    { value: 4, label: '4 Cores' },
    { value: 6, label: '6 Cores' },
    { value: 8, label: '8 Cores' },
    { value: 10, label: '10 Cores' },
    { value: 12, label: '12 Cores' },
    { value: 16, label: '16 Cores' },
  ]

  const gpuMemoryOptions = [
    { value: 0, label: '0 GB' },
    { value: 4, label: '4 GB' },
    { value: 6, label: '6 GB' },
    { value: 8, label: '8 GB' },
    { value: 12, label: '12 GB' },
    { value: 16, label: '16 GB' },
  ]

  const resolutionOptions = [
    { value: 2073600, label: 'Full HD (1920x1080)' },
    { value: 3686400, label: 'QHD (2560x1440)' },
    { value: 8294400, label: '4K UHD (3840x2160)' },
  ]

  if (loading) {
    return (
      <div className="flex items-center justify-center min-h-[60vh]">
        <div className="text-center">
          <div className="spinner mx-auto mb-4" />
          <p className="text-surface-400">Loading configuration...</p>
        </div>
      </div>
    )
  }

  return (
    <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
      {/* Page Header */}
      <div className="mb-8">
        <h1 className="font-display text-3xl font-bold text-white mb-2">
          Price <span className="gradient-text">Predictor</span>
        </h1>
        <p className="text-surface-400">
          Configure your ideal computer specifications and get an AI-powered price estimate
        </p>
      </div>

      <div className="flex justify-center mb-6">
        <div className="inline-flex rounded-xl border border-surface-800 bg-surface-900/60 p-1">
          <button
            onClick={() => setMode('simple')}
            className={`px-4 py-2 rounded-lg text-sm font-medium transition-all ${
              mode === 'simple'
                ? 'bg-primary-500 text-white'
                : 'text-surface-400 hover:text-surface-200'
            }`}
          >
            Simple Mode
          </button>
          <button
            onClick={() => setMode('advanced')}
            className={`px-4 py-2 rounded-lg text-sm font-medium transition-all ${
              mode === 'advanced'
                ? 'bg-primary-500 text-white'
                : 'text-surface-400 hover:text-surface-200'
            }`}
          >
            Advanced Mode
          </button>
        </div>
      </div>

      {/* Results Section (shown after prediction) */}
      {prediction && (
        <motion.div
          initial={{ opacity: 0, y: -20 }}
          animate={{ opacity: 1, y: 0 }}
          className="mb-6"
        >
          {/* Top row: Price + Config Summary + Toggle Button */}
          <div className="flex flex-wrap items-start gap-4 mb-4">
            {/* Price Card */}
            <div className="glass-card p-5 flex-1 min-w-[200px]">
              <p className="text-surface-400 text-sm mb-1">Estimated Price</p>
              <div className="text-4xl font-bold gradient-text mb-2">
                €{Math.round(prediction.predicted_price).toLocaleString()}
              </div>
              <div className="flex items-center gap-2 text-sm text-surface-400 mb-3">
                <span>€{Math.round(prediction.price_range.min).toLocaleString()}</span>
                <span className="text-surface-600">—</span>
                <span>€{Math.round(prediction.price_range.max).toLocaleString()}</span>
              </div>
              <div className={`inline-flex items-center gap-1.5 px-2.5 py-1 rounded-full text-xs ${
                prediction.confidence === 'high'
                  ? 'bg-emerald-500/10 text-emerald-400 border border-emerald-500/20'
                  : prediction.confidence === 'medium'
                  ? 'bg-amber-500/10 text-amber-400 border border-amber-500/20'
                  : 'bg-red-500/10 text-red-400 border border-red-500/20'
              }`}>
                {prediction.confidence === 'high' ? <Check className="w-3 h-3" /> : 
                 prediction.confidence === 'medium' ? <AlertCircle className="w-3 h-3" /> :
                 <X className="w-3 h-3" />}
                {prediction.confidence.charAt(0).toUpperCase() + prediction.confidence.slice(1)} Confidence
              </div>
            </div>

            {/* Config Summary Card */}
            <div className="glass-card p-5 flex-1 min-w-[200px]">
              <h3 className="text-white text-sm font-medium mb-3">Configuration</h3>
              <div className="grid grid-cols-2 gap-x-6 gap-y-1 text-sm">
                <div className="flex justify-between">
                  <span className="text-surface-500">CPU</span>
                  <span className="text-surface-300">{cpuBrand} {cpuFamily}</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-surface-500">RAM</span>
                  <span className="text-surface-300">{ramGb} GB</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-surface-500">Storage</span>
                  <span className="text-surface-300">{ssdGb >= 1000 ? `${ssdGb / 1000} TB` : `${ssdGb} GB`}</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-surface-500">GPU</span>
                  <span className="text-surface-300">{gpuIntegrated ? 'Integrated' : gpuSeries}</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-surface-500">Screen</span>
                  <span className="text-surface-300">{screenSize}"</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-surface-500">Use Case</span>
                  <span className="text-surface-300">{USE_CASE_INFO[useCase].label}</span>
                </div>
              </div>
            </div>

            {/* Toggle Config Button */}
            <button
              onClick={() => setShowConfig(!showConfig)}
              className="btn-secondary px-4 py-3 flex items-center gap-2"
            >
              <Sliders className="w-4 h-4" />
              {showConfig ? 'Hide Config' : 'Modify Config'}
            </button>
          </div>

          {/* Charts Row - Full Width, Bigger */}
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
            {/* SHAP Chart */}
            {prediction.shap_features && prediction.shap_features.length > 0 && (
              <div className="glass-card p-5">
                <h3 className="text-white font-medium mb-1">Your Price Breakdown</h3>
                <p className="text-surface-400 text-sm mb-3">How each feature affects YOUR predicted price</p>
                <ShapBreakdownChart features={prediction.shap_features} basePrice={prediction.predicted_price} />
              </div>
            )}

            {/* Model Importance Chart */}
            {prediction.top_features && (
              <div className="glass-card p-5">
                <h3 className="text-white font-medium mb-1">Model Feature Importance</h3>
                <p className="text-surface-400 text-sm mb-3">What the model generally considers important</p>
                <FeatureImportanceChart features={prediction.top_features} />
              </div>
            )}
          </div>

          {/* Feedback Widget */}
          <div className="mt-4">
            <FeedbackWidget
              feature="predictor"
              context={{
                predicted_price: prediction.predicted_price,
                brand: brand,
                cpu_family: cpuFamily,
                ram_gb: ramGb,
                ssd_gb: ssdGb,
              }}
            />
          </div>
        </motion.div>
      )}

      {/* Config Form - Collapsible */}
      <AnimatePresence>
      {showConfig && (
        <motion.div
          initial={{ opacity: 0, height: 0 }}
          animate={{ opacity: 1, height: 'auto' }}
          exit={{ opacity: 0, height: 0 }}
          transition={{ duration: 0.3 }}
          className="overflow-hidden"
        >
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Form Column */}
        <div className="lg:col-span-2 space-y-4">
          {/* Use Case Selection */}
          <div className="glass-card p-4">
            <h3 className="text-white font-medium mb-4">What's your primary use case?</h3>
            <div className="grid grid-cols-2 md:grid-cols-5 gap-2">
              {Object.entries(USE_CASE_INFO).map(([key, info]) => (
                <button
                  key={key}
                  onClick={() => setUseCase(key)}
                  className={`p-3 rounded-xl border transition-all ${
                    useCase === key
                      ? 'bg-primary-500/20 border-primary-500/50 text-white'
                      : 'bg-surface-800/30 border-surface-700 text-surface-400 hover:border-surface-600'
                  }`}
                >
                  <info.icon className={`w-5 h-5 mx-auto mb-1 ${useCase === key ? 'text-primary-400' : ''}`} />
                  <span className="text-xs font-medium">{info.label}</span>
                </button>
              ))}
            </div>
            <p className="text-surface-500 text-sm mt-3">
              {USE_CASE_INFO[useCase].description}
            </p>
          </div>

          {/* Basic Configuration */}
          <FormSection title="Basic Configuration" icon={Laptop}>
            <div className={`grid grid-cols-1 ${mode === 'simple' ? 'md:grid-cols-2' : ''} gap-4`}>
              <SelectInput
                label="Brand (Optional)"
                value={brand}
                onChange={setBrand}
                options={options?.brands || []}
                placeholder="Any Brand"
              />
              {mode === 'simple' && (
                <SliderInput
                  label="Screen Size"
                  value={screenSize}
                  onChange={setScreenSize}
                  min={11}
                  max={18}
                  step={0.1}
                  unit='"'
                />
              )}
            </div>
            {mode === 'advanced' && (
              <p className="text-surface-500 text-sm">Screen size can be adjusted in the Advanced Controls below.</p>
            )}
          </FormSection>

          {/* Processor */}
          <FormSection title="Processor (CPU)" icon={Cpu}>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              <SelectInput
                label="CPU Brand"
                value={cpuBrand}
                onChange={(v) => {
                  setCpuBrand(v)
                  setCpuFamily(options?.cpu_families?.[v]?.[0] || '')
                }}
                options={Object.keys(options?.cpu_families || {})}
              />
              <SelectInput
                label="CPU Family"
                value={cpuFamily}
                onChange={setCpuFamily}
                options={options?.cpu_families?.[cpuBrand] || []}
              />
            </div>
          </FormSection>

          {/* Graphics */}
          <FormSection title="Graphics (GPU)" icon={Monitor}>
            <CheckboxInput
              label="Integrated Graphics"
              checked={gpuIntegrated}
              onChange={(v) => {
                setGpuIntegrated(v)
                if (v) setGpuSeries('')
              }}
              description="Built-in graphics (no dedicated GPU)"
            />
            {!gpuIntegrated && (
              <motion.div
                initial={{ opacity: 0, height: 0 }}
                animate={{ opacity: 1, height: 'auto' }}
                exit={{ opacity: 0, height: 0 }}
                className="grid grid-cols-1 md:grid-cols-2 gap-4"
              >
                <SelectInput
                  label="GPU Brand"
                  value={gpuType}
                  onChange={(v) => {
                    setGpuType(v)
                    setGpuSeries(options?.gpu_options?.[v]?.[0] || '')
                  }}
                  options={Object.keys(options?.gpu_options || {}).filter(k => k !== 'Integrated')}
                />
                <SelectInput
                  label="GPU Model"
                  value={gpuSeries}
                  onChange={setGpuSeries}
                  options={options?.gpu_options?.[gpuType] || []}
                />
              </motion.div>
            )}
          </FormSection>

          {/* Memory & Storage */}
          <FormSection title="Memory & Storage" icon={HardDrive}>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
              <SliderInput
                label="RAM"
                value={ramGb}
                onChange={setRamGb}
                options={ramOptions}
              />
              <SliderInput
                label="Storage (SSD)"
                value={ssdGb}
                onChange={setSsdGb}
                options={ssdOptions}
              />
            </div>
          </FormSection>

          {mode === 'advanced' && (
            <>
              <FormSection title="Display & Build" icon={Gauge}>
                <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                  <SliderInput
                    label="Screen Size"
                    value={screenSize}
                    onChange={setScreenSize}
                    min={11}
                    max={18}
                    step={0.1}
                    unit='"'
                  />
                  <SliderInput
                    label="Refresh Rate"
                    value={refreshRate}
                    onChange={setRefreshRate}
                    options={refreshOptions}
                  />
                  <SliderInput
                    label="Weight (kg)"
                    value={weightKg}
                    onChange={setWeightKg}
                    min={0.8}
                    max={4.5}
                    step={0.1}
                    unit="kg"
                  />
                </div>
              </FormSection>

              <FormSection title="Performance Tuning" icon={Sliders}>
                <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                  <SliderInput
                    label="CPU Cores"
                    value={cpuCores}
                    onChange={setCpuCores}
                    options={cpuCoreOptions}
                  />
                  <SliderInput
                    label="GPU Memory"
                    value={gpuMemoryGb}
                    onChange={setGpuMemoryGb}
                    options={gpuMemoryOptions}
                  />
                  <SelectInput
                    label="Screen Resolution"
                    value={resolutionPixels}
                    onChange={(v) => setResolutionPixels(Number(v))}
                    options={resolutionOptions}
                  />
                </div>
                <div className="grid grid-cols-1 md:grid-cols-3 gap-4 pt-4 border-t border-surface-800">
                  <CheckboxInput
                    label="Wi-Fi"
                    checked={hasWifi}
                    onChange={setHasWifi}
                    description="Wireless connectivity included"
                  />
                  <CheckboxInput
                    label="Bluetooth"
                    checked={hasBluetooth}
                    onChange={setHasBluetooth}
                    description="Bluetooth support"
                  />
                  <CheckboxInput
                    label="Webcam"
                    checked={hasWebcam}
                    onChange={setHasWebcam}
                    description="Built-in webcam"
                  />
                </div>
              </FormSection>
            </>
          )}

          {/* Action Buttons */}
          <div className="flex gap-3">
            <button
              onClick={handlePredict}
              disabled={predicting}
              className="btn-primary flex-1 flex items-center justify-center gap-2"
            >
              {predicting ? (
                <>
                  <div className="spinner w-5 h-5" />
                  Calculating...
                </>
              ) : (
                <>
                  <Calculator className="w-5 h-5" />
                  Predict Price
                </>
              )}
            </button>
            <button onClick={handleReset} className="btn-secondary px-4">
              <RefreshCw className="w-5 h-5" />
            </button>
          </div>
        </div>

        {/* Results Column - Placeholder or Feedback */}
        <div className="space-y-4">
          {/* Placeholder when no prediction */}
          {!prediction && (
            <motion.div
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              className="glass-card p-6"
            >
              <h3 className="text-surface-400 text-sm mb-4">Estimated Price</h3>
              <div className="text-center py-8">
                <Calculator className="w-12 h-12 text-surface-600 mx-auto mb-3" />
                <p className="text-surface-500">Configure specs and click "Predict Price"</p>
              </div>
            </motion.div>
          )}

          {/* Error Display */}
          {error && (
            <motion.div
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              className="glass-card p-4 border-red-500/30 bg-red-500/5"
            >
              <div className="flex items-start gap-3">
                <AlertCircle className="w-5 h-5 text-red-400 flex-shrink-0 mt-0.5" />
                <div>
                  <p className="text-red-400 font-medium">Prediction Error</p>
                  <p className="text-surface-400 text-sm">{error}</p>
                </div>
              </div>
            </motion.div>
          )}
        </div>
      </div>
        </motion.div>
      )}
      </AnimatePresence>
    </div>
  )
}


