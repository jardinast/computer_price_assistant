import { useState } from 'react';
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from 'recharts';
import { Loader2, Calculator, ChevronDown, ChevronUp, Info } from 'lucide-react';
import { predictPrice } from '../services/api';
import type { PredictionRequest, PredictionResponse } from '../types';

const USE_CASES = [
  { value: 'general', label: 'General Use' },
  { value: 'gaming', label: 'Gaming' },
  { value: 'work', label: 'Work/Business' },
  { value: 'creative', label: 'Creative (Video/Photo)' },
  { value: 'student', label: 'Student' },
];

const BRANDS = ['ASUS', 'Dell', 'HP', 'Lenovo', 'Apple', 'Acer', 'MSI', 'Razer', 'Samsung', 'Microsoft'];
const CPU_BRANDS = ['Intel', 'AMD', 'Apple'];
const CPU_FAMILIES: Record<string, string[]> = {
  Intel: ['Core i3', 'Core i5', 'Core i7', 'Core i9', 'Core Ultra 5', 'Core Ultra 7', 'Core Ultra 9', 'Celeron', 'Pentium'],
  AMD: ['Ryzen 3', 'Ryzen 5', 'Ryzen 7', 'Ryzen 9', 'Athlon'],
  Apple: ['M1', 'M1 Pro', 'M1 Max', 'M2', 'M2 Pro', 'M2 Max', 'M3', 'M3 Pro', 'M3 Max'],
};
const GPU_OPTIONS = [
  'Integrated',
  'NVIDIA RTX 4050',
  'NVIDIA RTX 4060',
  'NVIDIA RTX 4070',
  'NVIDIA RTX 4080',
  'NVIDIA RTX 3050',
  'NVIDIA RTX 3060',
  'AMD RX 6600',
  'AMD RX 7600',
];

export default function PricePredictor() {
  const [mode, setMode] = useState<'simple' | 'advanced'>('simple');
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState<PredictionResponse | null>(null);
  const [error, setError] = useState<string | null>(null);

  // Form state
  const [formData, setFormData] = useState({
    use_case: 'general',
    brand: 'ASUS',
    cpu_brand: 'Intel',
    cpu_family: 'Core i5',
    ram_gb: 16,
    ssd_gb: 512,
    gpu_option: 'Integrated',
    // Advanced options
    screen_size: 15.6,
    refresh_rate: 60,
    weight_kg: 1.8,
    cpu_cores: 8,
    gpu_memory: 0,
  });

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setLoading(true);
    setError(null);

    try {
      const isIntegrated = formData.gpu_option === 'Integrated';
      let gpuSeries = 'integrated';
      let gpuBrand = 'intel';

      if (!isIntegrated) {
        if (formData.gpu_option.includes('NVIDIA')) {
          gpuBrand = 'nvidia';
          gpuSeries = formData.gpu_option.replace('NVIDIA ', '').toLowerCase();
        } else if (formData.gpu_option.includes('AMD')) {
          gpuBrand = 'amd';
          gpuSeries = formData.gpu_option.replace('AMD ', '').toLowerCase();
        }
      }

      const request: PredictionRequest = {
        brand: formData.brand,
        cpu_brand: formData.cpu_brand,
        cpu_family: formData.cpu_family.toLowerCase(),
        ram_gb: formData.ram_gb,
        ssd_gb: formData.ssd_gb,
        gpu_brand: gpuBrand,
        gpu_series: gpuSeries,
        gpu_is_integrated: isIntegrated,
        use_case: formData.use_case,
        ...(mode === 'advanced' && {
          screen_size: formData.screen_size,
          refresh_rate: formData.refresh_rate,
          weight_kg: formData.weight_kg,
        }),
      };

      const response = await predictPrice(request);
      setResult(response);
    } catch (err) {
      setError('Failed to get prediction. Please try again.');
      console.error(err);
    } finally {
      setLoading(false);
    }
  };

  const updateField = (field: string, value: any) => {
    setFormData((prev) => ({ ...prev, [field]: value }));
  };

  const featureData = result?.top_features.slice(0, 8).map((f) => ({
    name: f.readable_name,
    importance: f.importance_pct,
  }));

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-3xl font-bold text-gray-900">Price Predictor</h1>
        <p className="text-gray-600 mt-1">
          Enter laptop specifications to get a price estimate
        </p>
      </div>

      {/* Mode Toggle */}
      <div className="flex justify-center">
        <div className="inline-flex rounded-lg border border-gray-200 bg-white p-1">
          <button
            onClick={() => setMode('simple')}
            className={`px-4 py-2 rounded-md text-sm font-medium transition-colors ${
              mode === 'simple'
                ? 'bg-primary-500 text-white'
                : 'text-gray-600 hover:bg-gray-100'
            }`}
          >
            Simple Mode
          </button>
          <button
            onClick={() => setMode('advanced')}
            className={`px-4 py-2 rounded-md text-sm font-medium transition-colors ${
              mode === 'advanced'
                ? 'bg-primary-500 text-white'
                : 'text-gray-600 hover:bg-gray-100'
            }`}
          >
            Advanced Mode
          </button>
        </div>
      </div>

      <div className="grid lg:grid-cols-2 gap-6">
        {/* Form */}
        <div className="card">
          <h2 className="text-lg font-semibold text-gray-900 mb-4">Specifications</h2>

          <form onSubmit={handleSubmit} className="space-y-4">
            {/* Use Case */}
            <div>
              <label className="label">What will you use this laptop for?</label>
              <select
                value={formData.use_case}
                onChange={(e) => updateField('use_case', e.target.value)}
                className="input"
              >
                {USE_CASES.map((uc) => (
                  <option key={uc.value} value={uc.value}>{uc.label}</option>
                ))}
              </select>
            </div>

            {/* Brand */}
            <div>
              <label className="label">Brand</label>
              <select
                value={formData.brand}
                onChange={(e) => updateField('brand', e.target.value)}
                className="input"
              >
                {BRANDS.map((b) => (
                  <option key={b} value={b}>{b}</option>
                ))}
              </select>
            </div>

            {/* CPU */}
            <div className="grid grid-cols-2 gap-4">
              <div>
                <label className="label">CPU Brand</label>
                <select
                  value={formData.cpu_brand}
                  onChange={(e) => {
                    updateField('cpu_brand', e.target.value);
                    updateField('cpu_family', CPU_FAMILIES[e.target.value][0]);
                  }}
                  className="input"
                >
                  {CPU_BRANDS.map((b) => (
                    <option key={b} value={b}>{b}</option>
                  ))}
                </select>
              </div>
              <div>
                <label className="label">CPU Type</label>
                <select
                  value={formData.cpu_family}
                  onChange={(e) => updateField('cpu_family', e.target.value)}
                  className="input"
                >
                  {CPU_FAMILIES[formData.cpu_brand].map((f) => (
                    <option key={f} value={f}>{f}</option>
                  ))}
                </select>
              </div>
            </div>

            {/* RAM & Storage */}
            <div className="grid grid-cols-2 gap-4">
              <div>
                <label className="label">RAM (GB)</label>
                <select
                  value={formData.ram_gb}
                  onChange={(e) => updateField('ram_gb', Number(e.target.value))}
                  className="input"
                >
                  {[4, 8, 16, 32, 64, 128].map((r) => (
                    <option key={r} value={r}>{r} GB</option>
                  ))}
                </select>
              </div>
              <div>
                <label className="label">Storage</label>
                <select
                  value={formData.ssd_gb}
                  onChange={(e) => updateField('ssd_gb', Number(e.target.value))}
                  className="input"
                >
                  {[128, 256, 512, 1000, 2000].map((s) => (
                    <option key={s} value={s}>{s < 1000 ? `${s} GB` : `${s / 1000} TB`}</option>
                  ))}
                </select>
              </div>
            </div>

            {/* GPU */}
            <div>
              <label className="label">Graphics</label>
              <select
                value={formData.gpu_option}
                onChange={(e) => updateField('gpu_option', e.target.value)}
                className="input"
              >
                {GPU_OPTIONS.map((g) => (
                  <option key={g} value={g}>{g}</option>
                ))}
              </select>
            </div>

            {/* Advanced Options */}
            {mode === 'advanced' && (
              <div className="space-y-4 pt-4 border-t border-gray-200">
                <h3 className="font-medium text-gray-900">Advanced Options</h3>

                <div className="grid grid-cols-3 gap-4">
                  <div>
                    <label className="label">Screen Size</label>
                    <input
                      type="number"
                      step="0.1"
                      min="11"
                      max="18"
                      value={formData.screen_size}
                      onChange={(e) => updateField('screen_size', Number(e.target.value))}
                      className="input"
                    />
                  </div>
                  <div>
                    <label className="label">Refresh Rate (Hz)</label>
                    <select
                      value={formData.refresh_rate}
                      onChange={(e) => updateField('refresh_rate', Number(e.target.value))}
                      className="input"
                    >
                      {[60, 90, 120, 144, 165, 240, 360].map((r) => (
                        <option key={r} value={r}>{r} Hz</option>
                      ))}
                    </select>
                  </div>
                  <div>
                    <label className="label">Weight (kg)</label>
                    <input
                      type="number"
                      step="0.1"
                      min="0.5"
                      max="5"
                      value={formData.weight_kg}
                      onChange={(e) => updateField('weight_kg', Number(e.target.value))}
                      className="input"
                    />
                  </div>
                </div>
              </div>
            )}

            <button
              type="submit"
              disabled={loading}
              className="btn-primary w-full flex items-center justify-center gap-2"
            >
              {loading ? (
                <Loader2 className="h-5 w-5 animate-spin" />
              ) : (
                <Calculator className="h-5 w-5" />
              )}
              {loading ? 'Calculating...' : 'Predict Price'}
            </button>
          </form>

          {error && (
            <div className="mt-4 p-3 bg-red-50 text-red-700 rounded-lg">
              {error}
            </div>
          )}
        </div>

        {/* Results */}
        <div className="card">
          <h2 className="text-lg font-semibold text-gray-900 mb-4">Price Estimate</h2>

          {result ? (
            <div className="space-y-6">
              {/* Price Display */}
              <div className="text-center p-6 bg-gradient-to-br from-primary-50 to-primary-100 rounded-xl">
                <div className="text-4xl font-bold text-primary-600">
                  €{result.predicted_price.toFixed(0)}
                </div>
                <div className="text-gray-600 mt-2">
                  Range: €{result.price_min.toFixed(0)} - €{result.price_max.toFixed(0)}
                </div>
                <div className={`inline-block mt-3 px-3 py-1 rounded-full text-sm font-medium ${
                  result.confidence === 'high'
                    ? 'bg-green-100 text-green-700'
                    : result.confidence === 'medium'
                    ? 'bg-yellow-100 text-yellow-700'
                    : 'bg-red-100 text-red-700'
                }`}>
                  Confidence: {result.confidence}
                </div>
              </div>

              {/* Explanation */}
              <div className="p-4 bg-blue-50 rounded-lg flex items-start gap-3">
                <Info className="h-5 w-5 text-blue-500 flex-shrink-0 mt-0.5" />
                <p className="text-sm text-blue-700">{result.explanation}</p>
              </div>

              {/* Feature Importance Chart */}
              <div>
                <h3 className="font-medium text-gray-900 mb-3">What's Driving This Price?</h3>
                <ResponsiveContainer width="100%" height={250}>
                  <BarChart data={featureData} layout="vertical">
                    <CartesianGrid strokeDasharray="3 3" />
                    <XAxis type="number" unit="%" />
                    <YAxis dataKey="name" type="category" width={120} tick={{ fontSize: 12 }} />
                    <Tooltip formatter={(value: number) => `${value.toFixed(1)}%`} />
                    <Bar dataKey="importance" fill="#4F8BF9" />
                  </BarChart>
                </ResponsiveContainer>
              </div>
            </div>
          ) : (
            <div className="text-center py-12 text-gray-500">
              <Calculator className="h-12 w-12 mx-auto mb-4 text-gray-300" />
              <p>Enter specifications and click "Predict Price" to see the estimate</p>
            </div>
          )}
        </div>
      </div>

      {/* Info Note */}
      <div className="text-center text-sm text-gray-500">
        Price predictions are estimates based on a machine learning model. Actual prices may vary.
        Model accuracy: approximately ±20% (MAPE).
      </div>
    </div>
  );
}
