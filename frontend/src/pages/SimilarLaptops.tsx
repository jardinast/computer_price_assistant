import { useState } from 'react';
import { RadarChart, PolarGrid, PolarAngleAxis, PolarRadiusAxis, Radar, Legend, ResponsiveContainer, BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip } from 'recharts';
import { Loader2, Search, Laptop, ChevronDown, ChevronUp } from 'lucide-react';
import { findSimilarLaptops } from '../services/api';
import type { SimilarLaptop } from '../types';

const BRANDS = ['Any', 'ASUS', 'Dell', 'HP', 'Lenovo', 'Apple', 'Acer', 'MSI'];
const CPU_BRANDS = ['Any', 'Intel', 'AMD', 'Apple'];

export default function SimilarLaptops() {
  const [loading, setLoading] = useState(false);
  const [results, setResults] = useState<SimilarLaptop[]>([]);
  const [error, setError] = useState<string | null>(null);

  const [formData, setFormData] = useState({
    price_target: 1000,
    ram_gb: 16,
    ssd_gb: 512,
    screen_size: 15.6,
    brand: 'Any',
    cpu_brand: 'Any',
    num_results: 5,
  });

  const handleSearch = async (e: React.FormEvent) => {
    e.preventDefault();
    setLoading(true);
    setError(null);

    try {
      const params = {
        price_target: formData.price_target,
        ram_gb: formData.ram_gb,
        ssd_gb: formData.ssd_gb,
        screen_size: formData.screen_size,
        num_results: formData.num_results,
        ...(formData.brand !== 'Any' && { brand: formData.brand }),
        ...(formData.cpu_brand !== 'Any' && { cpu_brand: formData.cpu_brand }),
      };

      const response = await findSimilarLaptops(params);
      setResults(response.laptops);
    } catch (err) {
      setError('Failed to find similar laptops. Please try again.');
      console.error(err);
    } finally {
      setLoading(false);
    }
  };

  const updateField = (field: string, value: any) => {
    setFormData((prev) => ({ ...prev, [field]: value }));
  };

  const getMatchQuality = (distance: number, maxDistance: number) => {
    const score = 1 - distance / (maxDistance + 0.01);
    if (score >= 0.8) return { label: 'Excellent', color: 'bg-green-100 text-green-700' };
    if (score >= 0.6) return { label: 'Good', color: 'bg-blue-100 text-blue-700' };
    if (score >= 0.4) return { label: 'Fair', color: 'bg-yellow-100 text-yellow-700' };
    return { label: 'Partial', color: 'bg-orange-100 text-orange-700' };
  };

  const maxDistance = results.length > 0 ? Math.max(...results.map((r) => r.distance)) : 1;

  // Prepare radar chart data
  const radarData = results.length > 0
    ? [
        { metric: 'Price', target: formData.price_target / 100, ...Object.fromEntries(results.map((r, i) => [`laptop${i}`, r.price / 100])) },
        { metric: 'RAM', target: formData.ram_gb, ...Object.fromEntries(results.map((r, i) => [`laptop${i}`, r.ram_gb || 0])) },
        { metric: 'Storage', target: formData.ssd_gb / 100, ...Object.fromEntries(results.map((r, i) => [`laptop${i}`, (r.ssd_gb || 0) / 100])) },
        { metric: 'Screen', target: formData.screen_size, ...Object.fromEntries(results.map((r, i) => [`laptop${i}`, r.screen_size || 0])) },
      ]
    : [];

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-3xl font-bold text-gray-900">Find Similar Laptops</h1>
        <p className="text-gray-600 mt-1">
          Discover laptops that match your requirements
        </p>
      </div>

      <div className="grid lg:grid-cols-3 gap-6">
        {/* Search Form */}
        <div className="card">
          <h2 className="text-lg font-semibold text-gray-900 mb-4">Search Criteria</h2>

          <form onSubmit={handleSearch} className="space-y-4">
            <div>
              <label className="label">Target Price (EUR)</label>
              <input
                type="number"
                value={formData.price_target}
                onChange={(e) => updateField('price_target', Number(e.target.value))}
                min={200}
                max={5000}
                step={50}
                className="input"
              />
            </div>

            <div>
              <label className="label">RAM (GB)</label>
              <select
                value={formData.ram_gb}
                onChange={(e) => updateField('ram_gb', Number(e.target.value))}
                className="input"
              >
                {[4, 8, 16, 32, 64].map((r) => (
                  <option key={r} value={r}>{r} GB</option>
                ))}
              </select>
            </div>

            <div>
              <label className="label">Storage (GB)</label>
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

            <div>
              <label className="label">Screen Size (inches)</label>
              <input
                type="number"
                value={formData.screen_size}
                onChange={(e) => updateField('screen_size', Number(e.target.value))}
                min={11}
                max={18}
                step={0.1}
                className="input"
              />
            </div>

            <div>
              <label className="label">Preferred Brand</label>
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

            <div>
              <label className="label">CPU Brand</label>
              <select
                value={formData.cpu_brand}
                onChange={(e) => updateField('cpu_brand', e.target.value)}
                className="input"
              >
                {CPU_BRANDS.map((b) => (
                  <option key={b} value={b}>{b}</option>
                ))}
              </select>
            </div>

            <div>
              <label className="label">Number of Results</label>
              <select
                value={formData.num_results}
                onChange={(e) => updateField('num_results', Number(e.target.value))}
                className="input"
              >
                {[3, 5, 10, 15, 20].map((n) => (
                  <option key={n} value={n}>{n}</option>
                ))}
              </select>
            </div>

            <button
              type="submit"
              disabled={loading}
              className="btn-primary w-full flex items-center justify-center gap-2"
            >
              {loading ? (
                <Loader2 className="h-5 w-5 animate-spin" />
              ) : (
                <Search className="h-5 w-5" />
              )}
              {loading ? 'Searching...' : 'Find Matches'}
            </button>
          </form>

          {error && (
            <div className="mt-4 p-3 bg-red-50 text-red-700 rounded-lg text-sm">
              {error}
            </div>
          )}
        </div>

        {/* Results */}
        <div className="lg:col-span-2 space-y-6">
          {results.length > 0 ? (
            <>
              {/* Laptop Cards */}
              <div className="space-y-4">
                <h3 className="font-semibold text-gray-900">
                  Found {results.length} similar laptops
                </h3>

                {results.map((laptop) => {
                  const match = getMatchQuality(laptop.distance, maxDistance);
                  return (
                    <div key={laptop.rank} className="card">
                      <div className="flex items-start justify-between">
                        <div className="flex-1">
                          <div className="flex items-center gap-2 mb-1">
                            <span className="text-lg font-bold text-gray-400">#{laptop.rank}</span>
                            <span className={`px-2 py-0.5 rounded-full text-xs font-medium ${match.color}`}>
                              {match.label} Match
                            </span>
                          </div>
                          <h4 className="font-medium text-gray-900 line-clamp-2">
                            {laptop.title}
                          </h4>
                          <div className="flex flex-wrap gap-2 mt-2">
                            <span className="text-xs bg-gray-100 px-2 py-1 rounded">
                              {laptop.brand}
                            </span>
                            {laptop.ram_gb && (
                              <span className="text-xs bg-gray-100 px-2 py-1 rounded">
                                {laptop.ram_gb} GB RAM
                              </span>
                            )}
                            {laptop.ssd_gb && (
                              <span className="text-xs bg-gray-100 px-2 py-1 rounded">
                                {laptop.ssd_gb < 1000 ? `${laptop.ssd_gb} GB` : `${laptop.ssd_gb / 1000} TB`}
                              </span>
                            )}
                            {laptop.screen_size && (
                              <span className="text-xs bg-gray-100 px-2 py-1 rounded">
                                {laptop.screen_size.toFixed(1)}"
                              </span>
                            )}
                            <span className="text-xs bg-gray-100 px-2 py-1 rounded">
                              {laptop.cpu_brand}
                            </span>
                          </div>
                        </div>
                        <div className="text-right ml-4">
                          <div className="text-2xl font-bold text-primary-600">
                            €{laptop.price.toFixed(0)}
                          </div>
                          <div className="text-xs text-gray-500">
                            Range: €{laptop.price_min.toFixed(0)} - €{laptop.price_max.toFixed(0)}
                          </div>
                        </div>
                      </div>
                    </div>
                  );
                })}
              </div>

              {/* Comparison Charts */}
              <div className="card">
                <h3 className="font-semibold text-gray-900 mb-4">Price Comparison</h3>
                <ResponsiveContainer width="100%" height={250}>
                  <BarChart
                    data={results.map((r) => ({
                      name: `#${r.rank} ${r.brand}`,
                      price: r.price,
                    }))}
                  >
                    <CartesianGrid strokeDasharray="3 3" />
                    <XAxis dataKey="name" tick={{ fontSize: 12 }} />
                    <YAxis />
                    <Tooltip formatter={(value: number) => `€${value.toFixed(0)}`} />
                    <Bar dataKey="price" fill="#4F8BF9" />
                  </BarChart>
                </ResponsiveContainer>

                {/* Target price line annotation */}
                <div className="mt-2 text-center text-sm text-gray-500">
                  Your target: €{formData.price_target}
                </div>
              </div>
            </>
          ) : (
            <div className="card text-center py-12">
              <Laptop className="h-16 w-16 mx-auto text-gray-300 mb-4" />
              <h3 className="text-lg font-medium text-gray-900">No Results Yet</h3>
              <p className="text-gray-500 mt-1">
                Set your criteria and click "Find Matches" to discover similar laptops
              </p>
            </div>
          )}
        </div>
      </div>

      {/* Info Note */}
      <div className="text-center text-sm text-gray-500">
        Similarity is calculated based on price, RAM, storage, and screen size.
        Results show laptops from our database that most closely match your requirements.
      </div>
    </div>
  );
}
