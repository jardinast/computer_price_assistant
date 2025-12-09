import { useState } from 'react'
import { ThumbsUp, ThumbsDown, Send, X, ChevronDown, ChevronUp } from 'lucide-react'
import axios from 'axios'
import { API_BASE } from '../config'

interface SliderConfig {
  key: string
  label: string
}

interface FeedbackWidgetProps {
  feature: 'predictor' | 'similar' | 'chat' | 'dashboard'
  context?: Record<string, any>
  onClose?: () => void
}

// Feature-specific slider configurations
const SLIDER_CONFIGS: Record<string, SliderConfig[]> = {
  predictor: [
    { key: 'accuracy', label: 'Accuracy' },
    { key: 'helpfulness', label: 'Helpfulness' },
    { key: 'ease_of_use', label: 'Ease of Use' },
  ],
  similar: [
    { key: 'relevance', label: 'Relevance' },
    { key: 'helpfulness', label: 'Helpfulness' },
    { key: 'ease_of_use', label: 'Ease of Use' },
  ],
  chat: [
    { key: 'helpfulness', label: 'Helpfulness' },
    { key: 'clarity', label: 'Clarity' },
    { key: 'speed', label: 'Response Speed' },
  ],
  dashboard: [
    { key: 'helpfulness', label: 'Helpfulness' },
    { key: 'clarity', label: 'Clarity' },
  ],
}

export default function FeedbackWidget({ feature, context, onClose }: FeedbackWidgetProps) {
  const [rating, setRating] = useState<'positive' | 'negative' | null>(null)
  const [showDetails, setShowDetails] = useState(false)
  const [sliders, setSliders] = useState<Record<string, number>>({})
  const [comment, setComment] = useState('')
  const [submitted, setSubmitted] = useState(false)
  const [submitting, setSubmitting] = useState(false)

  const sliderConfigs = SLIDER_CONFIGS[feature] || []

  const handleRatingClick = (value: 'positive' | 'negative') => {
    setRating(value)
    setShowDetails(true)
  }

  const handleSliderChange = (key: string, value: number) => {
    setSliders(prev => ({ ...prev, [key]: value }))
  }

  const handleSubmit = async () => {
    if (!rating) return

    setSubmitting(true)
    try {
      await axios.post(`${API_BASE}/feedback`, {
        feature,
        rating,
        sliders: Object.keys(sliders).length > 0 ? sliders : null,
        comment: comment.trim() || null,
        context,
      })
      setSubmitted(true)
    } catch (error) {
      console.error('Failed to submit feedback:', error)
    } finally {
      setSubmitting(false)
    }
  }

  const handleSkip = () => {
    handleSubmit()
  }

  if (submitted) {
    return (
      <div className="glass-card p-4 text-center">
        <div className="text-emerald-400 mb-2">✓ Thank you for your feedback!</div>
        <p className="text-surface-400 text-sm">Your input helps us improve.</p>
      </div>
    )
  }

  return (
    <div className="glass-card p-4">
      {/* Header */}
      <div className="flex items-center justify-between mb-3">
        <span className="text-surface-300 text-sm font-medium">Was this helpful?</span>
        {onClose && (
          <button onClick={onClose} className="text-surface-500 hover:text-surface-300">
            <X className="w-4 h-4" />
          </button>
        )}
      </div>

      {/* Thumbs buttons */}
      <div className="flex gap-3 mb-4">
        <button
          onClick={() => handleRatingClick('positive')}
          className={`flex items-center gap-2 px-4 py-2 rounded-lg border transition-all ${
            rating === 'positive'
              ? 'bg-emerald-500/20 border-emerald-500/50 text-emerald-400'
              : 'bg-surface-800/50 border-surface-700 text-surface-400 hover:border-surface-600'
          }`}
        >
          <ThumbsUp className="w-5 h-5" />
          <span>Yes</span>
        </button>
        <button
          onClick={() => handleRatingClick('negative')}
          className={`flex items-center gap-2 px-4 py-2 rounded-lg border transition-all ${
            rating === 'negative'
              ? 'bg-red-500/20 border-red-500/50 text-red-400'
              : 'bg-surface-800/50 border-surface-700 text-surface-400 hover:border-surface-600'
          }`}
        >
          <ThumbsDown className="w-5 h-5" />
          <span>No</span>
        </button>
      </div>

      {/* Expanded details section */}
      {rating && (
        <div className="border-t border-surface-700 pt-4">
          <button
            onClick={() => setShowDetails(!showDetails)}
            className="flex items-center gap-2 text-surface-400 text-sm mb-3 hover:text-surface-300"
          >
            {showDetails ? <ChevronUp className="w-4 h-4" /> : <ChevronDown className="w-4 h-4" />}
            <span>Rate specific aspects (optional)</span>
          </button>

          {showDetails && (
            <div className="space-y-4">
              {/* Sliders */}
              {sliderConfigs.map(({ key, label }) => (
                <div key={key}>
                  <div className="flex justify-between text-sm mb-1">
                    <span className="text-surface-400">{label}</span>
                    <span className="text-surface-300 font-mono">
                      {sliders[key] || 5}/10
                    </span>
                  </div>
                  <input
                    type="range"
                    min="1"
                    max="10"
                    value={sliders[key] || 5}
                    onChange={(e) => handleSliderChange(key, parseInt(e.target.value))}
                    className="w-full"
                  />
                </div>
              ))}

              {/* Comment */}
              <div>
                <label className="block text-surface-400 text-sm mb-2">
                  Any other comments? (optional)
                </label>
                <textarea
                  value={comment}
                  onChange={(e) => setComment(e.target.value)}
                  placeholder="Tell us more..."
                  className="w-full px-3 py-2 bg-surface-800/50 border border-surface-700 rounded-lg text-surface-100 placeholder-surface-500 text-sm resize-none focus:outline-none focus:border-primary-500/50"
                  rows={2}
                />
              </div>
            </div>
          )}

          {/* Submit buttons */}
          <div className="flex gap-2 mt-4">
            <button
              onClick={handleSubmit}
              disabled={submitting}
              className="btn-primary flex items-center gap-2 text-sm py-2"
            >
              {submitting ? (
                <div className="spinner w-4 h-4" />
              ) : (
                <Send className="w-4 h-4" />
              )}
              Submit
            </button>
            {!showDetails && (
              <button
                onClick={handleSkip}
                className="btn-secondary text-sm py-2"
              >
                Skip details
              </button>
            )}
          </div>
        </div>
      )}
    </div>
  )
}

