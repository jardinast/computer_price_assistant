const STORAGE_KEY = 'similarOffers:lastPredictionConfig'

export function saveSimilarConfig(config) {
  if (typeof window === 'undefined') return

  try {
    window.localStorage.setItem(STORAGE_KEY, JSON.stringify(config))
  } catch (err) {
    console.warn('Failed to persist similar config', err)
  }
}

export function loadSimilarConfig() {
  if (typeof window === 'undefined') return null

  try {
    const raw = window.localStorage.getItem(STORAGE_KEY)
    return raw ? JSON.parse(raw) : null
  } catch (err) {
    console.warn('Failed to load similar config', err)
    return null
  }
}
