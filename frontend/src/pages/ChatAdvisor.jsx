import { useState, useEffect, useRef } from 'react'
import { Send, Loader2, Key, MessageSquare, Trash2 } from 'lucide-react'
import axios from 'axios'

const API = axios.create({
  baseURL: '/api',
  headers: { 'Content-Type': 'application/json' },
})

const QUICK_PROMPTS = [
  { label: 'Gaming Laptop', text: 'I need a gaming laptop for playing AAA games' },
  { label: 'Work Laptop', text: 'I need a laptop for work and productivity' },
  { label: 'Creative Work', text: 'I need a laptop for video editing and design' },
  { label: 'Student Budget', text: 'I need an affordable laptop for school' },
]

async function fetchGreeting() {
  const { data } = await API.get('/chat/greeting')
  return data
}

async function postChatMessage(message, apiKey, history) {
  const { data } = await API.post('/chat', {
    message,
    api_key: apiKey,
    conversation_history: history,
  })
  return data
}

export default function ChatAdvisor() {
  const [apiKey, setApiKey] = useState('')
  const [apiKeySet, setApiKeySet] = useState(false)
  const [messages, setMessages] = useState([])
  const [input, setInput] = useState('')
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState(null)
  const messagesEndRef = useRef(null)

  useEffect(() => {
    if (apiKeySet && messages.length === 0) {
      loadGreeting()
    }
  }, [apiKeySet])

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [messages])

  const loadGreeting = async () => {
    try {
      const { greeting } = await fetchGreeting()
      setMessages([{ role: 'assistant', content: greeting }])
    } catch (err) {
      console.error(err)
    }
  }

  const handleSetApiKey = () => {
    if (apiKey.trim()) {
      setApiKeySet(true)
    }
  }

  const handleSend = async (text) => {
    const messageText = text || input.trim()
    if (!messageText || loading) return

    setInput('')
    setError(null)
    const userMessage = { role: 'user', content: messageText }
    const previousHistory = [...messages]
    setMessages((prev) => [...prev, userMessage])
    setLoading(true)

    try {
      const response = await postChatMessage(messageText, apiKey, previousHistory)
      setMessages(response.conversation_history)
    } catch (err) {
      const detail = err.response?.data?.detail || err.message || 'Failed to send message'
      setError(detail)
      console.error(err)
    } finally {
      setLoading(false)
    }
  }

  const handleClearChat = () => {
    setMessages([])
    loadGreeting()
  }

  const handleKeyPress = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault()
      handleSend()
    }
  }

  if (!apiKeySet) {
    return (
      <div className="max-w-xl mx-auto space-y-6">
        <div className="text-center">
          <MessageSquare className="h-16 w-16 mx-auto text-primary-500 mb-4" />
          <h1 className="text-3xl font-bold text-white">Chat Advisor</h1>
          <p className="text-surface-400 mt-2">
            Get personalized laptop recommendations from our AI advisor
          </p>
        </div>

        <div className="glass-card p-6">
          <h2 className="text-lg font-semibold text-white mb-4 flex items-center gap-2">
            <Key className="h-5 w-5 text-primary-400" />
            Enter Your OpenAI API Key
          </h2>
          <p className="text-sm text-surface-400 mb-4">
            This feature requires an OpenAI API key. Your key is only used for this session
            and is never stored.
          </p>
          <div className="space-y-4">
            <input
              type="password"
              value={apiKey}
              onChange={(e) => setApiKey(e.target.value)}
              placeholder="sk-..."
              className="input-field w-full"
            />
            <button
              onClick={handleSetApiKey}
              disabled={!apiKey.trim()}
              className="btn-primary w-full"
            >
              Start Chatting
            </button>
          </div>
          <p className="text-xs text-surface-500 mt-4">
            No key yet?{' '}
            <a
              href="https://platform.openai.com/api-keys"
              target="_blank"
              rel="noreferrer"
              className="text-primary-400 hover:underline"
            >
              Create one at platform.openai.com
            </a>
          </p>
        </div>
      </div>
    )
  }

  return (
    <div className="space-y-4">
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-bold text-white">Chat Advisor</h1>
          <p className="text-surface-400 text-sm">Ask anything about laptops or pricing</p>
        </div>
        <button onClick={handleClearChat} className="btn-secondary flex items-center gap-2">
          <Trash2 className="h-4 w-4" />
          Clear Chat
        </button>
      </div>

      <div className="glass-card p-4 h-[60vh] overflow-y-auto space-y-4">
        {messages.map((message, index) => (
          <div
            key={`${message.role}-${index}`}
            className={`flex ${message.role === 'user' ? 'justify-end' : 'justify-start'}`}
          >
            <div
              className={`max-w-[80%] rounded-2xl px-4 py-3 text-sm ${
                message.role === 'user'
                  ? 'bg-primary-500 text-white'
                  : 'bg-surface-800 text-surface-100'
              }`}
            >
              <div className="whitespace-pre-wrap">{message.content}</div>
            </div>
          </div>
        ))}

        {loading && (
          <div className="flex justify-start">
            <div className="bg-surface-800 rounded-2xl px-4 py-3">
              <Loader2 className="h-5 w-5 animate-spin text-surface-400" />
            </div>
          </div>
        )}

        {error && (
          <div className="text-center">
            <div className="inline-block bg-red-500/10 text-red-300 px-4 py-2 rounded-lg text-sm">
              {error}
            </div>
          </div>
        )}

        <div ref={messagesEndRef} />
      </div>

      {messages.length <= 1 && (
        <div className="flex flex-wrap gap-2">
          {QUICK_PROMPTS.map((prompt) => (
            <button
              key={prompt.label}
              onClick={() => handleSend(prompt.text)}
              className="px-3 py-2 bg-surface-800/70 border border-surface-700 rounded-lg text-sm text-surface-200 hover:bg-surface-700 transition-colors"
            >
              {prompt.label}
            </button>
          ))}
        </div>
      )}

      <div className="flex gap-2">
        <input
          type="text"
          value={input}
          onChange={(e) => setInput(e.target.value)}
          onKeyDown={handleKeyPress}
          placeholder="Tell me about your laptop needs..."
          className="input-field flex-1"
          disabled={loading}
        />
        <button onClick={() => handleSend()} disabled={!input.trim() || loading} className="btn-primary px-4">
          <Send className="h-5 w-5" />
        </button>
      </div>

      <p className="text-center text-xs text-surface-500">
        Powered by GPT-4 · Provide budget/use case details for best results.
      </p>
    </div>
  )
}
