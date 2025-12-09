import { Routes, Route, NavLink, useLocation } from 'react-router-dom'
import { motion, AnimatePresence } from 'framer-motion'
import {
  BarChart3,
  Calculator,
  Search,
  Laptop,
  TrendingUp,
  Cpu,
  MessageSquare,
} from 'lucide-react'

// Pages
import Dashboard from './pages/Dashboard'
import Predictor from './pages/Predictor'
import SimilarOffers from './pages/SimilarOffers'
import ChatAdvisor from './pages/ChatAdvisor'
import Admin from './pages/Admin'

function App() {
  const location = useLocation()

  const navItems = [
    { path: '/', label: 'Overview', icon: BarChart3 },
    { path: '/predict', label: 'Predict', icon: Calculator },
    { path: '/similar', label: 'Similar', icon: Search },
    { path: '/chat', label: 'Chat', icon: MessageSquare },
  ]

  return (
    <div className="min-h-screen flex flex-col">
      {/* Header */}
      <header className="sticky top-0 z-50 glass-card rounded-none border-x-0 border-t-0">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex items-center justify-between h-16">
            {/* Logo */}
            <NavLink to="/" className="flex items-center gap-3 group">
              <div className="w-10 h-10 rounded-xl bg-gradient-to-br from-primary-500 to-accent-500 flex items-center justify-center group-hover:scale-105 transition-transform">
                <Laptop className="w-5 h-5 text-white" />
              </div>
              <div>
                <h1 className="font-display font-bold text-lg gradient-text">PriceWise</h1>
                <p className="text-xs text-surface-500">Computer Price Predictor</p>
              </div>
            </NavLink>

            {/* Navigation */}
            <nav className="flex items-center gap-1">
              {navItems.map((item) => (
                <NavLink
                  key={item.path}
                  to={item.path}
                  className={({ isActive }) =>
                    `relative px-4 py-2 rounded-lg flex items-center gap-2 transition-all duration-200 ${
                      isActive
                        ? 'text-white bg-surface-800/80'
                        : 'text-surface-400 hover:text-surface-200 hover:bg-surface-800/40'
                    }`
                  }
                >
                  {({ isActive }) => (
                    <>
                      <item.icon className={`w-4 h-4 ${isActive ? 'text-primary-400' : ''}`} />
                      <span className="font-medium">{item.label}</span>
                      {isActive && (
                        <motion.div
                          layoutId="nav-indicator"
                          className="absolute -bottom-px left-2 right-2 h-0.5 bg-gradient-to-r from-primary-500 to-accent-500 rounded-full"
                        />
                      )}
                    </>
                  )}
                </NavLink>
              ))}
            </nav>

            {/* Right side */}
            <div className="flex items-center gap-4">
              <div className="hidden md:flex items-center gap-2 text-xs text-surface-500">
                <div className="w-2 h-2 rounded-full bg-emerald-500 animate-pulse" />
                <span>API Connected</span>
              </div>
            </div>
          </div>
        </div>
      </header>

      {/* Main Content */}
      <main className="flex-1">
        <AnimatePresence mode="wait">
          <motion.div
            key={location.pathname}
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -20 }}
            transition={{ duration: 0.2 }}
          >
            <Routes>
              <Route path="/" element={<Dashboard />} />
              <Route path="/predict" element={<Predictor />} />
              <Route path="/similar" element={<SimilarOffers />} />
              <Route path="/chat" element={<ChatAdvisor />} />
              <Route path="/admin" element={<Admin />} />
            </Routes>
          </motion.div>
        </AnimatePresence>
      </main>

      {/* Footer */}
      <footer className="py-6 border-t border-surface-800">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex items-center justify-between text-sm text-surface-500">
            <p>DAI Assignment 2 - Computer Price Prediction</p>
            <div className="flex items-center gap-4">
              <span className="flex items-center gap-1.5">
                <Cpu className="w-4 h-4" />
                ML-Powered
              </span>
              <span className="flex items-center gap-1.5">
                <TrendingUp className="w-4 h-4" />
                Real-time Analytics
              </span>
            </div>
          </div>
        </div>
      </footer>
    </div>
  )
}

export default App



