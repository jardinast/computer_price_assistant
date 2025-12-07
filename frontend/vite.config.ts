import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// Read API target from environment (set in Dockerfile for Docker, defaults to localhost)
const apiTarget = process.env.VITE_API_TARGET || 'http://localhost:8000';
console.log('[Vite Config] Proxy target:', apiTarget);

export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    host: '0.0.0.0',
    proxy: {
      '/api': {
        target: apiTarget,
        changeOrigin: true,
      },
    },
  },
})
