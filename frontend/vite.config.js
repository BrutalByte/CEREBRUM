import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

export default defineConfig({
  plugins: [react()],
  base: '/app/',
  server: {
    proxy: {
      '/v1': 'http://localhost:8200',
      '/ws': { target: 'ws://localhost:8200', ws: true },
    },
  },
})
