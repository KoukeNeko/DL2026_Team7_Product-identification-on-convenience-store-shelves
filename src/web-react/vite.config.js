import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// base './' → assets load with relative paths when served from FastAPI StaticFiles at /.
// dev proxy forwards /recognize to the FastAPI backend during `vite dev`.
export default defineConfig({
  plugins: [react()],
  base: './',
  server: {
    proxy: { '/recognize': 'http://127.0.0.1:8000' },
  },
  build: { outDir: 'dist', emptyOutDir: true },
})
