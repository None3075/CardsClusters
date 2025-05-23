import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import tailwindcss from '@tailwindcss/vite'


// https://vite.dev/config/
export default defineConfig({
  plugins: [react(), tailwindcss()],
  server: {
    host: '0.0.0.0',  // Esto expone el servidor a todas las IPs externas
    port: 3000,        // Asegúrate de usar el puerto que mapeaste en el Docker Compose
  },
  preview: {
    host: '0.0.0.0',
    port: 3000,
  },
  define: {
    // Make process.env values available at build time
    'process.env': process.env
  }
})
