import { defineConfig } from "vite"
import react from "@vitejs/plugin-react"
import tsconfigPaths from "vite-tsconfig-paths"

export default defineConfig({
  plugins: [react(), tsconfigPaths()],
  server: {
    proxy: {
      '/predict': 'http://127.0.0.1:8000',
      '/generate-report': 'http://127.0.0.1:8000',
      '/get-latest-report': 'http://127.0.0.1:8000',
      '/get_latest_results': 'http://127.0.0.1:8000',
      '/chat_with_report': 'http://127.0.0.1:8000',
      '/api': 'http://127.0.0.1:8000',
    },
  },
})
