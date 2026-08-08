import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// The dev server proxies /api to the FastAPI backend, so the browser sees a
// single origin and there is no CORS configuration anywhere in this project.
// In production the same is true by construction: `npm run build` emits
// web/dist, which api.py mounts at /.
export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    proxy: {
      "/api": { target: "http://127.0.0.1:8000", changeOrigin: true, ws: false },
    },
  },
  build: { outDir: "dist", sourcemap: true },
});
