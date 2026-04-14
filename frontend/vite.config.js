import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    proxy: {
      // Forward /api/* to Flask during development
      "/api": {
        target: "http://localhost:5000",
        changeOrigin: true,
      },
    },
  },
});
