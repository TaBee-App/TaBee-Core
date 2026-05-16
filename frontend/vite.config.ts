import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

export default defineConfig({
  plugins: [react()],
  optimizeDeps: {
    include: ["react", "react-dom/client", "react-router-dom", "lucide-react"],
    exclude: ["@coderline/alphatab"]
  },
  server: {
    proxy: {
      "/api": "http://127.0.0.1:8080",
      "/uploads": "http://127.0.0.1:8080"
    }
  }
});
