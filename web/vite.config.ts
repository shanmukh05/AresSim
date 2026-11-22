import tailwindcss from "@tailwindcss/vite";
import react from "@vitejs/plugin-react";
import { defineConfig } from "vite";

const isDeployed = process.env.VITE_DEPLOYED === "true" || process.env.VITE_DEPLOYED === "1";
const base = process.env.VITE_BASE_PATH || (isDeployed ? "/aressim/" : "/");

export default defineConfig({
  base,
  plugins: [react(), tailwindcss()],
  server: {
    proxy: {
      "/api": "http://127.0.0.1:8000",
      "/aressim/api": {
        target: "http://127.0.0.1:8000",
        rewrite: (path) => path.replace(/^\/aressim/, ""),
      },
    },
  },
  resolve: {
    alias: {
      "react-reconciler/constants": "react-reconciler/constants.js",
    },
  },
});
