import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

const backend = process.env.HANDSPEAK_BACKEND || "http://localhost:8000";

export default defineConfig({
  plugins: [react()],
  server: { proxy: { "/api": backend, "/media": backend } },
});
