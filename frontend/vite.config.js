import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

const backend = process.env.HANDSPEAK_BACKEND || "http://127.0.0.1:8000";

// 127.0.0.1, not "localhost": on Windows Vite would otherwise listen on IPv6 ::1 only, and IPv4 clients are refused.
export default defineConfig({
  plugins: [react()],
  server: { host: "127.0.0.1", port: 5173, strictPort: true, proxy: { "/api": backend, "/media": backend } },
  // `npm run preview` (same host and proxy as above, port 4173) behind a Cloudflare quick tunnel: accept the
  // tunnel's *.trycloudflare.com Host header (docs/DEPLOYMENT.md, route A).
  preview: { allowedHosts: [".trycloudflare.com"] },
});
