const BASE = import.meta.env.VITE_API_URL || "/api";

async function req(path, body) {
  const r = await fetch(BASE + path, body && {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  if (!r.ok) {
    const e = await r.json().catch(() => ({}));
    throw new Error(typeof e.detail === "string" ? e.detail : "Request failed");
  }
  return r.json();
}

export const api = {
  status: () => req("/status"),
  signs: (q = "", category = "") => {
    const params = new URLSearchParams();
    if (q) params.set("q", q);
    if (category) params.set("category", category);
    const qs = params.toString();
    return req(`/signs${qs ? "?" + qs : ""}`);
  },
  sign: (id) => req(`/signs/${id}`),
  recognize: (payload) => {
    // payload can be { frames, image } or array of frames (backward compatibility)
    const body = Array.isArray(payload) ? { frames: payload } : payload;
    return req("/recognize", body);
  },
  score: (sign_id, frames) => req("/score", { sign_id, frames }),
  textToSign: (text) => req("/text-to-sign", { text }),
  addSign: (label, frames) => req("/signs", { label, frames }),
};
