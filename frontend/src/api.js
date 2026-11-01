import { learner } from "./lib/identity.js";

const BASE = import.meta.env.VITE_API_URL || "/api";
// Media (reference videos) is served next to the API: /api -> /media
export const MEDIA_BASE = BASE.replace(/\/api\/?$/, "");
// The API is on another host (a free hosting plan), which stops it when nobody has used it for a while.
export const REMOTE_BACKEND = /^https?:\/\//i.test(BASE);

// While a stopped backend boots, its host's proxy answers instead: a network error (its error page carries no CORS
// headers) or a 502/503/504 that isn't the API's own JSON. Until a remote backend has answered once, those are
// retried; a same-origin backend (local, or behind a tunnel) is never retried.
const WAKE_DELAYS_S = [2, 3, 5, 5, 10, 10, 15, 15, 15, 15, 15, 15, 15, 15];   // ~2.5 min in all
let awake = !REMOTE_BACKEND;

const fromProxy = (r) => [502, 503, 504].includes(r.status) && !(r.headers.get("content-type") || "").includes("json");

async function send(url, opts) {
  for (let i = 0; ; i++) {
    const r = await fetch(url, opts).catch((e) => {
      if (awake || i >= WAKE_DELAYS_S.length) throw e;
      return null;
    });
    if (r && (awake || !fromProxy(r) || i >= WAKE_DELAYS_S.length)) {
      awake = awake || !fromProxy(r);
      return r;
    }
    await new Promise((ok) => setTimeout(ok, WAKE_DELAYS_S[i] * 1000));
  }
}

async function req(path, body, method) {
  const opts = body instanceof FormData
    ? { method: "POST", body }
    : body !== undefined
      ? { method: method || "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) }
      : method ? { method } : undefined;
  const r = await send(BASE + path, opts);
  if (!r.ok) {
    const e = await r.json().catch(() => ({}));
    const detail = typeof e.detail === "string" ? e.detail : Array.isArray(e.detail) ? e.detail.map((d) => d.msg).join("; ") : "";
    const err = new Error(detail || `Request failed (${r.status})`);
    err.status = r.status;
    throw err;
  }
  return r.json();
}

const qs = (o) => {
  const p = new URLSearchParams();
  Object.entries(o).forEach(([k, v]) => v !== undefined && v !== null && v !== "" && p.set(k, v));
  const s = p.toString();
  return s ? "?" + s : "";
};

export const mediaUrl = (u) => (u && u.startsWith("/media") ? MEDIA_BASE + u : u);

export const api = {
  status: () => req("/status"),
  signs: (q = "", category = "", extra = {}) => req(`/signs${qs({ q, category, ...extra })}`),
  sign: (id) => req(`/signs/${id}`),
  recognize: (payload) => req("/recognize", { left_handed: learner().leftHanded, ...payload }),
  score: (sign_id, frames, mode = "practice") =>
    req("/score", { sign_id, frames, mode, left_handed: learner().leftHanded, learner_id: learner().id }),
  textToSign: (text, normalize = false) => req("/text-to-sign", { text, normalize }),
  addSign: (label, frames) => req("/signs", { label, frames, learner_id: learner().id }),
  deleteSign: (id) => req(`/signs/${id}`, undefined, "DELETE"),
  recordSign: (id, frames, videoBlob) => {
    const fd = new FormData();
    fd.append("meta", JSON.stringify({ learner_id: learner().id, left_handed: learner().leftHanded }));
    fd.append("landmarks", new Blob([JSON.stringify(frames)], { type: "application/json" }), "landmarks.json");
    if (videoBlob) fd.append("video", videoBlob, "take.webm");
    return req(`/signs/${id}/recording`, fd);
  },
  attempt: (sign_id, score, extra = {}) => req("/attempts", { learner_id: learner().id, sign_id, score, ...extra }),
  progress: () => req(`/progress${qs({ learner_id: learner().id })}`),
  recordingSummary: (signer_id) => req(`/recordings/summary${qs({ signer_id })}`),
  uploadRecording: (meta, frames, videoBlob) => {
    const fd = new FormData();
    fd.append("meta", JSON.stringify(meta));
    fd.append("landmarks", new Blob([JSON.stringify(frames)], { type: "application/json" }), "landmarks.json");
    if (videoBlob) fd.append("video", videoBlob, "clip.webm");
    return req("/recordings", fd);
  },
};
