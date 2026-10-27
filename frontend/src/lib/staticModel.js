// In-browser alphabet classifier: runs the browser.json exported by backend/app/train_static.py.
// Supports the two shipped kinds: "mlp" (scaler + dense layers) and "rf" (tree arrays).

function softmax(z) {
  const m = Math.max(...z);
  const e = z.map((v) => Math.exp(v - m));
  const s = e.reduce((a, b) => a + b, 0);
  return e.map((v) => v / s);
}

export function calibrate(p, temperature) {
  return softmax(p.map((v) => Math.log(Math.min(Math.max(v, 1e-12), 1)) / temperature));
}

function mlpProba(m, x) {
  let h = x.map((v, i) => (v - m.scaler.mean[i]) / (m.scaler.scale[i] || 1));
  m.layers.forEach((L, li) => {
    const out = L.b.slice();
    for (let i = 0; i < h.length; i++) {
      const hi = h[i];
      if (hi === 0) continue;
      const row = L.W[i];
      for (let j = 0; j < out.length; j++) out[j] += hi * row[j];
    }
    h = li < m.layers.length - 1 ? out.map((v) => (m.activation === "relu" ? Math.max(0, v) : Math.tanh(v))) : out;
  });
  if (h.length === 1) { const p1 = 1 / (1 + Math.exp(-h[0])); return [1 - p1, p1]; }
  return softmax(h);
}

function rfProba(m, x) {
  const k = m.classes.length;
  const acc = new Array(k).fill(0);
  const xf = x.map((v) => Math.fround(v)); // sklearn compares float32 features
  for (const t of m.trees) {
    let node = 0;
    while (t.l[node] !== -1) node = xf[t.f[node]] <= t.t[node] ? t.l[node] : t.r[node];
    const leaf = t.leaf[node];
    for (const c in leaf) acc[+c] += leaf[c];
  }
  const s = acc.reduce((a, b) => a + b, 0) || 1;
  return acc.map((v) => v / s);
}

export class StaticModel {
  constructor(json) {
    this.m = json;
    this.classes = json.classes;
    this.version = json.version;
    this.rejectThreshold = json.reject_threshold ?? 0;
  }

  /** Calibrated class probabilities for one 93-dim static vector. */
  proba(vec) {
    const raw = this.m.kind === "mlp" ? mlpProba(this.m, vec) : rfProba(this.m, vec);
    return calibrate(raw, this.m.temperature ?? 1);
  }

  /** Mean probabilities over several vectors -> sorted [{label, confidence}] */
  rank(vectors, topK = 5) {
    if (!vectors.length) return [];
    const acc = new Array(this.classes.length).fill(0);
    for (const v of vectors) this.proba(v).forEach((p, i) => { acc[i] += p / vectors.length; });
    return acc.map((p, i) => ({ label: this.classes[i], confidence: p }))
      .sort((a, b) => b.confidence - a.confidence).slice(0, topK);
  }
}

let cached;
export async function loadStaticModel(base = import.meta.env?.VITE_API_URL || "/api") {
  cached ??= fetch(`${base}/models/alphabet`).then((r) => {
    if (!r.ok) throw new Error("Alphabet model not trained yet");
    return r.json();
  }).then((j) => new StaticModel(j)).catch((e) => { cached = undefined; throw e; });
  return cached;
}
