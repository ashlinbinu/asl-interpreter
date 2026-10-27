// In-browser word model: the same forward pass as backend/app/word_model.py (WordNet), on FEATURE_SPEC sequences.
// The model file comes from GET /api/models/words (exported weights only; references and scoring stay server-side).
import { toSequence } from "./features.js";

const relu = (x) => (x > 0 ? x : 0);
const sigmoid = (x) => 1 / (1 + Math.exp(-x));

function conv(x, W, b, pad) {
  // x: T x Cin, W: k x Cin x Cout -> T x Cout ("same" padding)
  const T = x.length, k = W.length, cin = W[0].length, cout = b.length;
  const out = [];
  for (let t = 0; t < T; t++) {
    const row = b.slice();
    for (let j = 0; j < k; j++) {
      const s = t + j - pad;
      if (s < 0 || s >= T) continue;
      const xs = x[s], Wj = W[j];
      for (let i = 0; i < cin; i++) {
        const v = xs[i];
        if (v === 0) continue;
        const w = Wj[i];
        for (let o = 0; o < cout; o++) row[o] += v * w[o];
      }
    }
    out.push(row.map(relu));
  }
  return out;
}

function gru(x, [Wih, Whh, bih, bhh], reverse) {
  const H = Whh[0].length, T = x.length;
  const out = new Array(T);
  let h = new Array(H).fill(0);
  for (let step = 0; step < T; step++) {
    const t = reverse ? T - 1 - step : step;
    const xt = x[t];
    const gi = bih.slice(), gh = bhh.slice();
    for (let r = 0; r < 3 * H; r++) {
      let a = 0, c = 0;
      const wi = Wih[r], wh = Whh[r];
      for (let i = 0; i < xt.length; i++) a += wi[i] * xt[i];
      for (let i = 0; i < H; i++) c += wh[i] * h[i];
      gi[r] += a; gh[r] += c;
    }
    const nh = new Array(H);
    for (let i = 0; i < H; i++) {
      const r = sigmoid(gi[i] + gh[i]);
      const z = sigmoid(gi[H + i] + gh[H + i]);
      const n = Math.tanh(gi[2 * H + i] + r * gh[2 * H + i]);
      nh[i] = (1 - z) * n + z * h[i];
    }
    h = nh;
    out[t] = h;
  }
  return out;
}

export class WordModel {
  constructor(d) {
    this.version = d.version;
    this.classes = d.classes;
    this.temperature = d.temperature;
    this.rejectThreshold = d.reject_threshold;
    this.net = d.net;
  }

  /** seq: T x F raw features -> logits */
  logits(seq) {
    const { net } = this;
    let x = seq.map((row) => row.map((v, j) => (v - net.mean[j]) / net.std[j]));
    if (net.kind === "cnn") {
      for (const c of net.convs) x = conv(x, c.W, c.b, c.pad);
    } else {
      x = x.map((row) => net.proj_b.map((b, o) => relu(row.reduce((a, v, i) => a + v * net.proj_W[i][o], b))));
      const f = gru(x, net.fw, false), bw = gru(x, net.bw, true);
      x = f.map((row, t) => row.concat(bw[t]));
    }
    const C = x[0].length;
    const pooled = new Array(2 * C).fill(0);
    for (let o = 0; o < C; o++) {
      let s = 0, m = -Infinity;
      for (const row of x) { s += row[o]; if (row[o] > m) m = row[o]; }
      pooled[o] = s / x.length;
      pooled[C + o] = m;
    }
    return net.out_b.map((b, k) => pooled.reduce((a, v, i) => a + v * net.out_W[i][k], b));
  }

  /** calibrated class probabilities for a feature sequence */
  proba(seq) {
    const z = this.logits(seq).map((v) => v / this.temperature);
    const m = Math.max(...z);
    const e = z.map((v) => Math.exp(v - m));
    const s = e.reduce((a, v) => a + v, 0);
    return e.map((v) => v / s);
  }

  /** RawFrames -> probabilities (null if too few hand frames) */
  probaFrames(frames, leftHanded = false) {
    const seq = toSequence(frames, leftHanded);
    return seq ? this.proba(seq) : null;
  }
}

export async function loadWordModel(base = import.meta.env?.VITE_API_URL || "/api") {
  const r = await fetch(`${base}/models/words`);
  if (!r.ok) throw new Error("Word model not trained yet");
  return new WordModel(await r.json());
}
