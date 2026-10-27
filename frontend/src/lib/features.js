// Landmark -> feature pipeline, implementing docs/FEATURE_SPEC.md (v1.0).
// Mirrors backend/app/features.py; backend/tests/test_feature_parity.py keeps them equal.
// Pure module (no DOM) so it runs in the browser and in Node.

export const SPEC_VERSION = "1.0";
export const T = 32;
export const F = 270;
export const HAND = 128;
export const GLOBAL = 256;
export const STATIC_DIM = 93;
const GAP_MAX = 3;
const MIN_CUTOFF = 1.5, BETA = 5.0, D_CUTOFF = 1.0;
const VIS_MIN = 0.5, WRIST_VIS_MIN = 0.3;
const POSE_N = 17;
const MIN_HAND_FRAMES = 4;

const POSE_SWAP = [[1, 4], [2, 5], [3, 6], [7, 8], [9, 10], [11, 12], [13, 14], [15, 16]];
const CHAINS = [[0, 1, 2, 3, 4], [0, 5, 6, 7, 8], [0, 9, 10, 11, 12], [0, 13, 14, 15, 16], [0, 17, 18, 19, 20]];
const FINGER_DIRS = [[2, 4], [5, 8], [9, 12], [13, 16], [17, 20]];
const TIP_PAIRS = [[4, 8], [4, 12], [4, 16], [4, 20], [8, 12]];
const PALM = [0, 5, 9, 13, 17];
export const ANCHORS = ["forehead", "eye_d", "eye_nd", "nose", "mouth", "chin",
  "cheek_d", "cheek_nd", "ear_d", "ear_nd", "chest"];
const FACE_POINTS = [0, 2, 5, 9, 10];
const FLAG_IDX = [0, HAND, GLOBAL, GLOBAL + 1];

// ------------------------------------------------------------ small vector helpers
const sub = (a, b) => a.map((v, i) => v - b[i]);
const dot = (a, b) => a.reduce((s, v, i) => s + v * b[i], 0);
const norm = (a) => Math.sqrt(dot(a, a));
const hypot2 = (x, y) => Math.hypot(x, y);
const cross = (a, b) => [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];
const unit = (v) => { const n = norm(v); return n > 1e-9 ? v.map((x) => x / n) : v.map(() => 0); };
const angle = (u, v) => {
  const d = dot(u, v) / Math.max(norm(u) * norm(v), 1e-12);
  return Math.acos(Math.max(-1, Math.min(1, d))) / Math.PI;
};
const median = (arr) => {
  const s = [...arr].sort((a, b) => a - b);
  const m = s.length >> 1;
  return s.length % 2 ? s[m] : (s[m - 1] + s[m]) / 2;
};

function toBody(body, x, y) {
  const c = Math.cos(body.roll), s = Math.sin(body.roll);
  const dx = x - body.ox, dy = y - body.oy;
  return [(dx * c + dy * s) / body.scale, (-dx * s + dy * c) / body.scale];
}

function rot(v, roll) {
  const c = Math.cos(roll), s = Math.sin(roll);
  return [v[0] * c + v[1] * s, -v[0] * s + v[1] * c, v[2]];
}

// ------------------------------------------------------------ parsing / canonicalisation
export function normalizeFrame(f, i = 0) {
  if (f && !Array.isArray(f)) return f;
  return { t: (i * 1000) / 30, w: 640, h: 480, hands: (f || []).map((lm) => ({ handedness: null, lm })), pose: null };
}

function poseSides(hs, p) {
  const wl = p[15], wr = p[16];
  const d = hs.map((h) => [hypot2(h[0][0] - wl[0], h[0][1] - wl[1]), hypot2(h[0][0] - wr[0], h[0][1] - wr[1])]);
  if (hs.length === 1) return [d[0][0] < d[0][1] ? "L" : "R"];
  const straight = d[0][1] + d[1][0];
  const crossed = d[0][0] + d[1][1];
  return straight <= crossed ? ["R", "L"] : ["L", "R"];
}

function labelSides(labels, xs) {
  const sides = labels.map((l) => ({ Left: "R", Right: "L" })[l] ?? null);
  if (sides.length === 2 && (sides.includes(null) || sides[0] === sides[1])) return xs[0] < xs[1] ? ["R", "L"] : ["L", "R"];
  if (sides.length === 1 && sides[0] === null) return ["R"];
  return sides;
}

export function canonicalize(frames, leftHanded = false) {
  const n = frames.length;
  const hands = [[], []], hmask = [[], []], pose = [], pmask = [], t = [];
  for (let i = 0; i < n; i++) {
    const f = normalizeFrame(frames[i], i);
    t.push((f.t ?? (i * 1000) / 30) / 1000);
    const w = +f.w || 640, h = +f.h || 480;
    const a = h ? w / h : 4 / 3;
    let p = null;
    if (f.pose && f.pose.length >= POSE_N) {
      p = f.pose.slice(0, POSE_N).map((q) => [q[0] * a, q[1], q.length > 3 ? q[3] : 1]);
    }
    const hs = [], labels = [];
    for (const hd of (f.hands || []).slice(0, 2)) {
      const lm = Array.isArray(hd) ? hd : hd.lm;
      hs.push(lm.map((q) => [q[0] * a, q[1], q[2] * a]));
      labels.push(Array.isArray(hd) ? null : hd.handedness ?? null);
    }
    let sides;
    if (p && p[15][2] >= WRIST_VIS_MIN && p[16][2] >= WRIST_VIS_MIN && hs.length) sides = poseSides(hs, p);
    else sides = labelSides(labels, hs.map((hh) => hh[0][0]));
    if (leftHanded) {
      for (const hh of hs) for (const q of hh) q[0] = a - q[0];
      sides = sides.map((s) => (s === "R" ? "L" : "R"));
      if (p) {
        for (const q of p) q[0] = a - q[0];
        for (const [x, y] of POSE_SWAP) [p[x], p[y]] = [p[y], p[x]];
      }
    }
    const zeroHand = () => Array.from({ length: 21 }, () => [0, 0, 0]);
    hands[0].push(zeroHand()); hands[1].push(zeroHand());
    hmask[0].push(false); hmask[1].push(false);
    hs.forEach((hh, j) => { const k = sides[j] === "R" ? 0 : 1; hands[k][i] = hh; hmask[k][i] = true; });
    pose.push(p ?? Array.from({ length: POSE_N }, () => [0, 0, 0]));
    pmask.push(!!p);
  }
  return { hands, hmask, pose, pmask, t };
}

// ------------------------------------------------------------ gap fill + smoothing
// values: array of flat Float64 arrays (one per frame); mask: bool array. Mutates both.
function fillGaps(values, mask, t) {
  const n = mask.length;
  let i = 0;
  while (i < n) {
    if (mask[i]) { i++; continue; }
    let j = i;
    while (j < n && !mask[j]) j++;
    if (i > 0 && j < n && j - i <= GAP_MAX) {
      const a = i - 1, b = j, span = t[b] - t[a];
      for (let k = i; k < j; k++) {
        const alpha = span > 0 ? (t[k] - t[a]) / span : (k - a) / (b - a);
        values[k] = values[a].map((v, d) => (1 - alpha) * v + alpha * values[b][d]);
        mask[k] = true;
      }
    }
    i = j;
  }
}

function oneEuro(values, mask, t) {
  const out = values.map((v) => v.slice());
  let xHat = null, dxHat = null;
  for (let i = 0; i < mask.length; i++) {
    if (!mask[i]) continue;
    const x = values[i];
    if (i === 0 || !mask[i - 1]) {
      xHat = x.slice();
      dxHat = x.map(() => 0);
    } else {
      let dt = t[i] - t[i - 1];
      if (dt <= 0) dt = 1 / 30;
      const ad = 1 / (1 + 1 / (2 * Math.PI * D_CUTOFF) / dt);
      const nx = new Array(x.length);
      for (let d = 0; d < x.length; d++) {
        const dx = (x[d] - xHat[d]) / dt;
        dxHat[d] = ad * dx + (1 - ad) * dxHat[d];
        const fc = MIN_CUTOFF + BETA * Math.abs(dxHat[d]);
        const tau = 1 / (2 * Math.PI * fc);
        const al = 1 / (1 + tau / dt);
        nx[d] = al * x[d] + (1 - al) * xHat[d];
      }
      xHat = nx;
    }
    out[i] = xHat.slice();
  }
  return out;
}

// ------------------------------------------------------------ body frame + anchors
export function bodyFrame(pose, pmask) {
  const ox = [], oy = [], sc = [], rl = [];
  for (let i = 0; i < pose.length; i++) {
    const p = pose[i];
    if (!pmask[i] || p[11][2] < VIS_MIN || p[12][2] < VIS_MIN) continue;
    const L = p[11], R = p[12];
    ox.push((L[0] + R[0]) / 2); oy.push((L[1] + R[1]) / 2);
    sc.push(hypot2(L[0] - R[0], L[1] - R[1]));
    rl.push(Math.atan2(L[1] - R[1], L[0] - R[0]));
  }
  if (!ox.length) return null;
  const b = { ox: median(ox), oy: median(oy), scale: median(sc), roll: median(rl) };
  return b.scale >= 1e-6 ? b : null;
}

function faceAnchors(p, body) {
  const P = p.map((q) => [q[0], q[1]]);
  const eyes = [(P[2][0] + P[5][0]) / 2, (P[2][1] + P[5][1]) / 2];
  const mouth = [(P[9][0] + P[10][0]) / 2, (P[9][1] + P[10][1]) / 2];
  const up = [eyes[0] - mouth[0], eyes[1] - mouth[1]];
  const pts = [
    [eyes[0] + 0.8 * up[0], eyes[1] + 0.8 * up[1]], P[5], P[2], P[0], mouth,
    [mouth[0] - 0.7 * up[0], mouth[1] - 0.7 * up[1]],
    [(P[10][0] + P[8][0]) / 2, (P[10][1] + P[8][1]) / 2], [(P[9][0] + P[7][0]) / 2, (P[9][1] + P[7][1]) / 2],
    P[8], P[7],
  ];
  return pts.map((q) => toBody(body, q[0], q[1]));
}

// ------------------------------------------------------------ per-frame features
function handBlock(p, chir, body, anchors, chest, out, o) {
  out[o] = 1;
  const roll = body ? body.roll : 0;
  const r = p.map((q) => rot(sub(q, p[0]), roll));
  const sh = Math.max(norm(r[9]), 1e-6);
  for (let j = 0; j < 21; j++) for (let d = 0; d < 3; d++) out[o + 1 + j * 3 + d] = r[j][d] / sh;
  let k = o + 64;
  for (const ch of CHAINS) for (const j of [1, 2, 3]) out[k++] = angle(sub(r[ch[j - 1]], r[ch[j]]), sub(r[ch[j + 1]], r[ch[j]]));
  const dirs = FINGER_DIRS.map(([a, b]) => sub(r[b], r[a]));
  for (let j = 0; j < 4; j++) out[o + 79 + j] = angle(dirs[j], dirs[j + 1]);
  TIP_PAIRS.forEach(([a, b], j) => { out[o + 83 + j] = norm(sub(r[a], r[b])) / sh; });
  const nrm = unit(cross(r[5], r[17]));
  for (let d = 0; d < 3; d++) out[o + 88 + d] = chir * nrm[d];
  const dir = unit(r[9]);
  for (let d = 0; d < 3; d++) out[o + 91 + d] = dir[d];
  if (body) {
    const palm = [0, 1].map((d) => PALM.reduce((s, j) => s + p[j][d], 0) / PALM.length);
    const locs = [toBody(body, p[0][0], p[0][1]), toBody(body, p[8][0], p[8][1]), toBody(body, palm[0], palm[1])];
    locs.forEach((l, j) => { out[o + 94 + j * 2] = l[0]; out[o + 95 + j * 2] = l[1]; });
    [locs[1], locs[2]].forEach((q, qi) => {
      const base = o + 100 + qi * 11;
      if (anchors) anchors.forEach((an, j) => { out[base + j] = hypot2(an[0] - q[0], an[1] - q[1]); });
      out[base + 10] = hypot2(chest[0] - q[0], chest[1] - q[1]);
    });
  }
}

/** RawFrames -> { feats: Float64Array[N] of length F, t, handMask, body } */
export function extract(frames, leftHanded = false) {
  const { hands, hmask, pose, pmask, t } = canonicalize(frames, leftHanded);
  const n = t.length;
  for (let k = 0; k < 2; k++) {
    const flat = hands[k].map((h) => h.flat());
    fillGaps(flat, hmask[k], t);
    const sm = oneEuro(flat, hmask[k], t);
    hands[k] = sm.map((v) => Array.from({ length: 21 }, (_, j) => [v[j * 3], v[j * 3 + 1], v[j * 3 + 2]]));
  }
  const pflat = pose.map((p) => p.flat());
  fillGaps(pflat, pmask, t);
  const xy = pflat.map((v) => { const o = []; for (let j = 0; j < POSE_N; j++) o.push(v[j * 3], v[j * 3 + 1]); return o; });
  const sxy = oneEuro(xy, pmask, t);
  const P = pflat.map((v, i) => Array.from({ length: POSE_N }, (_, j) => [sxy[i][j * 2], sxy[i][j * 2 + 1], v[j * 3 + 2]]));

  const body = bodyFrame(P, pmask);
  const chest = [0, 0.6];
  const feats = [];
  for (let i = 0; i < n; i++) {
    const out = new Float64Array(F);
    let anchors = null, face = false;
    if (body && pmask[i] && FACE_POINTS.every((j) => P[i][j][2] >= VIS_MIN)) {
      anchors = faceAnchors(P[i], body);
      face = true;
    }
    for (let k = 0; k < 2; k++) if (hmask[k][i]) handBlock(hands[k][i], k === 0 ? 1 : -1, body, anchors, chest, out, k * HAND);
    const g = GLOBAL;
    const bodyI = body && pmask[i] && P[i][11][2] >= VIS_MIN && P[i][12][2] >= VIS_MIN;
    out[g + 1] = face ? 1 : 0;
    if (bodyI) {
      out[g] = 1;
      [[12, 2], [11, 4]].forEach(([j, off]) => { const b = toBody(body, P[i][j][0], P[i][j][1]); out[g + off] = b[0]; out[g + off + 1] = b[1]; });
      [[14, 6], [13, 8]].forEach(([j, off]) => {
        if (P[i][j][2] >= VIS_MIN) { const b = toBody(body, P[i][j][0], P[i][j][1]); out[g + off] = b[0]; out[g + off + 1] = b[1]; }
      });
    }
    if (body && hmask[0][i] && hmask[1][i]) {
      const dx = out[94] - out[HAND + 94], dy = out[95] - out[HAND + 95];
      out[g + 10] = dx; out[g + 11] = dy; out[g + 12] = hypot2(dx, dy);
      out[g + 13] = hypot2(out[96] - out[HAND + 98], out[97] - out[HAND + 99]);
    }
    feats.push(out);
  }
  if (body) {
    for (let i = 1; i < n; i++) {
      let dt = t[i] - t[i - 1];
      if (dt <= 0) dt = 1 / 30;
      for (let k = 0; k < 2; k++) {
        if (!(hmask[k][i] && hmask[k][i - 1])) continue;
        const o = k * HAND;
        for (let d = 0; d < 6; d++) feats[i][o + 122 + d] = (feats[i][o + 94 + d] - feats[i - 1][o + 94 + d]) / dt;
      }
    }
  }
  const handMask = t.map((_, i) => [hmask[0][i], hmask[1][i]]);
  return { feats, t, handMask, body };
}

// ------------------------------------------------------------ resampling + static vectors
export function resample(feats, t, handMask, length = T) {
  const idx = [];
  handMask.forEach((m, i) => { if (m[0] || m[1]) idx.push(i); });
  if (idx.length < MIN_HAND_FRAMES) return null;
  const a = idx[0], b = idx[idx.length - 1];
  const seg = feats.slice(a, b + 1), ts = t.slice(a, b + 1);
  const n = seg.length, span = ts[n - 1] - ts[0];
  const u = ts.map((x, i) => (span > 0 ? (x - ts[0]) / span : i / Math.max(n - 1, 1)));
  const out = [];
  for (let k = 0; k < length; k++) {
    const uk = k / (length - 1);
    if (n === 1) { out.push(Float64Array.from(seg[0])); continue; }
    let i = -1;
    for (let j = 0; j < n; j++) if (u[j] <= uk) i = j;   // last u_j <= uk  (== searchsorted right - 1)
    i = Math.min(Math.max(i, 0), n - 2);
    const du = u[i + 1] - u[i];
    let alpha = du > 0 ? (uk - u[i]) / du : 0;
    alpha = Math.min(Math.max(alpha, 0), 1);
    const A = seg[i], B = seg[i + 1];
    if (FLAG_IDX.every((j) => A[j] === B[j])) out.push(A.map((v, d) => (1 - alpha) * v + alpha * B[d]));
    else out.push(Float64Array.from(alpha < 0.5 ? A : B));
  }
  return out;
}

export function toSequence(frames, leftHanded = false) {
  if (!frames?.length) return null;
  const ex = extract(frames, leftHanded);
  return resample(ex.feats, ex.t, ex.handMask);
}

export function staticVector(row) {
  let v;
  if (row[0] === 1) v = Array.from(row.slice(1, 1 + STATIC_DIM));
  else if (row[HAND] === 1) {
    v = Array.from(row.slice(HAND + 1, HAND + 1 + STATIC_DIM));
    for (let j = 0; j < 63; j += 3) v[j] = -v[j];
    v[87] = -v[87];
    v[90] = -v[90];
  } else return null;
  return v;
}

/** RawFrames -> { vectors: number[M][93], index: number[M] } for frames containing a hand. */
export function staticVectors(frames, leftHanded = false) {
  if (!frames?.length) return { vectors: [], index: [] };
  const { feats } = extract(frames, leftHanded);
  const vectors = [], index = [];
  feats.forEach((row, i) => { const v = staticVector(row); if (v) { vectors.push(v); index.push(i); } });
  return { vectors, index };
}
