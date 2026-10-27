// Automatic sign segmentation from hand motion/visibility (pure; runs in the browser and Node).
//  dynamic: start when the fastest wrist moves faster than `vStart` for `startFrames`, stop after `stopFrames`
//           slow (or hand-less) frames; a short pre-roll is kept so the start of the sign is not lost.
//  static:  emit when a hand has been held still for `holdMs` (fingerspelling); re-arm once the hand moves or leaves.
// Speeds are in body units per second (shoulder width, or 2.5 x hand size without pose).

export const DEFAULTS = {
  vStart: 0.9, vStop: 0.35, startFrames: 3, stopFrames: 12, preRoll: 8,
  minMs: 400, maxMs: 4000, cooldownMs: 700, holdMs: 450, stillV: 0.45,
};

function scaleOf(f) {
  const a = (f.w || 640) / (f.h || 480);
  const p = f.pose;
  if (p && p[11][3] >= 0.5 && p[12][3] >= 0.5) return Math.hypot((p[11][0] - p[12][0]) * a, p[11][1] - p[12][1]);
  const h = f.hands?.[0]?.lm;
  return h ? 2.5 * Math.hypot((h[9][0] - h[0][0]) * a, h[9][1] - h[0][1]) : 0;
}

/** Max wrist speed between two RawFrames (hands matched by nearest wrist). */
export function wristSpeed(prev, cur) {
  if (!prev?.hands?.length || !cur?.hands?.length) return 0;
  const a = (cur.w || 640) / (cur.h || 480);
  const dt = Math.max((cur.t - prev.t) / 1000, 1 / 120);
  const s = scaleOf(cur) || 1;
  let best = 0;
  for (const h of cur.hands) {
    let d = Infinity;
    for (const g of prev.hands) d = Math.min(d, Math.hypot((h.lm[0][0] - g.lm[0][0]) * a, h.lm[0][1] - g.lm[0][1]));
    if (d < Infinity) best = Math.max(best, d / s / dt);
  }
  return best;
}

export class Segmenter {
  constructor(kind = "dynamic", opts = {}) {
    this.kind = kind;
    this.o = { ...DEFAULTS, ...opts };
    this.reset();
  }

  reset() {
    this.state = "idle";
    this.buf = [];       // recent frames (pre-roll / still window)
    this.seg = [];
    this.fast = 0; this.slow = 0; this.still = 0;
    this.prev = null; this.until = 0; this.armed = true;
  }

  /** Feed one frame; returns a finished segment (array of frames) or null. */
  push(f) {
    const o = this.o;
    const v = wristSpeed(this.prev, f);
    const hasHand = f.hands?.length > 0;
    this.prev = f;
    this.speed = v;
    this.buf.push(f);
    if (this.buf.length > Math.max(o.preRoll, 30)) this.buf.shift();
    if (f.t < this.until) return null;

    if (this.kind === "static") {
      // After a submission, wait until the hand moves or leaves before submitting again.
      if (!this.armed) {
        if (!hasHand || v > 2 * o.stillV) this.armed = true;
        return null;
      }
      this.still = hasHand && v < o.stillV ? this.still + 1 : 0;
      const window = this.buf.filter((g) => g.t >= f.t - o.holdMs);
      if (window.length >= 4 && this.still >= window.length && window.every((g) => g.hands?.length) && f.t - window[0].t >= o.holdMs * 0.9) {
        this.until = f.t + o.cooldownMs;
        this.still = 0;
        this.armed = false;
        return window;
      }
      return null;
    }

    if (this.state === "idle") {
      this.fast = hasHand && v > o.vStart ? this.fast + 1 : 0;
      if (this.fast >= o.startFrames) {
        this.state = "active";
        this.seg = this.buf.slice(-o.preRoll - o.startFrames);
        this.slow = 0;
      }
      return null;
    }
    this.seg.push(f);
    this.slow = !hasHand || v < o.vStop ? this.slow + 1 : 0;
    const dur = f.t - this.seg[0].t;
    if (this.slow >= o.stopFrames || dur >= o.maxMs) {
      const out = this.slow >= o.stopFrames ? this.seg.slice(0, this.seg.length - Math.floor(o.stopFrames / 2)) : this.seg;
      this.state = "idle";
      this.seg = [];
      this.fast = 0;
      this.until = f.t + o.cooldownMs;
      return out.length && out[out.length - 1].t - out[0].t >= o.minMs ? out : null;
    }
    return null;
  }
}
