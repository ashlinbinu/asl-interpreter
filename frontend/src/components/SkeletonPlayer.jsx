import { useEffect, useRef } from "react";
import { HAND_CONNECTIONS, POSE_UPPER } from "../lib/vision.js";

const FACE_DOTS = [0, 2, 5, 9, 10];

function bounds(frames) {
  let x0 = Infinity, y0 = Infinity, x1 = -Infinity, y1 = -Infinity;
  for (const f of frames) {
    const a = (f.w || 640) / (f.h || 480);
    const pts = [...(f.hands || []).flatMap((h) => h.lm), ...(f.pose ? FACE_DOTS.concat([11, 12, 13, 14]).map((j) => f.pose[j]) : [])];
    for (const p of pts) {
      x0 = Math.min(x0, p[0] * a); x1 = Math.max(x1, p[0] * a);
      y0 = Math.min(y0, p[1]); y1 = Math.max(y1, p[1]);
    }
  }
  return Number.isFinite(x0) ? { x0, y0, x1, y1 } : { x0: 0, y0: 0, x1: 1, y1: 1 };
}

function drawRaw(ctx, f, box, size, mirror) {
  const a = (f.w || 640) / (f.h || 480);
  const span = Math.max(box.x1 - box.x0, box.y1 - box.y0) * 1.15 || 1;
  const cx = (box.x0 + box.x1) / 2, cy = (box.y0 + box.y1) / 2;
  const P = (p) => {
    const x = ((p[0] * a - cx) / span) * size + size / 2;
    return [mirror ? size - x : x, ((p[1] - cy) / span) * size + size / 2];
  };
  if (f.pose) {
    ctx.strokeStyle = "rgba(255,255,255,0.5)";
    ctx.lineWidth = 3;
    for (const [i, j] of POSE_UPPER) {
      if (f.pose[i][3] < 0.5 || f.pose[j][3] < 0.5) continue;
      const A = P(f.pose[i]), B = P(f.pose[j]);
      ctx.beginPath(); ctx.moveTo(...A); ctx.lineTo(...B); ctx.stroke();
    }
    ctx.fillStyle = "#fff";
    for (const j of FACE_DOTS) if (f.pose[j][3] >= 0.5) { const A = P(f.pose[j]); ctx.beginPath(); ctx.arc(A[0], A[1], 3, 0, 7); ctx.fill(); }
  }
  for (const h of f.hands || []) {
    const color = h.handedness === "Right" ? "#3FE0FF" : "#FFE600";
    ctx.strokeStyle = color; ctx.shadowColor = color; ctx.shadowBlur = 8; ctx.lineWidth = 3;
    for (const { start, end } of HAND_CONNECTIONS) {
      const A = P(h.lm[start]), B = P(h.lm[end]);
      ctx.beginPath(); ctx.moveTo(...A); ctx.lineTo(...B); ctx.stroke();
    }
    ctx.shadowBlur = 0;
  }
}

function drawLegacy(ctx, f, size) {
  [0, 1].forEach((h) => {
    const v = f.slice(h * 63, h * 63 + 63);
    if (!v.some(Boolean)) return;
    const cx = size * (h ? 0.68 : 0.32), cy = size * 0.62, k = size * 0.3;
    const pt = (j) => [cx + v[j * 3] * k, cy + v[j * 3 + 1] * k];
    ctx.strokeStyle = "#8d8d80"; ctx.lineWidth = 3;
    HAND_CONNECTIONS.forEach(({ start, end }) => {
      ctx.beginPath(); ctx.moveTo(...pt(start)); ctx.lineTo(...pt(end)); ctx.stroke();
    });
  });
}

/** Loops real reference landmarks (RawFrames). Legacy 126-number frames render greyed (dev synthetic only). */
export default function SkeletonPlayer({ frames, size = 260, mirror = false, speed = 1, label }) {
  const ref = useRef();
  useEffect(() => {
    if (!frames?.length) return;
    const ctx = ref.current.getContext("2d");
    const raw = !Array.isArray(frames[0]) || typeof frames[0][0] !== "number";
    const reduce = window.matchMedia?.("(prefers-reduced-motion: reduce)").matches;
    const box = raw ? bounds(frames) : null;
    const t0 = raw ? frames[0].t : 0;
    const dur = raw ? Math.max(frames[frames.length - 1].t - t0, 1) : frames.length * 80;
    let raf, start = performance.now();
    const tick = (now) => {
      const el = ((now - start) * speed) % (dur + 600);   // short pause between loops
      let f;
      if (raw) {
        const target = t0 + Math.min(el, dur);
        f = frames.find((g) => g.t >= target) || frames[frames.length - 1];
      } else f = frames[Math.min(Math.floor(el / 80), frames.length - 1)];
      ctx.clearRect(0, 0, size, size);
      raw ? drawRaw(ctx, f, box, size, mirror) : drawLegacy(ctx, f, size);
      if (!reduce) raf = requestAnimationFrame(tick);
    };
    raf = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(raf);
  }, [frames, size, mirror, speed]);
  return <canvas ref={ref} width={size} height={size} className="skeleton" role="img" aria-label={label || "Reference skeleton animation"} />;
}
