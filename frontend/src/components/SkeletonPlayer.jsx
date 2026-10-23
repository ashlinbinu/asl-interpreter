import { useEffect, useRef } from "react";
import { HandLandmarker } from "@mediapipe/tasks-vision";

/** Loops a stored sign (32 frames x 126 numbers) as two animated neon hands. */
export default function SkeletonPlayer({ frames, size = 260 }) {
  const ref = useRef();
  useEffect(() => {
    if (!frames?.length) return;
    const ctx = ref.current.getContext("2d");
    let i = 0;
    const id = setInterval(() => {
      ctx.clearRect(0, 0, size, size);
      const f = frames[i++ % frames.length];
      [0, 1].forEach((h) => {
        const v = f.slice(h * 63, h * 63 + 63);
        if (!v.some(Boolean)) return;
        const cx = size * (h ? 0.68 : 0.32), cy = size * 0.62, k = size * 0.3;
        const pt = (j) => [cx + v[j * 3] * k, cy + v[j * 3 + 1] * k];
        ctx.strokeStyle = "#FFE600";
        ctx.lineWidth = 3;
        ctx.shadowColor = "#FFE600";
        ctx.shadowBlur = 8;
        HandLandmarker.HAND_CONNECTIONS.forEach(({ start, end }) => {
          const a = pt(start), b = pt(end);
          ctx.beginPath(); ctx.moveTo(...a); ctx.lineTo(...b); ctx.stroke();
        });
      });
    }, 80);
    return () => clearInterval(id);
  }, [frames, size]);
  return <canvas ref={ref} width={size} height={size} className="skeleton" />;
}
