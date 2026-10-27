import { useCallback, useEffect, useRef, useState } from "react";
import { learner } from "./identity.js";
import { loadWordModel } from "./wordModel.js";

const WINDOW_MS = 1800;     // sliding window: about one sign
const EVERY_MS = 300;       // live update period

/**
 * In-browser word recogniser over a sliding window of Camera.onFrame frames (same features and weights as the
 * server). Returns { ready, error, top, scores, onFrame, latencyMs }. Live values are a rough guide; the score
 * after a finished attempt (server: classifier + DTW components) is the one that counts.
 */
export function useLiveWords(enabled = true) {
  const [model, setModel] = useState(null);
  const [error, setError] = useState("");
  const [top, setTop] = useState([]);
  const [scores, setScores] = useState({});
  const [latencyMs, setLatency] = useState(0);
  const buf = useRef([]), last = useRef(0);

  useEffect(() => {
    if (!enabled) return;
    loadWordModel().then(setModel).catch((e) => setError(e.message));
  }, [enabled]);

  const onFrame = useCallback((frame) => {
    buf.current.push(frame);
    while (buf.current.length && frame.t - buf.current[0].t > WINDOW_MS) buf.current.shift();
    if (!model || frame.t - last.current < EVERY_MS) return;
    last.current = frame.t;
    const t0 = performance.now();
    const p = model.probaFrames(buf.current, learner().leftHanded);
    if (!p) { setTop([]); setScores({}); return; }
    const ranked = model.classes.map((label, i) => ({ label, confidence: p[i] })).sort((a, b) => b.confidence - a.confidence);
    setTop(ranked.slice(0, 5));
    setScores(Object.fromEntries(ranked.map((m) => [m.label, m.confidence])));
    setLatency(performance.now() - t0);
  }, [model]);

  return { ready: !!model, model, error, top, scores, onFrame, latencyMs };
}
