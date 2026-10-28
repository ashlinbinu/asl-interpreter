import { useCallback, useEffect, useRef, useState } from "react";
import { learner } from "./identity.js";
import { loadWordModel } from "./wordModel.js";

const WINDOW_MS = 1800;     // sliding window: about one sign
const EVERY_MS = 300;       // live update period
const MIN_HAND_FRAMES = 4;  // as in features.js: fewer frames with a hand give no prediction

/**
 * In-browser word recogniser over a sliding window of Camera.onFrame frames (same features and weights as the
 * server). The model runs in a Web Worker so it doesn't hold up the camera loop (on the main thread if workers are
 * unavailable), and windows without enough hand frames are skipped. Returns { ready, error, top, scores, onFrame,
 * latencyMs }. Live values are a rough guide; the score after a finished attempt (server) is the one that counts.
 */
export function useLiveWords(enabled = true) {
  const [model, setModel] = useState(null);
  const [error, setError] = useState("");
  const [top, setTop] = useState([]);
  const [scores, setScores] = useState({});
  const [latencyMs, setLatency] = useState(0);
  const buf = useRef([]), last = useRef(0), worker = useRef(null), busy = useRef(false), shown = useRef(false);

  const show = useCallback((m, p, ms) => {
    if (!p) {
      if (shown.current) { setTop([]); setScores({}); shown.current = false; }
      return;
    }
    const ranked = m.classes.map((label, i) => ({ label, confidence: p[i] })).sort((a, b) => b.confidence - a.confidence);
    setTop(ranked.slice(0, 5));
    setScores(Object.fromEntries(ranked.map((r) => [r.label, r.confidence])));
    setLatency(ms);
    shown.current = true;
  }, []);

  useEffect(() => {
    if (!enabled) return undefined;
    let alive = true;
    loadWordModel().then((m) => {
      if (!alive) return;
      try {
        const w = new Worker(new URL("./wordWorker.js", import.meta.url), { type: "module" });
        w.onmessage = ({ data }) => {
          busy.current = false;
          if (data.error) setError(data.error); else show(m, data.p, data.ms);
        };
        w.onerror = () => { busy.current = false; w.terminate(); worker.current = null; };   // main thread from now on
        w.postMessage({ type: "init", model: m.source });
        worker.current = w;
      } catch {
        worker.current = null;
      }
      setModel(m);
    }).catch((e) => setError(e.message));
    return () => { alive = false; worker.current?.terminate(); worker.current = null; busy.current = false; };
  }, [enabled, show]);

  const onFrame = useCallback((frame) => {
    buf.current.push(frame);
    while (buf.current.length && frame.t - buf.current[0].t > WINDOW_MS) buf.current.shift();
    if (!model || frame.t - last.current < EVERY_MS) return;
    last.current = frame.t;
    if (buf.current.filter((f) => f.hands?.length).length < MIN_HAND_FRAMES) { show(model, null, 0); return; }
    const leftHanded = learner().leftHanded;
    if (worker.current) {
      if (busy.current) return;                     // the previous window is still being classified
      busy.current = true;
      worker.current.postMessage({ type: "proba", frames: buf.current, leftHanded });
      return;
    }
    const t0 = performance.now();
    show(model, model.probaFrames(buf.current, leftHanded), performance.now() - t0);
  }, [model, show]);

  return { ready: !!model, model, error, top, scores, onFrame, latencyMs };
}
