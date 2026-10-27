import { useCallback, useEffect, useRef, useState } from "react";
import { staticVectors } from "./features.js";
import { loadStaticModel } from "./staticModel.js";
import { learner } from "./identity.js";

const WINDOW = 8;           // frames averaged per live prediction
const EVERY_MS = 150;       // live update period

/**
 * In-browser fingerspelling recogniser fed by Camera.onFrame.
 * Returns { ready, error, top, onFrame, latencyMs } where top = [{label, confidence}] (calibrated).
 */
export function useLiveLetters(enabled = true) {
  const [model, setModel] = useState(null);
  const [error, setError] = useState("");
  const [top, setTop] = useState([]);
  const [scores, setScores] = useState({});
  const [latencyMs, setLatency] = useState(0);
  const buf = useRef([]), last = useRef(0);

  useEffect(() => {
    if (!enabled) return;
    loadStaticModel().then(setModel).catch((e) => setError(e.message));
  }, [enabled]);

  const onFrame = useCallback((frame) => {
    buf.current.push(frame);
    if (buf.current.length > WINDOW) buf.current.shift();
    if (!model || frame.t - last.current < EVERY_MS) return;
    last.current = frame.t;
    const t0 = performance.now();
    const { vectors } = staticVectors(buf.current, learner().leftHanded);
    const ranked = vectors.length >= 3 ? model.rank(vectors, model.classes.length) : [];
    setTop(ranked.slice(0, 5));
    setScores(Object.fromEntries(ranked.map((m) => [m.label, m.confidence])));
    setLatency(performance.now() - t0);
  }, [model]);

  return { ready: !!model, model, error, top, scores, onFrame, latencyMs };
}
