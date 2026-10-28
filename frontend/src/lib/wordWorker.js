// Runs the in-browser word model off the main thread for useLiveWords (same code as on the main thread).
import { WordModel } from "./wordModel.js";

let model = null;

self.onmessage = ({ data }) => {
  if (data.type === "init") {
    model = new WordModel(data.model);
    return;
  }
  const t0 = performance.now();
  try {
    const p = model ? model.probaFrames(data.frames, data.leftHanded) : null;
    self.postMessage({ p, ms: performance.now() - t0 });
  } catch (e) {
    self.postMessage({ p: null, ms: 0, error: String(e?.message || e) });
  }
};
