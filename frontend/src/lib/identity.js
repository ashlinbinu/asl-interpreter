// Anonymous local identity + preferences (no accounts). Storage can be unavailable; fall back to memory.
const KEY = "handspeak.learner";
const memory = {};

function read() {
  try { return JSON.parse(localStorage.getItem(KEY) || "null") || memory.v; } catch { return memory.v; }
}

function write(v) {
  memory.v = v;
  try { localStorage.setItem(KEY, JSON.stringify(v)); } catch { /* private mode */ }
}

function newId() {
  const rnd = crypto?.randomUUID ? crypto.randomUUID().replace(/-/g, "") : Math.random().toString(36).slice(2) + Date.now().toString(36);
  return "l_" + rnd.slice(0, 20);
}

export function learner() {
  let v = read();
  if (!v?.id) { v = { id: newId(), leftHanded: false, highContrast: false }; write(v); }
  return v;
}

export function updateLearner(patch) {
  const v = { ...learner(), ...patch };
  write(v);
  window.dispatchEvent(new CustomEvent("handspeak:learner", { detail: v }));
  return v;
}
