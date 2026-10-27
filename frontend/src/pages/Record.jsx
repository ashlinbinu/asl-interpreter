import { useEffect, useRef, useState } from "react";
import Camera from "../components/Camera.jsx";
import ReferencePanel from "../components/ReferencePanel.jsx";
import { api } from "../api.js";
import { learner, updateLearner } from "../lib/identity.js";

const SIGNER_KEY = "handspeak.signer";
const STATIC_LETTERS = "ABCDEFGHIKLMNOPQRSTUVWXY".split("");

function check(frames, isLetter) {
  const n = frames.length || 1;
  const hand = frames.filter((f) => f.hands.length).length / n;
  const face = frames.filter((f) => f.pose && [0, 2, 5].every((j) => f.pose[j][3] >= 0.5)).length / n;
  const problems = [];
  if (hand < (isLetter ? 0.8 : 0.6)) problems.push(`hands visible in only ${Math.round(hand * 100)}% of frames`);
  if (!isLetter && face < 0.5) problems.push(`face visible in only ${Math.round(face * 100)}% of frames`);
  return { hand, face, problems };
}

export default function Record() {
  const cam = useRef();
  const [signer, setSigner] = useState(() => { try { return localStorage.getItem(SIGNER_KEY) || ""; } catch { return ""; } });
  const [vocab, setVocab] = useState([]);
  const [chosen, setChosen] = useState(STATIC_LETTERS);
  const [reps, setReps] = useState(5);
  const [counts, setCounts] = useState({});
  const [queue, setQueue] = useState([]);
  const [phase, setPhase] = useState("setup");   // setup | prompt | countdown | recording | review | done
  const [count, setCount] = useState(0);
  const [last, setLast] = useState(null);
  const [msg, setMsg] = useState("");
  const [leftHanded, setLeftHanded] = useState(learner().leftHanded);
  const current = queue[0];

  useEffect(() => { api.signs().then(setVocab).catch((e) => setMsg(e.message)); }, []);
  const byLabel = Object.fromEntries(vocab.map((s) => [s.label, s]));
  const validSigner = /^[A-Za-z0-9_-]{2,64}$/.test(signer);
  const refreshCounts = () => validSigner && api.recordingSummary(signer).then((r) => setCounts(r.counts)).catch(() => {});
  useEffect(() => { refreshCounts(); }, [signer]);

  const start = () => {
    try { localStorage.setItem(SIGNER_KEY, signer); } catch { /* ignore */ }
    updateLearner({ leftHanded });
    const q = [];
    for (const label of chosen) {
      const s = byLabel[label];
      if (!s) continue;
      const have = counts[s.id] || 0;
      for (let r = have; r < reps; r++) q.push({ sign: s, rep: r + 1 });
    }
    setQueue(q);
    setPhase(q.length ? "prompt" : "done");
  };

  const record = async () => {
    setPhase("countdown");
    for (let c = 3; c > 0; c--) { setCount(c); await new Promise((r) => setTimeout(r, 700)); }
    setPhase("recording");
    await cam.current.startTimed(current.sign.category === "alphabet" ? 1.5 : 3);
  };

  const onSegment = ({ frames, video }) => {
    const isLetter = current.sign.category === "alphabet";
    setLast({ frames, video, q: check(frames, isLetter) });
    setPhase("review");
  };

  const accept = async () => {
    setMsg("");
    try {
      await api.uploadRecording({ sign_id: current.sign.id, signer_id: signer, rep: current.rep, left_handed: leftHanded },
        last.frames, last.video);
      setQueue((q) => q.slice(1));
      setLast(null);
      refreshCounts();
      setPhase(queue.length > 1 ? "prompt" : "done");
    } catch (e) { setMsg(e.message); }
  };

  useEffect(() => {
    const onKey = (e) => {
      if (["INPUT", "SELECT", "TEXTAREA"].includes(e.target?.tagName)) return;
      if (e.code === "Space" && phase === "prompt") { e.preventDefault(); record(); }
      else if (e.code === "Space" && phase === "review" && !last?.q.problems.length) { e.preventDefault(); accept(); }
      else if (e.key.toLowerCase() === "r" && phase === "review") setPhase("prompt");
      else if (e.key === "Escape") setPhase("setup");
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  });

  const toggle = (label) => setChosen((c) => (c.includes(label) ? c.filter((x) => x !== label) : [...c, label]));
  const words = vocab.filter((s) => s.category !== "alphabet");

  return (
    <section className="two">
      <div>
        <h2>Record a dataset</h2>
        <p className="muted">Your own recordings are the most valuable training and reference data. They are saved on this
          computer only (backend store folder). Record only people who have agreed.</p>
        {phase === "setup" ? (
          <div className="form">
            <label>Signer id (e.g. <code>me_01</code>; use a different id per person)
              <input value={signer} onChange={(e) => setSigner(e.target.value.trim())} placeholder="me_01" />
            </label>
            <label className="inline-check"><input type="checkbox" checked={leftHanded} onChange={(e) => setLeftHanded(e.target.checked)} /> This signer is left-handed</label>
            <label>Repetitions per sign <input type="number" min={1} max={30} value={reps} onChange={(e) => setReps(+e.target.value || 1)} /></label>
            <p className="muted small">Letters ({STATIC_LETTERS.length} static; J and Z need the word recorder)</p>
            <div className="list">
              {STATIC_LETTERS.map((L) => (
                <button key={L} className={`chip letter ${chosen.includes(L) ? "on" : ""}`} onClick={() => toggle(L)}
                        aria-pressed={chosen.includes(L)}>{L}<small className="count">{counts[byLabel[L]?.id] || 0}</small></button>
              ))}
            </div>
            {words.length > 0 && (<>
              <p className="muted small">Words</p>
              <div className="list">
                {words.map((s) => (
                  <button key={s.id} className={`chip ${chosen.includes(s.label) ? "on" : ""}`} onClick={() => toggle(s.label)}
                          aria-pressed={chosen.includes(s.label)}>{s.label}<small className="count">{counts[s.id] || 0}</small></button>
                ))}
              </div>
            </>)}
            <button className="btn" disabled={!validSigner || !chosen.length} onClick={start}>Start session</button>
            {!validSigner && <p className="muted small">Signer id: 2–64 letters, digits, - or _.</p>}
          </div>
        ) : phase === "done" ? (
          <div><p className="note">Session complete.</p><button className="btn" onClick={() => setPhase("setup")}>Back</button>
            <p className="muted small">Next: <code>python -m scripts.train_alphabet</code> (see README).</p></div>
        ) : current && (
          <div className="prompt-card">
            <p className="muted small">{queue.length} take(s) left · rep {current.rep}/{reps}</p>
            <h1 className="prompt-label">{current.sign.label}</h1>
            <ReferencePanel signId={current.sign.id} compact />
          </div>
        )}
        {msg && <p className="error">{msg}</p>}
      </div>

      <div>
        <Camera ref={cam} mode="timed" kind={current?.sign.category === "alphabet" ? "static" : "dynamic"}
                recordVideo onSegment={onSegment} />
        {phase === "prompt" && <button className="btn" onClick={record}>Record (Space)</button>}
        {phase === "countdown" && <p className="countdown" aria-live="assertive">{count}</p>}
        {phase === "review" && last && (
          <div className="card" aria-live="polite">
            <p>Hands {Math.round(last.q.hand * 100)}% · face {Math.round(last.q.face * 100)}% · {last.frames.length} frames</p>
            {last.q.problems.length > 0 && <p className="warn">Low quality: {last.q.problems.join("; ")}.</p>}
            <div className="inline">
              <button className="btn" onClick={accept}>{last.q.problems.length ? "Keep anyway" : "Save (Space)"}</button>
              <button className="btn secondary" onClick={() => setPhase("prompt")}>Redo (R)</button>
            </div>
          </div>
        )}
      </div>
    </section>
  );
}
