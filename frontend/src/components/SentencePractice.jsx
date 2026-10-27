import { useEffect, useState } from "react";
import Camera from "./Camera.jsx";
import ReferencePanel from "./ReferencePanel.jsx";
import VideoSequencePlayer from "./VideoSequencePlayer.jsx";
import { api } from "../api.js";

const MOTION = new Set(["J", "Z"]);

/** Plan entries -> practice steps: one per sign, one per fingerspelled letter. */
export function sentenceSteps(entries) {
  return (entries || []).flatMap((e) => (e.fingerspelled
    ? e.letters.map((L) => ({ sign_id: L.sign_id, label: L.letter, word: e.word, kind: MOTION.has(L.letter) ? "dynamic" : "static" }))
    : [{ sign_id: e.sign_id, label: e.gloss, word: e.word, kind: e.kind }]));
}

/**
 * Sentence practice. "Word by word": each sign's reference is shown and scored in turn.
 * "Whole sentence": sign everything in one go; each automatically cut segment is scored against the next sign in
 * order (so a missed sign shifts the rest; a learning aid, not continuous recognition).
 */
export default function SentencePractice({ initialText = "" }) {
  const [text, setText] = useState(initialText);
  const [plan, setPlan] = useState(null);
  const [steps, setSteps] = useState([]);
  const [k, setK] = useState(0);
  const [results, setResults] = useState([]);
  const [mode, setMode] = useState("word");
  const [busy, setBusy] = useState(false);
  const [err, setErr] = useState("");

  const load = async (q = text) => {
    if (!q.trim()) return;
    setErr("");
    try {
      const p = await api.textToSign(q, true);
      setPlan(p);
      setSteps(sentenceSteps(p.entries));
      setK(0); setResults([]);
    } catch (e) { setErr(e.message); }
  };
  useEffect(() => { if (initialText) load(initialText); }, [initialText]);

  const step = steps[k];
  const done = steps.length > 0 && k >= steps.length;

  const onSegment = async ({ frames }) => {
    if (!step || busy) return;
    setBusy(true); setErr("");
    let r;
    try {
      r = await api.score(step.sign_id, frames, "sentence");
    } catch (e) {
      r = { score: null, hints: [e.message], feedback: "not scored" };
    } finally { setBusy(false); }
    setResults((rs) => [...rs, { ...step, ...r }]);
    setK((n) => n + 1);
  };

  const scored = results.filter((r) => r.score != null);
  const mean = scored.length ? Math.round(scored.reduce((a, r) => a + r.score, 0) / scored.length) : null;

  return (
    <div>
      <div className="inline">
        <input value={text} aria-label="Sentence to practise" placeholder="e.g. my mother is happy"
               onChange={(e) => setText(e.target.value)} onKeyDown={(e) => e.key === "Enter" && load()} />
        <button className="btn" onClick={() => load()} disabled={!text.trim()}>Start</button>
      </div>
      <div className="subnav slim" role="group" aria-label="Sentence mode">
        <button className={mode === "word" ? "on" : ""} onClick={() => setMode("word")}>Word by word</button>
        <button className={mode === "whole" ? "on" : ""} onClick={() => setMode("whole")}>Whole sentence</button>
      </div>
      {err && <p className="error">{err}</p>}
      {plan && <p className="muted small">{plan.note}{plan.dropped?.length ? ` Not signed: ${plan.dropped.join(", ")}.` : ""}</p>}
      {steps.length > 0 && (
        <div className="timeline" aria-label="Progress through the sentence">
          {steps.map((s, i) => {
            const r = results[i];
            const cls = r ? (r.score == null ? "" : r.score >= 80 ? "mastered" : r.score >= 50 ? "" : "low") : i === k ? "on" : "";
            return <span key={i} className={`chip small ${cls}`}>{s.label}{r?.score != null ? ` ${Math.round(r.score)}` : ""}</span>;
          })}
        </div>
      )}
      {step && (
        <div className="two">
          <div>
            {mode === "word" ? (
              <>
                <h3>Next: <span className="hl">{step.label}</span>{step.word && step.word.toUpperCase() !== step.label ? ` (from "${step.word}")` : ""}</h3>
                <ReferencePanel signId={step.sign_id} compact />
              </>
            ) : (
              <>
                <h3>Sign the whole sentence</h3>
                <VideoSequencePlayer entries={plan.entries} />
                <p className="muted small">Pause briefly between signs. Next expected: <b>{step.label}</b></p>
              </>
            )}
          </div>
          <div>
            <Camera mode="auto" kind={mode === "word" && step.kind === "static" ? "static" : "dynamic"}
                    onSegment={onSegment} disabled={busy} />
            {busy && <p className="note">Scoring…</p>}
            <button className="btn secondary" onClick={() => { setResults((rs) => [...rs, { ...step, score: null, hints: ["skipped"] }]); setK((n) => n + 1); }}>
              Skip {step.label}
            </button>
          </div>
        </div>
      )}
      {done && (
        <div className="score" aria-live="polite">
          <div className="score-head"><strong>{mean ?? "–"}</strong><span>sentence average ({scored.length} of {steps.length} signs scored)</span></div>
          <ul className="hints">
            {results.map((r, i) => (
              <li key={i}><b>{r.label}</b>: {r.score != null ? `${Math.round(r.score)}` : "not scored"}{r.hints?.[0] ? ` · ${r.hints[0]}` : ""}</li>
            ))}
          </ul>
          <button className="btn" onClick={() => { setK(0); setResults([]); }}>Try again</button>
        </div>
      )}
    </div>
  );
}
