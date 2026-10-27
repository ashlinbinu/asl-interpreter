import { useState } from "react";
import ReferencePanel from "../components/ReferencePanel.jsx";
import { api } from "../api.js";
import { speak } from "../lib/speech.js";

const EXAMPLES = ["hello", "thank you", "please help", "my name is Sam", "water"];

export default function TextToSign({ go }) {
  const [text, setText] = useState("");
  const [out, setOut] = useState(null);
  const [active, setActive] = useState(0);
  const [letter, setLetter] = useState(0);
  const [err, setErr] = useState("");
  const [loading, setLoading] = useState(false);

  const run = async (q = text) => {
    if (!q.trim()) return;
    setErr(""); setLoading(true);
    try {
      setOut(await api.textToSign(q));
      setActive(0); setLetter(0);
    } catch (e) { setErr(e.message); } finally { setLoading(false); }
  };

  const entry = out?.entries?.[active];
  const signId = entry ? (entry.fingerspelled ? entry.letters[letter]?.sign_id : entry.sign_id) : null;

  return (
    <section>
      <h2>Text → sign</h2>
      <p className="banner">Word-by-word glossing, not ASL grammar: real ASL word order and facial grammar differ.
        Unknown words are fingerspelled. The stitched reference-video player arrives in phase P4.</p>
      <div className="examples">
        <small className="muted">Try:</small>
        {EXAMPLES.map((ex) => <button key={ex} className="chip small" onClick={() => { setText(ex); run(ex); }}>{ex}</button>)}
      </div>
      <div className="inline">
        <input value={text} aria-label="Text to sign" placeholder="Type a word or sentence"
               onChange={(e) => setText(e.target.value)} onKeyDown={(e) => e.key === "Enter" && run()} />
        <button className="btn" onClick={() => run()} disabled={!text.trim() || loading}>{loading ? "…" : "Show signs"}</button>
        <button className="btn secondary" onClick={() => speak(text)} aria-label="Read aloud" disabled={!text.trim()}>🔊</button>
      </div>
      {err && <p className="error">{err}</p>}
      {out && (
        <>
          <div className="timeline" role="tablist" aria-label="Words">
            {out.entries.map((e, i) => (
              <button key={i} role="tab" aria-selected={i === active} className={`chip ${i === active ? "on" : ""}`}
                      onClick={() => { setActive(i); setLetter(0); }}>
                {e.fingerspelled ? `${e.word} (spelled)` : e.gloss}
              </button>
            ))}
          </div>
          {entry?.fingerspelled && (
            <div className="timeline small" role="tablist" aria-label="Letters">
              {entry.letters.map((L, j) => (
                <button key={j} className={`chip letter ${j === letter ? "on" : ""}`} onClick={() => setLetter(j)}>{L.letter}</button>
              ))}
            </div>
          )}
          {signId && (
            <div className="card">
              <ReferencePanel signId={signId} />
              <button className="btn secondary" onClick={() => go("practice", signId)}>Practise this sign</button>
            </div>
          )}
          {out.missing?.length > 0 && <p className="muted">Not available: {out.missing.join(", ")}</p>}
        </>
      )}
    </section>
  );
}
