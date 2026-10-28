import { useState } from "react";
import ReferencePanel from "../components/ReferencePanel.jsx";
import VideoSequencePlayer from "../components/VideoSequencePlayer.jsx";
import { api } from "../api.js";
import { speak } from "../lib/speech.js";

const EXAMPLES = ["hello", "thank you", "please help me", "my mother is happy", "where is the bathroom"];

export default function TextToSign({ go }) {
  const [text, setText] = useState("");
  const [normalize, setNormalize] = useState(true);
  const [out, setOut] = useState(null);
  const [active, setActive] = useState({ entry: 0, letter: undefined });
  const [err, setErr] = useState("");
  const [loading, setLoading] = useState(false);

  const run = async (q = text, n = normalize) => {
    if (!q.trim()) return;
    setErr(""); setLoading(true);
    try {
      setOut(await api.textToSign(q, n));
      setActive({ entry: 0, letter: undefined });
    } catch (e) { setErr(e.message); } finally { setLoading(false); }
  };

  const entry = out?.entries?.[active.entry];
  const signId = entry ? (entry.fingerspelled ? entry.letters[active.letter ?? 0]?.sign_id : entry.sign_id) : null;
  const changed = (out?.entries || []).filter((e) => e.normalized_from);

  return (
    <section className="two">
      <div>
        <h2>Text → sign</h2>
        <p className="banner">{out?.note || "Word-by-word glossing, not ASL grammar: real ASL word order and facial grammar differ."}
          {" "}Unknown words are fingerspelled.</p>
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
        <label className="inline-check">
          <input type="checkbox" checked={normalize} onChange={(e) => { setNormalize(e.target.checked); run(text, e.target.checked); }} />
          {" "}Approximate ASL-style glosses (drop "the", "is", "to"; eating → EAT)
        </label>
        {err && <p className="error">{err}</p>}
        {out && (
          <>
            {out.dropped?.length > 0 && <p className="muted small">Not signed: {out.dropped.join(", ")}</p>}
            {changed.length > 0 && (
              <p className="muted small">Changed: {changed.map((e) => `${e.normalized_from} → ${e.gloss}`).join(", ")}</p>
            )}
            {out.missing?.length > 0 && <p className="muted small">Not available: {out.missing.join(", ")}</p>}
            <VideoSequencePlayer entries={out.entries} onActive={(e, l) => setActive({ entry: e, letter: l })} />
          </>
        )}
      </div>
      <div>
        {signId ? (
          <div className="card">
            <h3>{entry.fingerspelled ? `Letter ${entry.letters[active.letter ?? 0]?.letter} of ${entry.word.toUpperCase()}` : entry.gloss}</h3>
            <ReferencePanel signId={signId} compact />
            <div className="inline">
              <button className="btn secondary" onClick={() => go("practice", signId)}>Practise this sign</button>
              {out?.entries?.length > 1 && (
                <button className="btn secondary" onClick={() => go("practice", null, { sentence: text })}>Practise the sentence</button>
              )}
            </div>
          </div>
        ) : <p className="muted">Each sign's reference, license and attribution appear here.</p>}
      </div>
    </section>
  );
}
