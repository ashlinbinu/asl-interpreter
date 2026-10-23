import { useState } from "react";
import SkeletonPlayer from "../components/SkeletonPlayer.jsx";
import { api } from "../api.js";

const QUICK_EXAMPLES = ["hello", "thank you", "please", "yes", "cat", "ASL", "water"];

export default function TextToSign() {
  const [text, setText] = useState("");
  const [out, setOut] = useState(null);
  const [err, setErr] = useState("");
  const [loading, setLoading] = useState(false);

  const go = async (inputStr) => {
    const query = inputStr !== undefined ? inputStr : text;
    if (!query.trim()) return;
    setErr("");
    setLoading(true);
    try {
      const res = await api.textToSign(query);
      setOut(res);
    } catch (e) {
      setErr(e.message);
    } finally {
      setLoading(false);
    }
  };

  const speak = () => {
    if ("speechSynthesis" in window && text.trim()) {
      window.speechSynthesis.cancel();
      window.speechSynthesis.speak(new SpeechSynthesisUtterance(text));
    }
  };

  return (
    <section>
      <h2>Text to Sign Translation</h2>
      <p className="muted">
        Type any sentence, word, or name. Known phrases display full sign gestures; unlisted words
        are automatically fingerspelled letter-by-letter with the canonical ASL alphabet.
      </p>

      <div style={{ display: "flex", gap: "0.4rem", flexWrap: "wrap", marginBottom: "1rem" }}>
        <small className="muted" style={{ alignSelf: "center", marginRight: "0.2rem" }}>Quick try:</small>
        {QUICK_EXAMPLES.map((ex) => (
          <button
            key={ex}
            className="chip"
            style={{ fontSize: "0.85rem", padding: "0.25rem 0.75rem" }}
            onClick={() => {
              setText(ex);
              go(ex);
            }}
          >
            {ex}
          </button>
        ))}
      </div>

      <div className="inline">
        <input
          value={text}
          placeholder="Try: 'hello', 'thank you', 'peace', 'my name is alex'"
          onChange={(e) => setText(e.target.value)}
          onKeyDown={(e) => e.key === "Enter" && go()}
        />
        <button className="btn" onClick={() => go()} disabled={!text.trim() || loading}>
          {loading ? "Translating..." : "Translate"}
        </button>
        {out?.signs?.length > 0 && (
          <button className="btn secondary" onClick={speak} title="Read aloud">
            🔊
          </button>
        )}
      </div>

      {err && <p className="error">{err}</p>}

      {out && (
        <>
          <div className="grid">
            {out.signs.map((s, i) => (
              <figure key={i}>
                <SkeletonPlayer frames={s.frames} size={220} />
                <figcaption>
                  <span>{s.label}</span>
                  {s.is_fingerspelled && <span className="tag-spell">Letter</span>}
                </figcaption>
              </figure>
            ))}
          </div>

          {out.signs.length === 0 && (
            <p className="muted" style={{ marginTop: "1rem" }}>
              No signs or alphabet characters found for that text.
            </p>
          )}

          {out.missing?.length > 0 && (
            <p className="muted" style={{ marginTop: "1rem" }}>
              Unrecognized symbols: {out.missing.join(", ")}
            </p>
          )}
        </>
      )}
    </section>
  );
}
