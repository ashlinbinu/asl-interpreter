import { useEffect, useState } from "react";
import Camera from "../components/Camera.jsx";
import ReferencePanel from "../components/ReferencePanel.jsx";
import SentencePractice from "../components/SentencePractice.jsx";
import SkeletonPlayer from "../components/SkeletonPlayer.jsx";
import { api } from "../api.js";
import { useLiveLetters } from "../lib/useLiveLetters.js";
import { useLiveWords } from "../lib/useLiveWords.js";
import { speak } from "../lib/speech.js";

const CATS = [["alphabet", "Alphabet"], ["word", "Words"], ["user", "My signs"], ["sentence", "Sentence"]];

function Meter({ label, value }) {
  const pct = Math.round((value ?? 0) * 100);
  return (
    <div className="meter" role="meter" aria-label={label} aria-valuemin={0} aria-valuemax={100} aria-valuenow={pct}>
      <span>{label}</span>
      <div className="bar"><i style={{ width: `${pct}%`, background: pct >= 80 ? "var(--green)" : undefined }} /></div>
      <b>{pct}</b>
    </div>
  );
}

export default function Practice({ initialSignId, initialSentence }) {
  const [category, setCategory] = useState(initialSentence ? "sentence" : "alphabet");
  const [attempt, setAttempt] = useState(null);
  const [showAttempt, setShowAttempt] = useState(false);
  const [list, setList] = useState([]);
  const [sel, setSel] = useState(null);
  const [res, setRes] = useState(null);
  const [err, setErr] = useState("");
  const [busy, setBusy] = useState(false);
  const [progress, setProgress] = useState({});
  const [capture, setCapture] = useState("auto");
  const [wordsReady, setWordsReady] = useState(false);
  const isStaticLetter = sel?.category === "alphabet" && sel?.kind === "static";
  const isMotionLetter = sel?.category === "alphabet" && sel?.kind === "dynamic";
  const isWord = sel?.category === "word";
  const isOwnSign = sel?.category === "user";
  const letters = useLiveLetters(isStaticLetter);
  const words = useLiveWords(isWord && wordsReady);
  const live = isStaticLetter ? letters : words;

  const loadProgress = () => api.progress().then((p) => setProgress(Object.fromEntries(p.signs.map((s) => [s.sign_id, s])))).catch(() => {});
  useEffect(() => { loadProgress(); }, []);
  useEffect(() => { api.status().then((s) => setWordsReady(!!s.models?.words?.ready)).catch(() => {}); }, []);

  useEffect(() => {
    const onKey = (e) => {
      if (["INPUT", "TEXTAREA", "SELECT"].includes(e.target?.tagName)) return;
      if (e.key.toLowerCase() === "r" && attempt) setShowAttempt((v) => !v);
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [attempt]);

  useEffect(() => {
    if (category === "sentence") return;
    api.signs("", category).then((data) => {
      setList(data);
      if (!sel || sel.category !== category) pick(data[0]);
    }).catch((e) => setErr(e.message));
  }, [category]);

  useEffect(() => {
    if (!initialSignId) return;
    api.sign(initialSignId).then((s) => { setCategory(s.category); setSel(s); setRes(null); }).catch(() => {});
  }, [initialSignId]);

  const pick = (s) => {
    if (!s) { setSel(null); return; }
    setRes(null); setErr(""); setAttempt(null);
    api.sign(s.id).then(setSel).catch((e) => setErr(e.message));
  };

  const onSegment = async ({ frames }) => {
    if (!sel || busy) return;
    setBusy(true); setErr("");
    setAttempt(frames);
    try {
      const r = await api.score(sel.id, frames);
      setRes(r);
      loadProgress();
    } catch (e) { setErr(e.message); } finally { setBusy(false); }
  };

  if (category === "sentence") {
    return (
      <section>
        <h2>Practice</h2>
        <div className="subnav" role="tablist">
          {CATS.map(([id, name]) => (
            <button key={id} role="tab" aria-selected={category === id} className={category === id ? "on" : ""}
                    onClick={() => setCategory(id)}>{name}</button>
          ))}
        </div>
        <SentencePractice initialText={initialSentence || ""} />
      </section>
    );
  }

  const target = sel?.label?.toUpperCase();
  const liveTarget = isStaticLetter ? live.scores[target] : isWord ? live.scores[sel?.label] : null;
  const livePred = live.top[0];
  const sameSign = (m) => (isStaticLetter ? m.label === target : m.label === sel?.label);

  return (
    <section className="two">
      <div>
        <h2>Practice</h2>
        <div className="subnav" role="tablist">
          {CATS.map(([id, name]) => (
            <button key={id} role="tab" aria-selected={category === id} className={category === id ? "on" : ""}
                    onClick={() => setCategory(id)}>{name}</button>
          ))}
        </div>
        <div className="list" role="listbox" aria-label="Signs">
          {list.map((s) => {
            const p = progress[s.id];
            return (
              <button key={s.id} role="option" aria-selected={sel?.id === s.id}
                      className={`chip ${s.label.length === 1 ? "letter" : ""} ${sel?.id === s.id ? "on" : ""} ${p?.mastered ? "mastered" : ""}`}
                      onClick={() => pick(s)} title={p ? `best ${p.best}, recent ${p.recent_mean}` : "not practised yet"}>
                {s.label}{p?.mastered ? " ✓" : ""}
              </button>
            );
          })}
          {list.length === 0 && <p className="muted">No signs in this category yet.</p>}
        </div>
        {sel && (
          <div style={{ marginTop: "1.2rem" }}>
            <div className="head-row">
              <h3>Reference: <span className="hl">{sel.label}</span></h3>
              <button className="speak-btn" onClick={() => speak(sel.label)}>🔊</button>
            </div>
            <ReferencePanel sign={sel} />
          </div>
        )}
      </div>

      <div>
        {!sel ? <p className="muted">Choose a sign.</p> : (
          <>
            <div className="head-row">
              <h2>Your turn</h2>
              <div className="subnav slim" role="group" aria-label="Capture mode">
                <button className={capture === "auto" ? "on" : ""} onClick={() => setCapture("auto")}>Auto</button>
                <button className={capture === "hold" ? "on" : ""} onClick={() => setCapture("hold")}>Hold to record</button>
              </div>
            </div>
            {isMotionLetter && (
              <p className="banner">{sel.label} is a motion letter. It isn't scored yet: there is no multi-signer motion data
                for it (record it on the Record page). You can still practise against the reference.</p>
            )}
            {isOwnSign && (
              <p className="banner">Your own signs aren't scored (there is no model for them); practise against your
                recorded reference.</p>
            )}
            {isWord && !wordsReady && (
              <p className="banner">Words are scored once the word model is trained
                (<code>python -m scripts.train_words</code>). You can still practise against the reference.</p>
            )}
            {isWord && wordsReady && (
              <p className="muted small">Sign once, then pause: your attempt is compared with real signers' productions
                (handshape, palm orientation, location, movement).</p>
            )}
            <Camera mode={capture} kind={isStaticLetter ? "static" : "dynamic"} onFrame={live.onFrame}
                    onSegment={onSegment} disabled={busy} holdLabel={`Hold to sign "${sel.label}"`} />
            {(isStaticLetter || (isWord && wordsReady && live.ready)) && (
              <div className="live-box" aria-live="polite">
                {live.error ? <p className="error">{live.error}</p> : (
                  <>
                    <Meter label={`Live match for ${target}`} value={liveTarget} />
                    {livePred && !sameSign(livePred) && livePred.confidence > 0.4 && (
                      <p className="muted small">Looks like <b>{livePred.label.toUpperCase()}</b> right now.</p>
                    )}
                    {isWord && <p className="muted small">Live guide from the last ~2 s ({live.latencyMs.toFixed(0)} ms per update); the score after you finish counts.</p>}
                  </>
                )}
              </div>
            )}
            {busy && <p className="note">Scoring…</p>}
            {err && <p className="error">{err}</p>}
            {res && (
              <div className={`score ${res.score >= 80 ? "good" : res.score >= 50 ? "mid" : "low"}`} aria-live="polite">
                <div className="score-head">
                  <strong>{Math.round(res.score)}</strong>
                  <span>{res.feedback}</span>
                </div>
                {Object.entries(res.components).map(([k, v]) => <Meter key={k} label={k.replace("_", " ")} value={v / 100} />)}
                <ul className="hints">{res.hints.map((h) => <li key={h}>{h}</li>)}</ul>
                <p className="muted small">Estimate from {res.method} ({res.model_version}). Scores are a learning aid, not a certification.</p>
              </div>
            )}
            {attempt && (
              <div className="card">
                <button className="chip small" onClick={() => setShowAttempt((v) => !v)} aria-expanded={showAttempt}>
                  {showAttempt ? "Hide" : "Replay"} my attempt (R)
                </button>
                {showAttempt && (
                  <div className="side-by-side">
                    <SkeletonPlayer frames={attempt} size={220} label="Your attempt (landmarks)" />
                    {sel.reference_landmarks && (
                      <SkeletonPlayer frames={sel.reference_landmarks.frames} size={220} label={`Reference landmarks for ${sel.label}`} />
                    )}
                  </div>
                )}
              </div>
            )}
          </>
        )}
      </div>
    </section>
  );
}
