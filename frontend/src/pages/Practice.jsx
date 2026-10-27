import { useEffect, useState } from "react";
import Camera from "../components/Camera.jsx";
import ReferencePanel from "../components/ReferencePanel.jsx";
import { api } from "../api.js";
import { useLiveLetters } from "../lib/useLiveLetters.js";
import { speak } from "../lib/speech.js";

const CATS = [["alphabet", "Alphabet"], ["word", "Words"], ["user", "My signs"]];

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

export default function Practice({ initialSignId }) {
  const [category, setCategory] = useState("alphabet");
  const [list, setList] = useState([]);
  const [sel, setSel] = useState(null);
  const [res, setRes] = useState(null);
  const [err, setErr] = useState("");
  const [busy, setBusy] = useState(false);
  const [progress, setProgress] = useState({});
  const [capture, setCapture] = useState("auto");
  const [speed, setSpeed] = useState(1);
  const isStaticLetter = sel?.category === "alphabet" && sel?.kind === "static";
  const live = useLiveLetters(isStaticLetter);

  const loadProgress = () => api.progress().then((p) => setProgress(Object.fromEntries(p.signs.map((s) => [s.sign_id, s])))).catch(() => {});
  useEffect(() => { loadProgress(); }, []);

  useEffect(() => {
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
    setRes(null); setErr("");
    api.sign(s.id).then(setSel).catch((e) => setErr(e.message));
  };

  const onSegment = async ({ frames }) => {
    if (!sel || busy) return;
    setBusy(true); setErr("");
    try {
      const r = await api.score(sel.id, frames);
      setRes(r);
      loadProgress();
    } catch (e) { setErr(e.message); } finally { setBusy(false); }
  };

  const target = sel?.label?.toUpperCase();
  const liveTarget = isStaticLetter ? live.scores[target] : null;
  const livePred = live.top[0];

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
            <ReferencePanel sign={sel} speed={speed} />
            <div className="speed" role="group" aria-label="Playback speed">
              {[0.5, 0.75, 1].map((v) => (
                <button key={v} className={`chip small ${speed === v ? "on" : ""}`} onClick={() => setSpeed(v)}>{v}×</button>
              ))}
            </div>
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
            {!isStaticLetter && (
              <p className="banner">Scoring for moving signs ({sel.label}) needs real reference clips and the dynamic scorer
                (phases P2–P3). You can still practise against the reference.</p>
            )}
            <Camera mode={capture} kind={isStaticLetter ? "static" : "dynamic"} onFrame={live.onFrame}
                    onSegment={onSegment} disabled={busy} holdLabel={`Hold to sign "${sel.label}"`} />
            {isStaticLetter && (
              <div className="live-box" aria-live="polite">
                {live.error ? <p className="error">{live.error}</p> : (
                  <>
                    <Meter label={`Live match for ${target}`} value={liveTarget} />
                    {livePred && livePred.label !== target && livePred.confidence > 0.4 && (
                      <p className="muted small">Looks like <b>{livePred.label}</b> right now.</p>
                    )}
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
                {Object.entries(res.components).map(([k, v]) => <Meter key={k} label={k} value={v / 100} />)}
                <ul className="hints">{res.hints.map((h) => <li key={h}>{h}</li>)}</ul>
                <p className="muted small">Estimate from {res.method} ({res.model_version}). Scores are a learning aid, not a certification.</p>
              </div>
            )}
          </>
        )}
      </div>
    </section>
  );
}
