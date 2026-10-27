import { useEffect, useState } from "react";
import Camera from "../components/Camera.jsx";
import ReferencePanel from "../components/ReferencePanel.jsx";
import { api } from "../api.js";
import { useLiveLetters } from "../lib/useLiveLetters.js";
import { speak } from "../lib/speech.js";

function Matches({ matches, rejectThreshold, onLearn, letterIds }) {
  if (!matches?.length) return null;
  const unsure = matches[0].confidence < rejectThreshold;
  return (
    <div>
      {unsure && <p className="muted">Not sure yet: hold the handshape steady, facing the camera.</p>}
      {matches.map((m, i) => (
        <div key={m.label} className="row match">
          <span className={i === 0 && !unsure ? "best" : ""}>{m.label}</span>
          <div className="bar" aria-hidden="true"><i style={{ width: `${Math.round(m.confidence * 100)}%` }} /></div>
          <b>{Math.round(m.confidence * 100)}%</b>
          <button className="chip small" onClick={() => onLearn(m.sign_id ?? letterIds[m.label], m.label)}
                  aria-label={`Learn the sign ${m.label}`}>Learn</button>
        </div>
      ))}
    </div>
  );
}

export default function Recognize({ go }) {
  const [mode, setMode] = useState("live");
  const live = useLiveLetters(mode === "live");
  const [server, setServer] = useState(null);
  const [err, setErr] = useState("");
  const [loading, setLoading] = useState(false);
  const [preview, setPreview] = useState(null);
  const [focus, setFocus] = useState(null);
  const [letterIds, setLetterIds] = useState({});

  useEffect(() => {
    api.signs("", "alphabet").then((l) => setLetterIds(Object.fromEntries(l.map((s) => [s.label, s.id])))).catch(() => {});
  }, []);

  const matches = mode === "live" ? live.top : server?.matches;
  const threshold = mode === "live" ? live.model?.rejectThreshold ?? 0 : 0;
  const learn = (signId, label) => setFocus({ id: signId, label });

  const onSegment = async ({ frames }) => {
    setErr("");
    try {
      setServer(await api.recognize({ frames, mode: "alphabet" }));
    } catch (e) { setErr(e.message); }
  };

  const onFile = (e) => {
    const file = e.target.files?.[0];
    if (!file) return;
    setErr(""); setLoading(true);
    const reader = new FileReader();
    reader.onload = async (ev) => {
      setPreview(ev.target.result);
      try {
        setServer(await api.recognize({ image: ev.target.result, mode: "alphabet" }));
      } catch (er) { setErr(er.message); setServer(null); } finally { setLoading(false); }
    };
    reader.readAsDataURL(file);
  };

  return (
    <section className="two">
      <div>
        <h2>Sign → text</h2>
        <div className="subnav" role="tablist">
          <button role="tab" aria-selected={mode === "live"} className={mode === "live" ? "on" : ""} onClick={() => setMode("live")}>Live camera</button>
          <button role="tab" aria-selected={mode === "upload"} className={mode === "upload" ? "on" : ""} onClick={() => setMode("upload")}>Image upload</button>
        </div>
        <p className="banner">Fingerspelling (static letters) is recognised now. Moving letters (J, Z) and whole words
          need the dynamic word model (phase P3).</p>
        {mode === "live" ? (
          <>
            {live.error && <p className="error">{live.error}. Train it first: see README "Train the alphabet model".</p>}
            <Camera mode="auto" kind="static" onFrame={live.onFrame} onSegment={onSegment} />
            {live.ready && <p className="muted small">In-browser model {live.model.version} · {live.latencyMs.toFixed(1)} ms per update</p>}
          </>
        ) : (
          <div>
            <label className="dropzone">
              <input type="file" accept="image/*" onChange={onFile} className="sr-only" />
              <p><strong>Choose a photo of a fingerspelled letter</strong></p>
              <span className="muted small">Landmarks are extracted on the server, then the same landmark model is used.</span>
            </label>
            {preview && <img src={preview} alt="Uploaded sign" className="preview" />}
          </div>
        )}
      </div>

      <div>
        <div className="head-row">
          <h2>Top matches</h2>
          {matches?.[0] && <button className="speak-btn" onClick={() => speak(matches[0].label)}>🔊 Speak "{matches[0].label}"</button>}
        </div>
        {loading && <p className="note">Extracting landmarks…</p>}
        {err && <p className="error">{err}</p>}
        {!matches?.length && !loading && !err && <p className="muted">Matches and confidence appear here.</p>}
        <Matches matches={matches} rejectThreshold={threshold} onLearn={learn} letterIds={letterIds} />
        {mode === "live" && server?.matches?.[0] && (
          <p className="muted small">Last auto-captured: <b>{server.matches[0].label}</b> ({Math.round(server.matches[0].confidence * 100)}%, server check)</p>
        )}
        {focus?.id && (
          <div className="card">
            <ReferencePanel signId={focus.id} compact />
            <button className="btn" onClick={() => go("practice", focus.id)}>Practise {focus.label}</button>
          </div>
        )}
      </div>
    </section>
  );
}
