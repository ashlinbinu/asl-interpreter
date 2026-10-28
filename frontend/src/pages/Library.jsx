import { useEffect, useState } from "react";
import Camera from "../components/Camera.jsx";
import { api } from "../api.js";

export default function Library() {
  const [label, setLabel] = useState("");
  const [msg, setMsg] = useState("");
  const [err, setErr] = useState("");
  const [loading, setLoading] = useState(false);
  const [mine, setMine] = useState(null);
  const [deleting, setDeleting] = useState(null);
  const [again, setAgain] = useState(null);     // custom sign being recorded again

  const loadMine = () => api.signs("", "user").then(setMine).catch((e) => setErr(e.message));
  useEffect(() => { loadMine(); }, []);

  const remove = async (s) => {
    if (!window.confirm(`Delete '${s.label}'? This cannot be undone.`)) return;
    setDeleting(s.id); setMsg(""); setErr("");
    try {
      const r = await api.deleteSign(s.id);
      setMsg(`Deleted "${r.label}" (${r.files_removed} file${r.files_removed === 1 ? "" : "s"} removed; past scores stay in Progress).`);
      if (again?.id === s.id) setAgain(null);
    } catch (e) {
      setErr(`Could not delete "${s.label}": ${e.message}`);
    } finally {
      setDeleting(null);
      loadMine();
    }
  };

  // The take's landmarks are the reference; its video (when the browser recorded one) is shown as the reference video.
  const onSegment = async ({ frames, video }) => {
    setLoading(true); setMsg(""); setErr("");
    try {
      let s;
      if (again) {
        s = await api.recordSign(again.id, frames, video);
        setMsg(`Recorded "${s.label}" again: the new take is its reference${video ? ", with your video" : ""}.`);
        setAgain(null);
      } else {
        s = await api.addSign(label.trim(), frames);
        if (video) s = await api.recordSign(s.id, frames, video);
        setMsg(`Saved "${s.label}" with your recording${video ? " and video" : ""} as its reference. Find it under Practice → My signs.`);
        setLabel("");
      }
      loadMine();
    } catch (e) { setErr(e.message); } finally { setLoading(false); }
  };

  const target = again ? again.label : label.trim();
  return (
    <section>
      <h2>Add a custom sign</h2>
      <p className="muted">Record your own sign (for example a name sign). Hold the button for the whole sign; the
        landmarks and the video are stored as its reference and shown in Practice and Text → Sign.</p>
      <div style={{ maxWidth: 520 }}>
        {again ? (
          <p className="banner">Recording <b>{again.label}</b> again: the new take replaces its old recording.{" "}
            <button className="chip small" onClick={() => setAgain(null)}>Cancel</button></p>
        ) : (
          <label>Word or phrase
            <input value={label} placeholder="e.g. coffee, my name sign" onChange={(e) => setLabel(e.target.value)} />
          </label>
        )}
        {target
          ? <Camera key={again ? `again-${again.id}` : "new"} mode="hold" recordVideo onSegment={onSegment}
                    holdLabel={`Hold to record "${target}"`} disabled={loading} />
          : <p className="muted">Enter a label to enable recording.</p>}
      </div>
      {loading && <p className="note">Saving…</p>}
      {err && <p className="error" role="alert">{err}</p>}
      {msg && <p className="note" role="status">{msg}</p>}

      <h3>Your custom signs</h3>
      {mine === null ? <p className="muted">Loading…</p> : mine.length === 0 ? (
        <p className="muted">No custom signs yet.</p>
      ) : (
        <ul className="custom-signs">
          {mine.map((s) => (
            <li key={s.id}>
              <span>{s.label}</span>
              <span className={s.reference_clips ? "muted small" : "warn small"}>
                {s.reference_clips
                  ? `${s.reference_clips} recording${s.reference_clips === 1 ? "" : "s"}${s.has_video ? " · video" : ""}`
                  : "no recording: can't be scored"}
              </span>
              <button className="chip small" onClick={() => { setAgain(s); setMsg(""); setErr(""); }}
                      disabled={loading || again?.id === s.id} aria-label={`Record ${s.label} again`}>Record again</button>
              <button className="chip small danger" onClick={() => remove(s)} disabled={deleting === s.id}
                      aria-label={`Delete ${s.label}`}>{deleting === s.id ? "Deleting…" : "Delete"}</button>
            </li>
          ))}
        </ul>
      )}
    </section>
  );
}
