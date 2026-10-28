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

  const loadMine = () => api.signs("", "user").then(setMine).catch((e) => setErr(e.message));
  useEffect(() => { loadMine(); }, []);

  const remove = async (s) => {
    if (!window.confirm(`Delete '${s.label}'? This cannot be undone.`)) return;
    setDeleting(s.id); setMsg(""); setErr("");
    try {
      const r = await api.deleteSign(s.id);
      setMsg(`Deleted "${r.label}" (${r.files_removed} file${r.files_removed === 1 ? "" : "s"} removed; past scores stay in Progress).`);
    } catch (e) {
      setErr(`Could not delete "${s.label}": ${e.message}`);
    } finally {
      setDeleting(null);
      loadMine();
    }
  };

  const onSegment = async ({ frames }) => {
    setLoading(true); setMsg(""); setErr("");
    try {
      const s = await api.addSign(label.trim(), frames);
      setMsg(`Saved "${s.label}" with your recording as its reference. Find it under Practice → My signs.`);
      setLabel("");
      loadMine();
    } catch (e) { setErr(e.message); } finally { setLoading(false); }
  };

  return (
    <section>
      <h2>Add a custom sign</h2>
      <p className="muted">Record your own sign (for example a name sign). The landmarks are stored as a real reference
        clip and shown in Practice and Text → Sign.</p>
      <div style={{ maxWidth: 520 }}>
        <label>Word or phrase
          <input value={label} placeholder="e.g. coffee, my name sign" onChange={(e) => setLabel(e.target.value)} />
        </label>
        {label.trim()
          ? <Camera mode="hold" onSegment={onSegment} holdLabel={`Hold to record "${label.trim()}"`} disabled={loading} />
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
              <span className="muted small">{s.reference_clips} recording{s.reference_clips === 1 ? "" : "s"}</span>
              <button className="chip small danger" onClick={() => remove(s)} disabled={deleting === s.id}
                      aria-label={`Delete ${s.label}`}>{deleting === s.id ? "Deleting…" : "Delete"}</button>
            </li>
          ))}
        </ul>
      )}
    </section>
  );
}
