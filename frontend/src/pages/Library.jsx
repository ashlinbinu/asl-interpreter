import { useState } from "react";
import Camera from "../components/Camera.jsx";
import { api } from "../api.js";

export default function Library() {
  const [label, setLabel] = useState("");
  const [msg, setMsg] = useState("");
  const [err, setErr] = useState("");
  const [loading, setLoading] = useState(false);

  const onSegment = async ({ frames }) => {
    setLoading(true); setMsg(""); setErr("");
    try {
      const s = await api.addSign(label.trim(), frames);
      setMsg(`Saved "${s.label}" with your recording as its reference. Find it under Practice → My signs.`);
      setLabel("");
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
      {err && <p className="error">{err}</p>}
      {msg && <p className="note">{msg}</p>}
    </section>
  );
}
