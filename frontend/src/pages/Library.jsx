import { useState } from "react";
import Camera from "../components/Camera.jsx";
import { api } from "../api.js";

export default function Library() {
  const [label, setLabel] = useState("");
  const [msg, setMsg] = useState("");
  const [err, setErr] = useState("");
  const [loading, setLoading] = useState(false);

  const onCapture = async (data) => {
    const frames = data.frames || data;
    setLoading(true);
    setMsg("");
    setErr("");
    try {
      const s = await api.addSign(label.trim(), frames);
      setMsg(`Successfully saved custom sign "${s.label}". It is now saved in the database and immediately available in Practice, Recognize, and Text-to-Sign!`);
      setLabel("");
    } catch (e) {
      setErr(e.message);
    } finally {
      setLoading(false);
    }
  };

  return (
    <section>
      <h2>Add a Custom Sign</h2>
      <p className="muted">
        Create and record custom vocabulary or specialized gestures. Once recorded, they are stored
        directly in the database and loaded into the recognition and translation pipelines.
      </p>

      <div style={{ maxWidth: "480px" }}>
        <input
          value={label}
          placeholder="Word or phrase (e.g. 'Coffee', 'Welcome', 'My Name')"
          onChange={(e) => setLabel(e.target.value)}
        />

        {label.trim() ? (
          <div>
            <p className="muted">Position your hand, click record, and perform your sign within 3 seconds:</p>
            <Camera onCapture={onCapture} buttonText={`Record and Save "${label.trim()}"`} />
          </div>
        ) : (
          <p className="muted">Enter a label above to enable camera recording.</p>
        )}
      </div>

      {loading && <p className="note">Saving sign to database...</p>}
      {err && <p className="error" style={{ marginTop: "1rem" }}>{err}</p>}
      {msg && <p className="note" style={{ marginTop: "1rem" }}>{msg}</p>}
    </section>
  );
}
