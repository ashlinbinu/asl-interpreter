import { useState } from "react";
import Camera from "../components/Camera.jsx";
import SkeletonPlayer from "../components/SkeletonPlayer.jsx";
import { api } from "../api.js";

export default function Recognize() {
  const [mode, setMode] = useState("webcam"); // "webcam" or "upload"
  const [matches, setMatches] = useState(null);
  const [method, setMethod] = useState("");
  const [ref, setRef] = useState(null);
  const [err, setErr] = useState("");
  const [loading, setLoading] = useState(false);
  const [previewImg, setPreviewImg] = useState(null);

  const speak = (text) => {
    if ("speechSynthesis" in window) {
      window.speechSynthesis.cancel();
      const u = new SpeechSynthesisUtterance(text);
      u.rate = 0.9;
      window.speechSynthesis.speak(u);
    }
  };

  const onCapture = async (data) => {
    setErr("");
    setRef(null);
    setLoading(true);
    try {
      // data is { frames, image }
      const res = await api.recognize(data);
      setMatches(res.matches || []);
      setMethod(res.method || "recognition");
      if (res.matches?.[0]) {
        // Auto-fetch reference skeleton if sign exists in database
        try {
          const signs = await api.signs(res.matches[0].label);
          const hit = signs.find((s) => s.label.toUpperCase() === res.matches[0].label.toUpperCase());
          if (hit) {
            setRef(await api.sign(hit.id));
          }
        } catch (_) {}
      }
    } catch (e) {
      setErr(e.message);
    } finally {
      setLoading(false);
    }
  };

  const handleFileUpload = (e) => {
    const file = e.target.files?.[0];
    if (!file) return;
    setErr("");
    setRef(null);
    setLoading(true);

    const reader = new FileReader();
    reader.onload = async (ev) => {
      const base64 = ev.target.result;
      setPreviewImg(base64);
      try {
        const res = await api.recognize({ image: base64 });
        setMatches(res.matches || []);
        setMethod(res.method || "image_classifier");
        if (res.matches?.[0]) {
          try {
            const signs = await api.signs(res.matches[0].label);
            const hit = signs.find((s) => s.label.toUpperCase() === res.matches[0].label.toUpperCase());
            if (hit) setRef(await api.sign(hit.id));
          } catch (_) {}
        }
      } catch (err) {
        setErr(err.message);
      } finally {
        setLoading(false);
      }
    };
    reader.readAsDataURL(file);
  };

  const topMatch = matches?.[0];

  return (
    <section className="two">
      <div>
        <h2>Sign Recognition</h2>
        <div className="subnav">
          <button className={mode === "webcam" ? "on" : ""} onClick={() => setMode("webcam")}>
            📹 Live Camera
          </button>
          <button className={mode === "upload" ? "on" : ""} onClick={() => setMode("upload")}>
            📁 Image Upload / Dataset
          </button>
        </div>

        {mode === "webcam" ? (
          <>
            <p className="muted">Make your sign in front of the camera, press record, and get instant interpretation.</p>
            <Camera onCapture={onCapture} />
          </>
        ) : (
          <div>
            <p className="muted">Upload an ASL sign image or sample from the dataset to test the trained AI classifier.</p>
            <label className="dropzone" style={{ display: "block" }}>
              <input type="file" accept="image/*" onChange={handleFileUpload} style={{ display: "none" }} />
              <p style={{ margin: "0.5rem 0", fontWeight: 600 }}>Click to select an image or drag & drop</p>
              <span className="muted" style={{ fontSize: "0.85rem" }}>Supports JPG, PNG (e.g. from Desktop/PYTHON/Data)</span>
            </label>
            {previewImg && (
              <div style={{ marginTop: "1rem" }}>
                <p className="muted">Uploaded Image:</p>
                <img src={previewImg} alt="Uploaded sign" style={{ maxWidth: "200px", borderRadius: "8px", border: "1px solid var(--line)" }} />
              </div>
            )}
          </div>
        )}
      </div>

      <div>
        <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between" }}>
          <h2>Interpretation Results</h2>
          {topMatch && (
            <button className="speak-btn" onClick={() => speak(topMatch.label)} title="Speak with voice">
              🔊 Speak "{topMatch.label}"
            </button>
          )}
        </div>

        {loading && <p className="note">Processing sign recognition...</p>}
        {err && <p className="error">{err}</p>}

        {!matches && !loading && !err && (
          <p className="muted">Your recognition matches and confidence scores will appear here.</p>
        )}

        {matches && matches.length === 0 && (
          <p className="muted">No match found. Please position hand clearly in frame.</p>
        )}

        {matches?.map((m, i) => (
          <div key={m.label || i} className="row">
            <span className={i === 0 ? "best" : ""}>
              {m.label} {i === 0 && "⭐"}
            </span>
            <div className="bar">
              <i style={{ width: `${Math.min(m.score, 100)}%` }} />
            </div>
            <b>{m.score}%</b>
          </div>
        ))}

        {method && (
          <p className="muted" style={{ fontSize: "0.78rem", marginTop: "0.8rem" }}>
            Engine: {method === "image_classifier" ? "⚡ AI Linear Classifier (HOG)" : method === "dtw_landmarks" ? "📐 3D Landmark DTW" : "⚡ Hybrid Fusion"}
          </p>
        )}

        {ref && (
          <div style={{ marginTop: "1.5rem" }}>
            <p className="muted">Reference animated sign for <b>{ref.label}</b>:</p>
            <SkeletonPlayer frames={ref.frames} />
          </div>
        )}
      </div>
    </section>
  );
}
