import { useEffect, useState } from "react";
import Camera from "../components/Camera.jsx";
import SkeletonPlayer from "../components/SkeletonPlayer.jsx";
import { api } from "../api.js";

export default function Practice() {
  const [category, setCategory] = useState("alphabet"); // "alphabet" | "common" | "user"
  const [list, setList] = useState([]);
  const [sel, setSel] = useState(null);
  const [res, setRes] = useState(null);
  const [err, setErr] = useState("");
  const [loading, setLoading] = useState(false);
  const [mastered, setMastered] = useState(() => new Set());
  const [streak, setStreak] = useState(0);

  useEffect(() => {
    api.signs("", category)
      .then((data) => {
        setList(data);
        if (data.length > 0 && !sel) {
          // Auto-select first item
          pick(data[0]);
        }
      })
      .catch((e) => setErr(e.message));
  }, [category]);

  const pick = async (s) => {
    setRes(null);
    setErr("");
    try {
      const full = await api.sign(s.id);
      setSel(full);
    } catch (e) {
      setErr(e.message);
    }
  };

  const onCapture = async (data) => {
    if (!sel) return;
    setLoading(true);
    setErr("");
    try {
      const frames = data.frames || data;
      const result = await api.score(sel.id, frames);
      setRes(result);
      if (result.score >= 75) {
        setMastered((prev) => new Set([...prev, sel.label]));
        setStreak((prev) => prev + 1);
      } else {
        setStreak(0);
      }
    } catch (e) {
      setErr(e.message);
    } finally {
      setLoading(false);
    }
  };

  const speak = (text) => {
    if ("speechSynthesis" in window) {
      window.speechSynthesis.cancel();
      window.speechSynthesis.speak(new SpeechSynthesisUtterance(text));
    }
  };

  return (
    <section className="two">
      <div>
        <h2>Interactive Practice Studio</h2>
        <div className="streak-box">
          <div className="streak-item">
            <small>Mastered</small>
            <strong>{mastered.size} Signs</strong>
          </div>
          <div className="streak-item">
            <small>Active Streak</small>
            <strong>{streak} 🔥</strong>
          </div>
          <div className="streak-item">
            <small>Total in Library</small>
            <strong>{list.length}</strong>
          </div>
        </div>

        <div className="subnav">
          <button className={category === "alphabet" ? "on" : ""} onClick={() => setCategory("alphabet")}>
            🔤 Alphabet (A-Z)
          </button>
          <button className={category === "common" ? "on" : ""} onClick={() => setCategory("common")}>
            💬 Everyday Words
          </button>
          <button className={category === "" ? "on" : ""} onClick={() => setCategory("")}>
            🌐 All Signs
          </button>
        </div>

        <p className="muted" style={{ fontSize: "0.88rem", marginBottom: "0.8rem" }}>
          Select a sign below to see its 3D neon reference animation:
        </p>

        <div className="list">
          {list.map((s) => {
            const isLetter = s.label.length === 1 && s.label.toUpperCase() === s.label;
            const isDone = mastered.has(s.label);
            return (
              <button
                key={s.id}
                className={`chip ${isLetter ? "letter" : ""} ${sel?.id === s.id ? "on" : ""}`}
                style={isDone ? { borderColor: "var(--green)" } : {}}
                onClick={() => pick(s)}
              >
                {s.label} {isDone && "✓"}
              </button>
            );
          })}
          {list.length === 0 && <p className="muted">No signs in this category.</p>}
        </div>
      </div>

      <div>
        {!sel ? (
          <p className="muted">Choose a sign on the left to see the reference animation.</p>
        ) : (
          <>
            <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between" }}>
              <h2>Target: <span style={{ color: "var(--neon)" }}>{sel.label}</span></h2>
              <button className="speak-btn" onClick={() => speak(sel.label)}>
                🔊 Pronounce
              </button>
            </div>

            <SkeletonPlayer frames={sel.frames} size={280} />

            <div style={{ marginTop: "1rem" }}>
              <p className="muted">Match the reference pose shown above and hold it in camera view:</p>
              <Camera onCapture={onCapture} buttonText={`Record Attempt for "${sel.label}"`} />
            </div>

            {loading && <p className="note">Analyzing your hand posture...</p>}
            {err && <p className="error">{err}</p>}

            {res && (
              <div
                className="score"
                style={{
                  borderColor: res.score >= 75 ? "var(--green)" : res.score >= 50 ? "var(--neon)" : "#ff6b6b",
                }}
              >
                <div style={{ display: "flex", alignItems: "baseline", gap: "0.75rem" }}>
                  <strong style={{ color: res.score >= 75 ? "var(--green)" : "var(--neon)" }}>
                    {res.score}%
                  </strong>
                  <span style={{ fontWeight: 600, color: res.score >= 75 ? "var(--green)" : "var(--muted)" }}>
                    {res.score >= 85 ? "Excellent!" : res.score >= 65 ? "Good Job!" : "Keep Trying!"}
                  </span>
                </div>
                <p>{res.feedback}</p>
              </div>
            )}
          </>
        )}
      </div>
    </section>
  );
}
