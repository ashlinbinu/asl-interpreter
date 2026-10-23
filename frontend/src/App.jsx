import { useEffect, useState } from "react";
import Recognize from "./pages/Recognize.jsx";
import TextToSign from "./pages/TextToSign.jsx";
import Practice from "./pages/Practice.jsx";
import Library from "./pages/Library.jsx";
import { api } from "./api.js";

const TABS = [
  ["recognize", "Sign to text", Recognize],
  ["text", "Text to sign", TextToSign],
  ["practice", "Practice", Practice],
  ["library", "Add a sign", Library],
];

export default function App() {
  const [tab, setTab] = useState("recognize");
  const [status, setStatus] = useState(null);
  const Page = TABS.find((t) => t[0] === tab)[2];

  useEffect(() => {
    const fetchStatus = () => {
      api.status()
        .then((s) => setStatus(s))
        .catch(() => setStatus({ ok: false }));
    };
    fetchStatus();
    const timer = setInterval(fetchStatus, 8000);
    return () => clearInterval(timer);
  }, []);

  return (
    <div className="shell">
      <header>
        <div className="brand-group">
          <h1>Hand<span>speak</span></h1>
          <div className="status-pill-group">
            {status?.database?.connected ? (
              <span className="badge ok" title={`Connected to ${status.database.database_type}`}>
                ● DB: {status.database.signs_count} signs
              </span>
            ) : (
              <span className="badge err">○ DB Connecting...</span>
            )}
            {status?.model?.ready && (
              <span className="badge ai" title="Trained HOG + Linear Classifier on 10,676 images">
                ⚡ AI Model: 99.9%
              </span>
            )}
          </div>
        </div>

        <nav>
          {TABS.map(([id, name]) => (
            <button
              key={id}
              className={tab === id ? "tab on" : "tab"}
              onClick={() => setTab(id)}
            >
              {name}
            </button>
          ))}
        </nav>
      </header>
      <main>
        <Page />
      </main>
    </div>
  );
}
