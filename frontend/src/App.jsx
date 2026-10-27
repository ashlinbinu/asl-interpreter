import { useEffect, useState } from "react";
import Recognize from "./pages/Recognize.jsx";
import TextToSign from "./pages/TextToSign.jsx";
import Practice from "./pages/Practice.jsx";
import Record from "./pages/Record.jsx";
import Library from "./pages/Library.jsx";
import { api } from "./api.js";
import { learner, updateLearner } from "./lib/identity.js";

const TABS = [
  ["recognize", "Sign → text", Recognize],
  ["text", "Text → sign", TextToSign],
  ["practice", "Practice", Practice],
  ["record", "Record dataset", Record],
  ["library", "Add a sign", Library],
];

function ModelBadge({ m }) {
  if (!m?.ready) return <span className="badge" title={m?.error || ""}>Alphabet model: not trained</span>;
  const acc = m.accuracy != null ? `${(m.accuracy * 100).toFixed(1)}%` : "n/a";
  const proto = m.eval_protocol === "signer_holdout" ? "held-out signers" : m.eval_protocol;
  return (
    <span className={`badge ${m.warning ? "warnb" : "ai"}`} title={`${m.version} · ${m.type} · ${proto}${m.warning ? " · " + m.warning : ""}`}>
      Alphabet: {acc} ({proto})
    </span>
  );
}

export default function App() {
  const [nav, setNav] = useState({ tab: "recognize", signId: null });
  const [status, setStatus] = useState(null);
  const [me, setMe] = useState(learner());
  const Page = TABS.find((t) => t[0] === nav.tab)[2];
  const go = (tab, signId = null) => setNav({ tab, signId });

  useEffect(() => {
    const fetchStatus = () => api.status().then(setStatus).catch(() => setStatus({ ok: false }));
    fetchStatus();
    const timer = setInterval(fetchStatus, 15000);
    const onMe = (e) => setMe(e.detail);
    window.addEventListener("handspeak:learner", onMe);
    return () => { clearInterval(timer); window.removeEventListener("handspeak:learner", onMe); };
  }, []);

  useEffect(() => { document.documentElement.dataset.contrast = me.highContrast ? "high" : ""; }, [me.highContrast]);

  return (
    <div className="shell">
      <a href="#main" className="skip">Skip to content</a>
      <header>
        <div className="brand-group">
          <h1>Hand<span>speak</span></h1>
          <div className="status-pill-group">
            {status?.database?.connected
              ? <span className="badge ok">● {status.database.signs_count} signs</span>
              : <span className="badge err">○ Backend unavailable</span>}
            {status?.models && <ModelBadge m={status.models.alphabet} />}
          </div>
        </div>
        <nav aria-label="Sections">
          {TABS.map(([id, name]) => (
            <button key={id} className={nav.tab === id ? "tab on" : "tab"} aria-current={nav.tab === id ? "page" : undefined}
                    onClick={() => go(id)}>{name}</button>
          ))}
        </nav>
        <div className="prefs">
          <label className="inline-check"><input type="checkbox" checked={me.leftHanded}
            onChange={(e) => updateLearner({ leftHanded: e.target.checked })} /> Left-handed</label>
          <label className="inline-check"><input type="checkbox" checked={me.highContrast}
            onChange={(e) => updateLearner({ highContrast: e.target.checked })} /> High contrast</label>
        </div>
      </header>
      <main id="main">
        <Page go={go} initialSignId={nav.signId} key={nav.tab + (nav.signId ?? "")} />
      </main>
      <footer className="muted small">
        Handspeak is a learning aid, not a certified ASL translator. Scores are estimates; facial grammar and other
        non-manual markers are only partly captured. Learn from Deaf signers whenever you can.
      </footer>
    </div>
  );
}
