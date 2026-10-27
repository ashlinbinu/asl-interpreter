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

const HELD_OUT = ["signer_holdout", "official_signer_split"];

function ModelBadge({ m, name }) {
  if (!m?.ready) return <span className="badge" title={m?.error || ""}>{name} model: not trained</span>;
  const acc = m.accuracy != null ? `${(m.accuracy * 100).toFixed(1)}%` : "n/a";
  const proto = HELD_OUT.includes(m.eval_protocol) ? "held-out signers" : m.eval_protocol;
  const why = m.verified ? "" : name === "Alphabet"
    ? ` · unverified: needs ≥3 held-out full-frame video signers (has ${m.video_signers ?? 0})`
    : " · unverified: fewer than 3 held-out test signers";
  const extra = m.top5_accuracy != null ? ` · top-5 ${(m.top5_accuracy * 100).toFixed(1)}%` : "";
  return (
    <span className={`badge ${m.verified && !m.warning ? "ai" : "warnb"}`}
          title={`${m.version} · ${m.type} · ${proto}${extra}${why}${m.warning ? " · " + m.warning : ""}`}>
      {name}: {acc} ({m.verified ? proto : "unverified"})
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
            {status?.models && <ModelBadge m={status.models.alphabet} name="Alphabet" />}
            {status?.models && <ModelBadge m={status.models.words} name="Words" />}
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
