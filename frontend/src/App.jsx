import { useEffect, useState } from "react";
import Recognize from "./pages/Recognize.jsx";
import TextToSign from "./pages/TextToSign.jsx";
import Practice from "./pages/Practice.jsx";
import Record from "./pages/Record.jsx";
import Library from "./pages/Library.jsx";
import Progress from "./pages/Progress.jsx";
import { api } from "./api.js";
import { learner, updateLearner } from "./lib/identity.js";

const TABS = [
  ["recognize", "Sign → text", Recognize],
  ["text", "Text → sign", TextToSign],
  ["practice", "Practice", Practice],
  ["progress", "Progress", Progress],
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
  // Bumped whenever the backend becomes reachable, so pages mount (or remount) only when their requests can succeed.
  const [epoch, setEpoch] = useState(0);
  const [me, setMe] = useState(learner());
  const Page = TABS.find((t) => t[0] === nav.tab)[2];
  const go = (tab, signId = null, extra = {}) => setNav({ tab, signId, ...extra });
  const up = !!status?.database?.connected;

  useEffect(() => {
    let alive = true, timer, wasUp = false;
    const poll = () => api.status().catch(() => ({ ok: false })).then((s) => {
      if (!alive) return;
      const nowUp = !!s?.database?.connected;
      if (nowUp && !wasUp) setEpoch((e) => e + 1);
      wasUp = nowUp;
      setStatus(s);
      timer = setTimeout(poll, nowUp ? 15000 : 2000);
    });
    poll();
    const onMe = (e) => setMe(e.detail);
    window.addEventListener("handspeak:learner", onMe);
    return () => { alive = false; clearTimeout(timer); window.removeEventListener("handspeak:learner", onMe); };
  }, []);

  useEffect(() => { document.documentElement.dataset.contrast = me.highContrast ? "high" : ""; }, [me.highContrast]);

  return (
    <div className="shell">
      <a href="#main" className="skip">Skip to content</a>
      <header>
        <div className="brand-group">
          <h1>Hand<span>speak</span></h1>
          <div className="status-pill-group">
            {up
              ? <span className="badge ok">● {status.database.signs_count} signs</span>
              : <span className="badge err">{status ? "○ Backend unavailable" : "○ Connecting…"}</span>}
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
        {epoch > 0 && !up && <p className="banner" role="alert">Lost the connection to the backend: retrying…</p>}
        {epoch === 0 ? (
          <section className="card" role="status">
            <h2>{status ? "Waiting for the backend…" : "Connecting…"}</h2>
            <p>Retrying every 2 seconds; the app opens by itself when the backend answers.</p>
            <p className="muted small">Start it with <code>run_all.bat</code>, or in the <code>backend</code> folder
              run <code>python -m uvicorn app.main:app</code>. It is ready when its window says
              "Application startup complete".</p>
          </section>
        ) : (
          <Page go={go} initialSignId={nav.signId} initialSentence={nav.sentence}
                key={`${epoch}|${nav.tab}|${nav.signId ?? ""}|${nav.sentence ?? ""}`} />
        )}
      </main>
      <footer className="muted small">
        Handspeak is a learning aid, not a certified ASL translator. Scores are estimates; facial grammar and other
        non-manual markers are only partly captured. Learn from Deaf signers whenever you can.
      </footer>
    </div>
  );
}
