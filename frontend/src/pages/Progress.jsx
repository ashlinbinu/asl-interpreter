import { useEffect, useState } from "react";
import { api } from "../api.js";

const fmt = (iso) => (iso ? new Date(iso.endsWith("Z") ? iso : iso + "Z").toLocaleString() : "–");

export default function Progress({ go }) {
  const [p, setP] = useState(null);
  const [err, setErr] = useState("");
  useEffect(() => { api.progress().then(setP).catch((e) => setErr(e.message)); }, []);

  if (err) return <section><h2>Progress</h2><p className="error">{err}</p></section>;
  if (!p) return <section><h2>Progress</h2><p className="muted">Loading…</p></section>;
  const byId = Object.fromEntries(p.signs.map((s) => [s.sign_id, s]));
  const mastered = p.signs.filter((s) => s.mastered).length;

  return (
    <section>
      <h2>Progress</h2>
      <p className="muted">This browser's practice history (anonymous learner id, stored on this computer's backend).
        Mastered = {p.mastery_rule}. {p.review_rule}.</p>
      <div className="stats" role="group" aria-label="Summary">
        <div><b>{p.signs.length}</b><span>signs practised</span></div>
        <div><b>{mastered}</b><span>mastered</span></div>
        <div><b>{p.review_queue.length}</b><span>due for review</span></div>
      </div>

      <h3>Review now</h3>
      {p.review_queue.length ? (
        <div className="list">
          {p.review_queue.map((id) => (
            <button key={id} className="chip" onClick={() => go("practice", id)} title={`box ${byId[id]?.box}, recent ${byId[id]?.recent_mean}`}>
              {byId[id]?.label}
            </button>
          ))}
        </div>
      ) : <p className="muted">Nothing due. Practise new signs, or come back later.</p>}

      <h3>All signs</h3>
      {p.signs.length === 0 ? <p className="muted">No attempts yet. Scores from Practice appear here.</p> : (
        <table className="table">
          <thead><tr><th>Sign</th><th>Attempts</th><th>Best</th><th>Recent mean</th><th>Box</th><th>Status</th><th>Last</th></tr></thead>
          <tbody>
            {p.signs.map((s) => (
              <tr key={s.sign_id}>
                <td>{s.deleted ? <span className="muted">{s.label}</span>
                  : <button className="link" onClick={() => go("practice", s.sign_id)}>{s.label}</button>}</td>
                <td>{s.attempts}</td><td>{Math.round(s.best)}</td><td>{s.recent_mean}</td><td>{s.box}</td>
                <td>{s.mastered ? "mastered ✓" : s.due ? "due" : "learning"}</td><td>{fmt(s.last_at)}</td>
              </tr>
            ))}
          </tbody>
        </table>
      )}

      <h3>Recent attempts</h3>
      <ul className="hints">
        {p.recent.map((a, i) => (
          <li key={i}>{byId[a.sign_id]?.label ?? a.sign_id}: {Math.round(a.score)} ({a.mode}) · {fmt(a.at)}</li>
        ))}
      </ul>
    </section>
  );
}
