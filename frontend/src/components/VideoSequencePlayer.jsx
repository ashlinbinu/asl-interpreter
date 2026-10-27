import { useEffect, useMemo, useRef, useState } from "react";
import { mediaUrl } from "../api.js";

const LETTER_MS = 900;    // a fingerspelled letter card at 1x
const CARD_MS = 1600;     // a known sign without a reference video yet

/** Plan entries (POST /api/text-to-sign) -> playable items: one per sign, one per fingerspelled letter. */
export function sequenceItems(entries) {
  const items = [];
  (entries || []).forEach((e, entry) => {
    if (e.fingerspelled) {
      e.letters.forEach((L, letter) => items.push({
        kind: L.video_url ? "video" : "letter", text: L.letter, url: mediaUrl(L.video_url), entry, letter, word: e.word,
      }));
    } else {
      items.push({ kind: e.video_url ? "video" : "card", text: e.gloss, url: mediaUrl(e.video_url), entry, word: e.word,
                   attribution: e.attribution, license: e.license });
    }
  });
  return items;
}

/**
 * Plays a sign sequence: two stacked <video> elements crossfade (~150 ms; off with reduced motion), letters and
 * signs without video are shown as timed cards. Keys: ←/→ previous/next, R replay, P play/pause.
 */
export default function VideoSequencePlayer({ entries, onActive }) {
  const items = useMemo(() => sequenceItems(entries), [entries]);
  const [i, setI] = useState(0);
  const [playing, setPlaying] = useState(true);
  const [speed, setSpeed] = useState(1);
  const [loop, setLoop] = useState(false);
  const [front, setFront] = useState(0);
  const [replay, setReplay] = useState(0);
  const vids = [useRef(null), useRef(null)];
  const frontRef = useRef(0);
  const timer = useRef(null);
  const item = items[i];

  const advance = () => setI((k) => {
    if (k + 1 < items.length) return k + 1;
    if (loop) return 0;
    setPlaying(false);
    return k;
  });

  useEffect(() => { setI(0); setPlaying(true); }, [items]);

  useEffect(() => {
    clearTimeout(timer.current);
    if (!item) return undefined;
    onActive?.(item.entry, item.letter);
    if (item.kind === "video") {
      const back = 1 - frontRef.current;
      const v = vids[back].current;
      const start = () => {
        vids[frontRef.current].current?.pause();
        frontRef.current = back;
        setFront(back);
        v.playbackRate = speed;
        if (playing) v.play().catch(() => {});
      };
      if (v.dataset.src !== item.url) { v.src = item.url; v.dataset.src = item.url; }
      v.currentTime = 0;
      if (v.readyState >= 2) start(); else v.addEventListener("loadeddata", start, { once: true });
      return () => v.removeEventListener("loadeddata", start);
    }
    vids[frontRef.current].current?.pause();
    if (playing) timer.current = setTimeout(advance, (item.kind === "letter" ? LETTER_MS : CARD_MS) / speed);
    return () => clearTimeout(timer.current);
  }, [i, items, replay, playing]);

  useEffect(() => { vids.forEach((r) => { if (r.current) r.current.playbackRate = speed; }); }, [speed]);

  useEffect(() => {
    const onKey = (e) => {
      if (["INPUT", "TEXTAREA", "SELECT"].includes(e.target?.tagName)) return;
      if (e.key === "ArrowRight") setI((k) => Math.min(k + 1, items.length - 1));
      else if (e.key === "ArrowLeft") setI((k) => Math.max(k - 1, 0));
      else if (e.key.toLowerCase() === "r") { setPlaying(true); setReplay((n) => n + 1); }
      else if (e.key.toLowerCase() === "p") setPlaying((p) => !p);
      else return;
      e.preventDefault();
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [items.length]);

  if (!items.length) return null;
  const onEnded = (k) => () => { if (k === frontRef.current && playing) advance(); };
  const jump = (entry) => { setI(items.findIndex((it) => it.entry === entry)); setPlaying(true); setReplay((n) => n + 1); };
  const showVideo = item?.kind === "video";
  const caption = item?.kind === "letter" ? `${item.text} (fingerspelling "${item.word}")`
    : item && item.word && item.word.toUpperCase().replace(/ /g, "-") !== item.text ? `${item.text} ("${item.word}")` : item?.text;

  return (
    <div className="seq-player" aria-roledescription="sign sequence player">
      <div className="seq-stage">
        {[0, 1].map((k) => (
          <video key={k} ref={vids[k]} muted playsInline preload="auto" onEnded={onEnded(k)}
                 className={`seq-video ${showVideo && front === k ? "on" : ""}`} aria-hidden={!(showVideo && front === k)} />
        ))}
        {!showVideo && item && (
          <div className={`seq-card ${item.kind}`} aria-hidden="true">
            <span>{item.text}</span>
            <small>{item.kind === "letter" ? "fingerspell" : "no reference video yet"}</small>
          </div>
        )}
      </div>
      <p className="caption" aria-live="polite">{caption}</p>
      <div className="seq-controls" role="group" aria-label="Playback">
        <button className="chip small" onClick={() => setI((k) => Math.max(k - 1, 0))} aria-label="Previous (←)">◀</button>
        <button className="chip small" onClick={() => setPlaying((p) => !p)} aria-label="Play or pause (P)">{playing ? "Pause" : "Play"}</button>
        <button className="chip small" onClick={() => { setPlaying(true); setReplay((n) => n + 1); }} aria-label="Replay (R)">Replay</button>
        <button className="chip small" onClick={() => setI((k) => Math.min(k + 1, items.length - 1))} aria-label="Next (→)">▶</button>
        {[0.5, 0.75, 1].map((v) => (
          <button key={v} className={`chip small ${speed === v ? "on" : ""}`} onClick={() => setSpeed(v)} aria-pressed={speed === v}>{v}×</button>
        ))}
        <label className="inline-check"><input type="checkbox" checked={loop} onChange={(e) => setLoop(e.target.checked)} /> loop</label>
      </div>
      <div className="timeline" role="tablist" aria-label="Signs in this sentence">
        {(entries || []).map((e, k) => (
          <button key={k} role="tab" aria-selected={item?.entry === k} className={`chip ${item?.entry === k ? "on" : ""}`}
                  onClick={() => jump(k)} title={e.normalized_from ? `from "${e.normalized_from}"` : undefined}>
            {e.fingerspelled ? `${e.word.toUpperCase()} (spelled)` : e.gloss}
          </button>
        ))}
      </div>
      {showVideo && item.attribution && <p className="muted small">{item.attribution} · {item.license}</p>}
      <p className="muted small">Keys: ←/→ previous/next · R replay · P play/pause</p>
    </div>
  );
}
