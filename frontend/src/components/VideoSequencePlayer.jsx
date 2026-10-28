import { useEffect, useMemo, useRef, useState } from "react";
import SpeedControl from "./SpeedControl.jsx";
import { mediaUrl } from "../api.js";
import { applyRate, getSpeed, scaled, useSpeed } from "../lib/playback.js";

const LETTER_MS = 900;    // a fingerspelled letter (photo or card) at 1x
const CARD_MS = 1600;     // a known sign without a reference video yet
const FADE_MS = 150;      // crossfade between clips at 1x

const media = (m) => (m.video_url ? { kind: "video", url: mediaUrl(m.video_url) }
  : m.image_url ? { kind: "photo", url: mediaUrl(m.image_url) } : null);

/**
 * Plan entries (POST /api/text-to-sign) -> playable items: one per sign, one per fingerspelled letter.
 * Real media first: video, else a photo of the handshape; without either the item is a "no video yet" card.
 */
export function sequenceItems(entries) {
  const items = [];
  (entries || []).forEach((e, entry) => {
    if (e.fingerspelled) {
      e.letters.forEach((L, letter) => items.push({
        kind: "letter", text: L.letter, ...media(L), spelled: true, entry, letter, word: e.word, ms: LETTER_MS,
        attribution: L.attribution, license: L.license,
      }));
    } else {
      items.push({ kind: "card", text: e.gloss, ...media(e), entry, word: e.word, ms: CARD_MS,
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
  const [speed] = useSpeed();
  const [loop, setLoop] = useState(false);
  const [front, setFront] = useState(0);
  const [replay, setReplay] = useState(0);
  const vids = [useRef(null), useRef(null)];
  const frontRef = useRef(0);
  const item = items[i];

  const advance = () => setI((k) => {
    if (k + 1 < items.length) return k + 1;
    if (loop) return 0;
    setPlaying(false);
    return k;
  });

  useEffect(() => { setI(0); setPlaying(true); }, [items]);

  useEffect(() => {
    if (!item) return undefined;
    onActive?.(item.entry, item.letter);
    if (item.kind === "video") {
      const back = 1 - frontRef.current;
      const v = vids[back].current;
      const start = () => {
        vids[frontRef.current].current?.pause();
        frontRef.current = back;
        setFront(back);
        applyRate(v, getSpeed());
        if (playing) v.play().catch(() => {});
      };
      if (v.dataset.src !== item.url) { v.src = item.url; v.dataset.src = item.url; }
      v.currentTime = 0;
      if (v.readyState >= 2) start(); else v.addEventListener("loadeddata", start, { once: true });
      return () => v.removeEventListener("loadeddata", start);
    }
    vids[frontRef.current].current?.pause();
    return undefined;
  }, [i, items, replay, playing]);

  // Photos and cards are timed: their time stretches with the speed, also when it changes while one is shown.
  useEffect(() => {
    if (!item || item.kind === "video" || !playing) return undefined;
    const t = setTimeout(advance, scaled(item.ms, speed));
    return () => clearTimeout(t);
  }, [i, items, replay, playing, speed]);

  useEffect(() => { vids.forEach((r) => applyRate(r.current, speed)); }, [speed]);

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
  const caption = item?.spelled ? `${item.text} (fingerspelling "${item.word}")`
    : item && item.word && item.word.toUpperCase().replace(/ /g, "-") !== item.text ? `${item.text} ("${item.word}")` : item?.text;

  return (
    <div className="seq-player" aria-roledescription="sign sequence player">
      <div className="seq-stage">
        {[0, 1].map((k) => (
          <video key={k} ref={vids[k]} muted playsInline preload="auto" onEnded={onEnded(k)}
                 style={{ transitionDuration: `${scaled(FADE_MS, speed)}ms` }}
                 className={`seq-video ${showVideo && front === k ? "on" : ""}`} aria-hidden={!(showVideo && front === k)} />
        ))}
        {item?.kind === "photo" && (
          <img className="seq-photo" src={item.url} alt={`Photo of the handshape for ${item.text}`} />
        )}
        {(item?.kind === "letter" || item?.kind === "card") && (
          <div className={`seq-card ${item.kind}`} aria-hidden="true">
            <span>{item.text}</span>
            <small>No video available yet</small>
          </div>
        )}
      </div>
      <p className="caption" aria-live="polite">{caption}</p>
      <div className="seq-controls" role="group" aria-label="Playback">
        <button className="chip small" onClick={() => setI((k) => Math.max(k - 1, 0))} aria-label="Previous (←)">◀</button>
        <button className="chip small" onClick={() => setPlaying((p) => !p)} aria-label="Play or pause (P)">{playing ? "Pause" : "Play"}</button>
        <button className="chip small" onClick={() => { setPlaying(true); setReplay((n) => n + 1); }} aria-label="Replay (R)">Replay</button>
        <button className="chip small" onClick={() => setI((k) => Math.min(k + 1, items.length - 1))} aria-label="Next (→)">▶</button>
        <SpeedControl />
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
      {(showVideo || item?.kind === "photo") && item.attribution && (
        <p className="muted small">{item.kind === "photo" ? "Photo: " : ""}{item.attribution} · {item.license}</p>
      )}
      <p className="muted small">Keys: ←/→ previous/next · R replay · P play/pause</p>
    </div>
  );
}
