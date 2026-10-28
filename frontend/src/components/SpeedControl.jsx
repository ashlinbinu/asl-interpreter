import { SPEEDS, useSpeed } from "../lib/playback.js";

/** 0.5x / 0.75x / 1x: one setting for every player, kept for the session. */
export default function SpeedControl() {
  const [speed, setSpeed] = useSpeed();
  return (
    <div className="speed" role="group" aria-label="Playback speed">
      {SPEEDS.map((v) => (
        <button key={v} className={`chip small ${speed === v ? "on" : ""}`} aria-pressed={speed === v}
                onClick={() => setSpeed(v)}>{v}×</button>
      ))}
    </div>
  );
}
