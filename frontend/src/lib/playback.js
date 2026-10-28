// One playback speed for every player (reference videos, skeletons, the sign sequence), kept for the session.
import { useSyncExternalStore } from "react";

export const SPEEDS = [0.5, 0.75, 1];
const KEY = "handspeak.speed";
const listeners = new Set();

function load() {
  try {
    const v = Number(globalThis.sessionStorage?.getItem(KEY));
    return SPEEDS.includes(v) ? v : 1;
  } catch {
    return 1;
  }
}

let current = load();

export const getSpeed = () => current;

export function setSpeed(v) {
  if (!SPEEDS.includes(v) || v === current) return;
  current = v;
  try { globalThis.sessionStorage?.setItem(KEY, String(v)); } catch { /* storage unavailable: memory only */ }
  listeners.forEach((f) => f());
}

export function subscribe(f) {
  listeners.add(f);
  return () => listeners.delete(f);
}

/** [speed, setSpeed] shared by every component that uses it. */
export function useSpeed() {
  return [useSyncExternalStore(subscribe, getSpeed), setSpeed];
}

/** A duration at 1x (card time, crossfade, frame interval) stretched for the given speed: 0.5x -> twice as long. */
export const scaled = (ms, speed) => ms / speed;

/** Apply the speed to a <video> so it survives new sources, loops and restarts (the load algorithm resets
 * playbackRate to defaultPlaybackRate, so both are set). */
export function applyRate(video, speed) {
  if (!video) return;
  video.defaultPlaybackRate = speed;
  video.playbackRate = speed;
}

/**
 * Frame to show for a looping skeleton after `mediaMs` of media time (wall time x speed).
 * RawFrames are placed by their own timestamps; legacy frames have no time and step every `stepMs`.
 * A `pauseMs` gap separates loops.
 */
export function skeletonFrameIndex(frames, mediaMs, { stepMs = 80, pauseMs = 600 } = {}) {
  if (!frames?.length) return -1;
  const timed = frames[0] && !Array.isArray(frames[0]) && typeof frames[0].t === "number";
  const t0 = timed ? frames[0].t : 0;
  const dur = timed ? Math.max(frames[frames.length - 1].t - t0, 1) : frames.length * stepMs;
  const el = mediaMs % (dur + pauseMs);
  if (!timed) return Math.min(Math.floor(el / stepMs), frames.length - 1);
  const target = t0 + Math.min(el, dur);
  const i = frames.findIndex((g) => g.t >= target);
  return i < 0 ? frames.length - 1 : i;
}
