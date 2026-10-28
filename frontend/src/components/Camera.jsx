import { forwardRef, useEffect, useImperativeHandle, useRef, useState } from "react";
import { detectFrame, getTrackers, HAND_CONNECTIONS, POSE_UPPER } from "../lib/vision.js";
import { Segmenter } from "../lib/segmenter.js";
import { learner } from "../lib/identity.js";

const FACE_DOTS = [0, 2, 5, 9, 10];
const QUALITY_WINDOW = 30;

function drawFrame(ctx, f, { leftHanded, highlight }) {
  const { width: W, height: H } = ctx.canvas;
  ctx.clearRect(0, 0, W, H);
  if (f.pose) {
    ctx.strokeStyle = "rgba(255,255,255,0.55)";
    ctx.lineWidth = 3;
    for (const [a, b] of POSE_UPPER) {
      const p = f.pose[a], q = f.pose[b];
      if (p[3] < 0.5 || q[3] < 0.5) continue;
      ctx.beginPath(); ctx.moveTo(p[0] * W, p[1] * H); ctx.lineTo(q[0] * W, q[1] * H); ctx.stroke();
    }
    ctx.fillStyle = "#ffffff";
    for (const j of FACE_DOTS) {
      const p = f.pose[j];
      if (p[3] >= 0.5) { ctx.beginPath(); ctx.arc(p[0] * W, p[1] * H, 3, 0, 7); ctx.fill(); }
    }
  }
  f.hands.forEach((h) => {
    // MediaPipe labels assume a mirrored image: "Left" is the signer's right hand.
    const personRight = h.handedness ? h.handedness === "Left" : true;
    const dominant = leftHanded ? !personRight : personRight;
    const color = dominant ? "#FFE600" : "#3FE0FF";
    const slot = dominant ? 0 : 1;
    const bad = highlight && highlight.slot === slot ? new Set(highlight.joints || []) : null;
    ctx.lineWidth = 4;
    for (const { start, end } of HAND_CONNECTIONS) {
      ctx.strokeStyle = bad && (bad.has(start) || bad.has(end)) ? "#ff4d4d" : color;
      ctx.beginPath(); ctx.moveTo(h.lm[start][0] * W, h.lm[start][1] * H); ctx.lineTo(h.lm[end][0] * W, h.lm[end][1] * H); ctx.stroke();
    }
    h.lm.forEach((p, j) => {
      ctx.fillStyle = bad?.has(j) ? "#ff4d4d" : "#ffffff";
      ctx.beginPath(); ctx.arc(p[0] * W, p[1] * H, bad?.has(j) ? 5 : 2.5, 0, 7); ctx.fill();
    });
  });
}

function qualityWarnings(recent, { kind, luminance, fps }) {
  const w = [];
  const n = recent.length || 1;
  const withHands = recent.filter((f) => f.hands.length).length;
  if (recent.length >= 10 && withHands === 0) w.push("No hands detected: raise your hand(s) into view.");
  const edge = recent.filter((f) => f.hands.some((h) => h.lm.some(([x, y]) => x < 0.02 || x > 0.98 || y < 0.02 || y > 0.98))).length;
  if (edge / n > 0.3) w.push("Hand near the edge of the frame: step back or centre yourself.");
  if (kind === "dynamic") {
    const face = recent.filter((f) => f.pose && [0, 2, 5].every((j) => f.pose[j][3] >= 0.5)).length;
    if (recent.length >= 10 && face / n < 0.5) w.push("Face/shoulders not visible: location feedback needs them in view.");
  }
  if (luminance !== null && luminance < 60) w.push("Too dark: add light in front of you.");
  if (fps && fps < 12) w.push(`Low frame rate (${fps.toFixed(0)} fps): close other tabs or apps.`);
  return w;
}

/**
 * Continuous camera with hand + pose tracking.
 * mode: "hold" (press-and-hold button / Space), "auto" (motion-based segmentation), "continuous" (frames only),
 *       "timed" (parent calls ref.startTimed(seconds)).
 * onFrame(frame) fires for every tracked frame; onSegment({frames, video, durationMs}) when a capture ends.
 */
const Camera = forwardRef(function Camera(
  { mode = "hold", kind = "dynamic", onFrame, onSegment, recordVideo = false, holdLabel = "Hold to record",
    disabled = false, highlight = null, compact = false }, ref) {
  const video = useRef(), canvas = useRef(), streamRef = useRef(null);
  const rec = useRef(null), recorder = useRef(null), chunks = useRef([]);
  const seg = useRef(new Segmenter(kind));
  const latest = useRef({});
  latest.current = { onFrame, onSegment, mode, kind, highlight, disabled, recordVideo };
  const [status, setStatus] = useState("Starting camera and trackers…");
  const [warnings, setWarnings] = useState([]);
  const [recording, setRecording] = useState(false);
  const [fps, setFps] = useState(0);
  const [tracker, setTracker] = useState("");
  const [autoState, setAutoState] = useState("idle");

  useEffect(() => { seg.current = new Segmenter(kind); }, [kind, mode]);

  const begin = () => {
    if (rec.current || latest.current.disabled) return;
    rec.current = [];
    setRecording(true);
    if (latest.current.recordVideo && streamRef.current && window.MediaRecorder) {
      chunks.current = [];
      try {
        const mr = new MediaRecorder(streamRef.current, { mimeType: MediaRecorder.isTypeSupported("video/webm;codecs=vp9") ? "video/webm;codecs=vp9" : "video/webm" });
        mr.ondataavailable = (e) => e.data.size && chunks.current.push(e.data);
        mr.start();
        recorder.current = mr;
      } catch { recorder.current = null; }
    }
  };

  const finish = (frames, videoBlob) => {
    if (!frames?.length) return;
    latest.current.onSegment?.({ frames, video: videoBlob || null, durationMs: frames[frames.length - 1].t - frames[0].t });
  };

  const end = () => {
    const frames = rec.current;
    if (!frames) return;
    rec.current = null;
    setRecording(false);
    const mr = recorder.current;
    recorder.current = null;
    if (mr && mr.state !== "inactive") {
      mr.onstop = () => finish(frames, new Blob(chunks.current, { type: "video/webm" }));
      mr.stop();
    } else finish(frames, null);
  };

  useImperativeHandle(ref, () => ({
    startTimed: (seconds) => new Promise((resolve) => {
      begin();
      setTimeout(() => { end(); resolve(); }, seconds * 1000);
    }),
    cancel: () => { rec.current = null; setRecording(false); recorder.current?.stop(); recorder.current = null; },
  }));

  useEffect(() => {
    let raf, stopped = false, lastVideoTime = -1, lastT = 0;
    const recent = [];
    let luminance = null, fpsEma = 0, lastUi = 0, lastLum = 0, hadHand = false;
    const lumCanvas = document.createElement("canvas");
    lumCanvas.width = 32; lumCanvas.height = 24;
    (async () => {
      try {
        const [tr, s] = await Promise.all([
          getTrackers(),
          navigator.mediaDevices.getUserMedia({ video: { width: { ideal: 640 }, height: { ideal: 480 }, frameRate: { ideal: 30 } } }),
        ]);
        if (stopped) return s.getTracks().forEach((t) => t.stop());
        streamRef.current = s;
        setTracker(tr.delegate);
        video.current.srcObject = s;
        await video.current.play();
        setStatus("");
        const ctx = canvas.current.getContext("2d");
        const loop = () => {
          if (stopped) return;
          const v = video.current;
          if (v.readyState >= 2 && v.currentTime !== lastVideoTime) {
            lastVideoTime = v.currentTime;
            if (canvas.current.width !== v.videoWidth) { canvas.current.width = v.videoWidth; canvas.current.height = v.videoHeight; }
            let t = performance.now();
            if (t <= lastT) t = lastT + 1;
            const dt = lastT ? t - lastT : 33;
            lastT = t;
            fpsEma = fpsEma ? 0.9 * fpsEma + 0.1 * (1000 / dt) : 1000 / dt;
            const frame = detectFrame(tr, v, t);
            const L = latest.current;
            drawFrame(ctx, frame, { leftHanded: learner().leftHanded, highlight: L.highlight });
            recent.push(frame);
            if (recent.length > QUALITY_WINDOW) recent.shift();
            L.onFrame?.(frame);
            if (rec.current) rec.current.push(frame);
            if (L.mode === "auto" && !L.disabled) {
              const out = seg.current.push(frame);
              if (out) finish(out, null);
            }
            if (t - lastLum > 1000) {
              lastLum = t;
              const lc = lumCanvas.getContext("2d", { willReadFrequently: true });
              lc.drawImage(v, 0, 0, 32, 24);
              const d = lc.getImageData(0, 0, 32, 24).data;
              let sum = 0;
              for (let i = 0; i < d.length; i += 4) sum += 0.299 * d[i] + 0.587 * d[i + 1] + 0.114 * d[i + 2];
              luminance = sum / (d.length / 4);
            }
            // Refresh every 400 ms, and at once when a hand comes into view (clears "No hands detected").
            const hasHand = frame.hands.length > 0, handAppeared = hasHand && !hadHand;
            hadHand = hasHand;
            if (t - lastUi > 400 || handAppeared) {
              lastUi = t;
              setFps(fpsEma);
              setWarnings(qualityWarnings(recent, { kind: L.kind, luminance, fps: fpsEma }));
              setAutoState(seg.current.state);
            }
          }
          raf = requestAnimationFrame(loop);
        };
        loop();
      } catch (e) {
        setStatus("Camera or tracker unavailable: " + (e?.message || e) + ". Allow camera access and reload.");
      }
    })();
    return () => {
      stopped = true;
      cancelAnimationFrame(raf);
      streamRef.current?.getTracks().forEach((t) => t.stop());
    };
  }, []);

  useEffect(() => {
    if (mode !== "hold") return;
    const typing = (e) => ["INPUT", "TEXTAREA", "SELECT"].includes(e.target?.tagName);
    const down = (e) => { if (e.code === "Space" && !e.repeat && !typing(e)) { e.preventDefault(); begin(); } };
    const up = (e) => { if (e.code === "Space" && !typing(e)) { e.preventDefault(); end(); } };
    window.addEventListener("keydown", down);
    window.addEventListener("keyup", up);
    return () => { window.removeEventListener("keydown", down); window.removeEventListener("keyup", up); };
  }, [mode]);

  const live = recording || (mode === "auto" && autoState === "active");
  return (
    <div className={compact ? "cam compact" : "cam"}>
      <div className={live ? "frame live" : "frame"}>
        <video ref={video} playsInline muted aria-label="Your camera (mirrored)" />
        <canvas ref={canvas} aria-hidden="true" />
        {status && <p className="overlay">{status}</p>}
        {!status && <span className="fps" aria-hidden="true">{fps.toFixed(0)} fps · {tracker}</span>}
      </div>
      {warnings.length > 0 && !status && (
        <ul className="warnings" role="status">{warnings.map((w) => <li key={w}>{w}</li>)}</ul>
      )}
      {mode === "hold" && (
        <button
          className="btn"
          disabled={disabled || !!status}
          onPointerDown={(e) => { e.currentTarget.setPointerCapture?.(e.pointerId); begin(); }}
          onPointerUp={end}
          onPointerCancel={end}
          onKeyDown={(e) => { if (e.key === "Enter" && !e.repeat) begin(); }}
          onKeyUp={(e) => { if (e.key === "Enter") end(); }}
        >
          {recording ? "Recording… release to finish" : `${holdLabel} (or hold Space)`}
        </button>
      )}
      {mode === "auto" && !status && (
        <p className="note" role="status">
          {autoState === "active" ? "● Signing detected…" : kind === "static" ? "Hold the handshape still to submit." : "Start signing: capture begins when your hand moves."}
        </p>
      )}
      {mode === "timed" && recording && <p className="note" role="status">● Recording…</p>}
    </div>
  );
});

export default Camera;
