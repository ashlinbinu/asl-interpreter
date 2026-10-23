import { useEffect, useRef, useState } from "react";
import { DrawingUtils, FilesetResolver, HandLandmarker } from "@mediapipe/tasks-vision";

let landmarkerPromise;
const getLandmarker = () =>
  (landmarkerPromise ??= FilesetResolver.forVisionTasks(
    "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/wasm"
  ).then((fileset) =>
    HandLandmarker.createFromOptions(fileset, {
      baseOptions: {
        modelAssetPath:
          "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task",
        delegate: "GPU",
      },
      runningMode: "VIDEO",
      numHands: 2,
    })
  ));

/** Live camera with hand tracking. Press record, sign, and onCapture({ frames, image }) fires after `seconds`. */
export default function Camera({ onCapture, seconds = 3, buttonText = "Record sign" }) {
  const video = useRef(), canvas = useRef(), rec = useRef(null);
  const [status, setStatus] = useState("Starting camera and hand tracker...");
  const [left, setLeft] = useState(0);

  useEffect(() => {
    let raf, stream, stopped = false;
    (async () => {
      try {
        const [lm, s] = await Promise.all([
          getLandmarker(),
          navigator.mediaDevices.getUserMedia({ video: { width: 640, height: 480 } }),
        ]);
        if (stopped) return s.getTracks().forEach((t) => t.stop());
        stream = s;
        video.current.srcObject = s;
        await video.current.play();
        setStatus("");
        const ctx = canvas.current.getContext("2d");
        const draw = new DrawingUtils(ctx);
        const loop = () => {
          if (stopped) return;
          const v = video.current;
          canvas.current.width = v.videoWidth;
          canvas.current.height = v.videoHeight;
          const res = lm.detectForVideo(v, performance.now());
          res.landmarks.forEach((h) => {
            draw.drawConnectors(h, HandLandmarker.HAND_CONNECTIONS, { color: "#FFE600", lineWidth: 4 });
            draw.drawLandmarks(h, { color: "#ffffff", radius: 2 });
          });
          // Raw (un-mirrored) landmarks: the mirror effect is CSS only, so data matches the dataset.
          rec.current?.push(res.landmarks.map((h) => h.map((p) => [+p.x.toFixed(4), +p.y.toFixed(4), +p.z.toFixed(4)])));
          raf = requestAnimationFrame(loop);
        };
        loop();
      } catch (e) {
        setStatus("Camera unavailable: " + e.message + ". Allow camera access and reload.");
      }
    })();
    return () => {
      stopped = true;
      cancelAnimationFrame(raf);
      stream?.getTracks().forEach((t) => t.stop());
    };
  }, []);

  const start = () => {
    rec.current = [];
    const t0 = Date.now();
    setLeft(seconds);
    const id = setInterval(() => {
      const remaining = seconds - (Date.now() - t0) / 1000;
      setLeft(Math.max(remaining, 0));
      if (remaining <= 0) {
        clearInterval(id);
        const frames = rec.current || [];
        rec.current = null;

        // Capture a frame snapshot
        let image = null;
        if (video.current && video.current.videoWidth) {
          try {
            const snap = document.createElement("canvas");
            snap.width = video.current.videoWidth;
            snap.height = video.current.videoHeight;
            const snapCtx = snap.getContext("2d");
            snapCtx.drawImage(video.current, 0, 0);
            image = snap.toDataURL("image/jpeg", 0.85);
          } catch (_) {}
        }

        onCapture({ frames, image });
      }
    }, 100);
  };

  return (
    <div className="cam">
      <div className={left > 0 ? "frame live" : "frame"}>
        <video ref={video} playsInline muted />
        <canvas ref={canvas} />
        {status && <p className="overlay">{status}</p>}
      </div>
      <button className="btn" onClick={start} disabled={left > 0 || !!status}>
        {left > 0 ? `Recording... ${left.toFixed(1)}s` : buttonText}
      </button>
    </div>
  );
}
