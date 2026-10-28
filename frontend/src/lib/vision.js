// MediaPipe Tasks trackers (Hand x2 + Pose lite) producing RawFrames (docs/FEATURE_SPEC.md §1).
// Versions and model URLs are pinned and identical to backend/app/mediapipe_models.py.
import { FilesetResolver, HandLandmarker, PoseLandmarker } from "@mediapipe/tasks-vision";

export const TASKS_VISION_VERSION = "0.10.14";
const WASM = `https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@${TASKS_VISION_VERSION}/wasm`;
const HAND_MODEL = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task";
const POSE_MODEL = "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task";
const POSE_KEEP = 17; // indices 0-16: face points, shoulders, elbows, wrists
// Pose runs on every POSE_EVERY-th frame and is reused in between: face and shoulders move little between frames,
// and pose costs about as much as hand tracking. Hands are tracked on every frame.
const POSE_EVERY = 2;
// On a software renderer the GPU delegate is several times slower than the CPU delegate (measured ~250 vs ~70 ms).
const SOFTWARE_GL = /swiftshader|basic render|llvmpipe|software/i;

export const HAND_CONNECTIONS = HandLandmarker.HAND_CONNECTIONS;
export const POSE_UPPER = [[11, 12], [11, 13], [13, 15], [12, 14], [14, 16], [9, 10]];

function gpuUsable() {
  const gl = document.createElement("canvas").getContext("webgl2");
  const info = gl?.getExtension("WEBGL_debug_renderer_info");
  return !!gl && (!info || !SOFTWARE_GL.test(gl.getParameter(info.UNMASKED_RENDERER_WEBGL)));
}

async function create(Cls, fileset, modelAssetPath, extra, gpu) {
  const opts = (delegate) => ({ baseOptions: { modelAssetPath, delegate }, runningMode: "VIDEO", ...extra });
  if (gpu) {
    try {
      return [await Cls.createFromOptions(fileset, opts("GPU")), "GPU"];
    } catch { /* fall through to CPU */ }
  }
  return [await Cls.createFromOptions(fileset, opts("CPU")), "CPU"];
}

let trackers;
export function getTrackers() {
  trackers ??= (async () => {
    const fileset = await FilesetResolver.forVisionTasks(WASM);
    const gpu = gpuUsable();
    const [[hand, delegate], [pose]] = await Promise.all([
      create(HandLandmarker, fileset, HAND_MODEL, { numHands: 2, minHandDetectionConfidence: 0.5,
        minHandPresenceConfidence: 0.5, minTrackingConfidence: 0.5 }, gpu),
      create(PoseLandmarker, fileset, POSE_MODEL, { numPoses: 1 }, gpu),
    ]);
    return { hand, pose, delegate, n: 0, lastPose: null };
  })().catch((e) => { trackers = undefined; throw e; });
  return trackers;
}

const r5 = (v) => Math.round(v * 1e5) / 1e5;

/** One RawFrame from the (un-mirrored) video element. tMs must increase strictly. */
export function detectFrame(tr, video, tMs) {
  const hr = tr.hand.detectForVideo(video, tMs);
  const labels = hr.handedness ?? hr.handednesses ?? [];
  const hands = hr.landmarks.map((lm, i) => ({
    handedness: labels[i]?.[0]?.categoryName ?? null,
    score: r5(labels[i]?.[0]?.score ?? 0),
    lm: lm.map((p) => [r5(p.x), r5(p.y), r5(p.z)]),
  }));
  if (tr.n++ % POSE_EVERY === 0) {
    const pl = tr.pose.detectForVideo(video, tMs).landmarks?.[0];
    tr.lastPose = pl ? pl.slice(0, POSE_KEEP).map((p) => [r5(p.x), r5(p.y), r5(p.z), r5(p.visibility ?? 1)]) : null;
  }
  return {
    t: Math.round(tMs * 1000) / 1000,
    w: video.videoWidth,
    h: video.videoHeight,
    hands,
    pose: tr.lastPose,
  };
}
