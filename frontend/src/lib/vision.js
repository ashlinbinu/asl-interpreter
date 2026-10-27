// MediaPipe Tasks trackers (Hand x2 + Pose lite) producing RawFrames (docs/FEATURE_SPEC.md §1).
// Versions and model URLs are pinned and identical to backend/app/mediapipe_models.py.
import { FilesetResolver, HandLandmarker, PoseLandmarker } from "@mediapipe/tasks-vision";

export const TASKS_VISION_VERSION = "0.10.14";
const WASM = `https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@${TASKS_VISION_VERSION}/wasm`;
const HAND_MODEL = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task";
const POSE_MODEL = "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task";
const POSE_KEEP = 17; // indices 0-16: face points, shoulders, elbows, wrists

export const HAND_CONNECTIONS = HandLandmarker.HAND_CONNECTIONS;
export const POSE_UPPER = [[11, 12], [11, 13], [13, 15], [12, 14], [14, 16], [9, 10]];

async function create(Cls, fileset, modelAssetPath, extra) {
  const opts = (delegate) => ({ baseOptions: { modelAssetPath, delegate }, runningMode: "VIDEO", ...extra });
  try {
    return await Cls.createFromOptions(fileset, opts("GPU"));
  } catch {
    return Cls.createFromOptions(fileset, opts("CPU"));
  }
}

let trackers;
export function getTrackers() {
  trackers ??= (async () => {
    const fileset = await FilesetResolver.forVisionTasks(WASM);
    const [hand, pose] = await Promise.all([
      create(HandLandmarker, fileset, HAND_MODEL, { numHands: 2, minHandDetectionConfidence: 0.5,
        minHandPresenceConfidence: 0.5, minTrackingConfidence: 0.5 }),
      create(PoseLandmarker, fileset, POSE_MODEL, { numPoses: 1 }),
    ]);
    return { hand, pose };
  })().catch((e) => { trackers = undefined; throw e; });
  return trackers;
}

const r5 = (v) => Math.round(v * 1e5) / 1e5;

/** One RawFrame from the (un-mirrored) video element. tMs must increase strictly. */
export function detectFrame({ hand, pose }, video, tMs) {
  const hr = hand.detectForVideo(video, tMs);
  const labels = hr.handedness ?? hr.handednesses ?? [];
  const hands = hr.landmarks.map((lm, i) => ({
    handedness: labels[i]?.[0]?.categoryName ?? null,
    score: r5(labels[i]?.[0]?.score ?? 0),
    lm: lm.map((p) => [r5(p.x), r5(p.y), r5(p.z)]),
  }));
  const pr = pose.detectForVideo(video, tMs);
  const pl = pr.landmarks?.[0];
  return {
    t: Math.round(tMs * 1000) / 1000,
    w: video.videoWidth,
    h: video.videoHeight,
    hands,
    pose: pl ? pl.slice(0, POSE_KEEP).map((p) => [r5(p.x), r5(p.y), r5(p.z), r5(p.visibility ?? 1)]) : null,
  };
}
