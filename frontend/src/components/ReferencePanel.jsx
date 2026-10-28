import { useEffect, useRef, useState } from "react";
import SkeletonPlayer from "./SkeletonPlayer.jsx";
import SpeedControl from "./SpeedControl.jsx";
import { api, mediaUrl } from "../api.js";
import { applyRate, getSpeed, useSpeed } from "../lib/playback.js";

function missingReason(sign) {
  if (sign.category === "alphabet" && sign.kind === "dynamic") {
    return `${sign.label} is a motion letter and needs a video; none from real signers is available yet.`;
  }
  if (sign.category === "user") return "Your own sign has a landmark recording only (use “Show landmarks”).";
  return "No real recording of this sign is available yet.";
}

/**
 * The reference for learning a sign: the real video of a human signer, else a real photo of the handshape (letters),
 * else a clear "No video available yet" placeholder. The landmark skeleton is an optional secondary view.
 */
export default function ReferencePanel({ signId, sign: given, compact = false }) {
  const [sign, setSign] = useState(given || null);
  const [speed] = useSpeed();
  const video = useRef(null);
  const [mirror, setMirror] = useState(false);
  const [skeleton, setSkeleton] = useState(false);
  const [err, setErr] = useState("");

  useEffect(() => {
    if (given) { setSign(given); return; }
    if (!signId) return;
    setErr("");
    api.sign(signId).then(setSign).catch((e) => setErr(e.message));
  }, [signId, given]);

  useEffect(() => applyRate(video.current, speed), [speed, sign?.reference_video_url]);

  if (err) return <p className="error">{err}</p>;
  if (!sign) return <p className="muted">Loading reference…</p>;
  const ref = sign.reference_landmarks;
  const size = compact ? 220 : 280;
  const moving = !!sign.reference_video_url || (skeleton && (ref?.frames || sign.frames || []).length > 1);
  const keepRate = (e) => applyRate(e.currentTarget, getSpeed());
  return (
    <figure className="reference">
      {sign.reference_video_url ? (
        <>
          <video src={mediaUrl(sign.reference_video_url)} controls loop muted playsInline autoPlay
                 style={{ maxWidth: size * 1.4 }} aria-label={`Reference video for ${sign.label}`}
                 ref={video} onLoadedMetadata={keepRate} onPlay={keepRate} />
          <figcaption className="muted small">{sign.attribution || "Reference video"} · {sign.license}</figcaption>
        </>
      ) : sign.reference_image_url ? (
        <>
          <img src={mediaUrl(sign.reference_image_url)} className="ref-photo" style={{ maxWidth: size }}
               alt={`Photo of the handshape for ${sign.label}`} />
          <figcaption className="muted small">Photo: {sign.image_attribution} · {sign.image_license}</figcaption>
        </>
      ) : (
        <div className="media-missing" role="img" aria-label={`No video available yet for ${sign.label}`}
             style={{ maxWidth: size * 1.4 }}>
          <b>No video available yet</b>
          <span>{missingReason(sign)}</span>
        </div>
      )}
      {(ref || sign.frames) && (
        <div className="skeleton-toggle">
          <button className="chip small" aria-pressed={skeleton} onClick={() => setSkeleton((v) => !v)}>
            {skeleton ? "Hide landmarks" : "Show landmarks"}
          </button>
          {skeleton && (ref ? (
            <>
              <SkeletonPlayer frames={ref.frames} size={size} mirror={mirror} label={`Reference landmarks for ${sign.label}`} />
              <p className="muted small">
                Real landmarks: {ref.source} signer {ref.signer_id}{ref.license ? ` · ${ref.license}` : ""}
                <label className="inline-check"><input type="checkbox" checked={mirror} onChange={(e) => setMirror(e.target.checked)} /> mirror</label>
              </p>
            </>
          ) : (
            <>
              <SkeletonPlayer frames={sign.frames} size={size} label={`Synthetic placeholder for ${sign.label}`} />
              <p className="warn small">DEV synthetic placeholder: not real signing.</p>
            </>
          ))}
        </div>
      )}
      {moving && <SpeedControl />}
      {sign.description && <p className="desc">{sign.description}</p>}
    </figure>
  );
}
