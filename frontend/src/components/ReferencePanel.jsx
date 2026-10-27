import { useEffect, useState } from "react";
import SkeletonPlayer from "./SkeletonPlayer.jsx";
import { api, mediaUrl } from "../api.js";

/** Shows the best real reference for a sign: video (with attribution) > real landmark recording > description. */
export default function ReferencePanel({ signId, sign: given, compact = false, speed = 1 }) {
  const [sign, setSign] = useState(given || null);
  const [mirror, setMirror] = useState(false);
  const [err, setErr] = useState("");

  useEffect(() => {
    if (given) { setSign(given); return; }
    if (!signId) return;
    setErr("");
    api.sign(signId).then(setSign).catch((e) => setErr(e.message));
  }, [signId, given]);

  if (err) return <p className="error">{err}</p>;
  if (!sign) return <p className="muted">Loading reference…</p>;
  const ref = sign.reference_landmarks;
  const size = compact ? 220 : 280;
  return (
    <figure className="reference">
      {sign.reference_video_url ? (
        <>
          <video src={mediaUrl(sign.reference_video_url)} controls loop muted playsInline autoPlay
                 style={{ maxWidth: size * 1.4 }} aria-label={`Reference video for ${sign.label}`}
                 ref={(v) => { if (v) v.playbackRate = speed; }} />
          <figcaption className="muted small">{sign.attribution || "Reference video"} · {sign.license}</figcaption>
        </>
      ) : ref ? (
        <>
          <SkeletonPlayer frames={ref.frames} size={size} mirror={mirror} speed={speed} label={`Reference landmarks for ${sign.label}`} />
          <figcaption className="muted small">
            Real landmarks: {ref.source} signer {ref.signer_id}{ref.license ? ` · ${ref.license}` : ""}
            <label className="inline-check"><input type="checkbox" checked={mirror} onChange={(e) => setMirror(e.target.checked)} /> mirror</label>
          </figcaption>
        </>
      ) : sign.frames ? (
        <>
          <SkeletonPlayer frames={sign.frames} size={size} label={`Synthetic placeholder for ${sign.label}`} />
          <figcaption className="warn small">DEV synthetic placeholder: not real signing.</figcaption>
        </>
      ) : (
        <p className="muted">No reference recording yet for <b>{sign.label}</b>. Add one on the Record page.</p>
      )}
      {sign.description && <p className="desc">{sign.description}</p>}
    </figure>
  );
}
