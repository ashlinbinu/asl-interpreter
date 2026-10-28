# Feature spec v1.0

This is the single definition of the landmark → feature pipeline. It has two implementations that must agree
(`tests/test_feature_parity.py` runs both on shared fixtures; tolerance 1e-6):

- Python: `backend/app/features.py`, used for training, offline ingest and the API.
- JavaScript: `frontend/src/lib/features.js`, used for live in-browser inference.

Any change here bumps `SPEC_VERSION` in both files, and models trained on an older version refuse to load.

## 0. Constants

| Name | Value | Meaning |
|---|---|---|
| `SPEC_VERSION` | `"1.0"` | |
| `T` | 32 | resampled length for classifiers |
| `F` | 270 | per-frame feature length |
| `GAP_MAX` | 3 | longest run of missing frames that is interpolated |
| One-Euro | `min_cutoff=1.5 Hz`, `beta=5.0`, `d_cutoff=1.0 Hz` | causal smoothing in canonical coordinates |
| `VIS_MIN` | 0.5 | pose visibility needed for shoulders, elbows, face points |
| `WRIST_VIS_MIN` | 0.3 | pose wrist visibility needed for hand-to-side assignment |

## 1. Input: `RawFrame`

```jsonc
{
  "t": 1234.5,                 // timestamp in ms (monotonic within a clip)
  "w": 640, "h": 480,          // source image size in px; only the aspect ratio is used
  "hands": [                   // 0–2 hands from MediaPipe HandLandmarker
    { "handedness": "Left",    // MediaPipe label (assumes a MIRRORED image) or null
      "score": 0.97,
      "lm": [[x, y, z], ...] } // 21 normalised image landmarks
  ],
  "pose": [[x, y, z, vis], ...] // ≥17 MediaPipe Pose landmarks (indices 0–16 are used) or null
}
```

Frames come from the **un-mirrored** camera image; mirroring is display-only.
Legacy frames (a bare list of hands, each 21×[x,y,z]) are accepted as `{t: i·1000/30, w: 640, h: 480, hands: [{handedness: null, lm}], pose: null}`.

## 2. Canonicalisation (per frame)

1. Aspect `a = w/h`. Hand points → `(x·a, y, z·a)`. Pose points → `(x·a, y)` with `vis`.
2. **Side assignment** (person's anatomical Left/Right):
   - If pose is present and **both** pose wrists (15 = left, 16 = right) have `vis ≥ WRIST_VIS_MIN`:
     one hand → nearest pose wrist (2-D distance from hand landmark 0); two hands → the pairing with
     the smaller summed distance (ties → hand 0 gets Right).
   - Otherwise use the label: `"Left"` → person Right, `"Right"` → person Left (MediaPipe assumes a mirrored image).
     If the label is null or both hands get the same side: with two hands, the hand with the smaller x is Right and the
     other Left; one unlabelled hand is Right.
3. **Left-handed signers** (`left_handed=true`): `x → a − x` for every point, swap pose pairs
   (1,4) (2,5) (3,6) (7,8) (9,10) (11,12) (13,14) (15,16), and swap Left/Right sides.
4. Slots: **slot 0 = dominant = Right** (after step 3), **slot 1 = non-dominant = Left**.

## 3. Gap fill

For each stream (slot 0 hand, slot 1 hand, pose), a run of missing frames `i..j` with present neighbours
`i−1` and `j+1` and length ≤ `GAP_MAX` is filled by linear interpolation in time:
`α = (t_k − t_{i−1}) / (t_{j+1} − t_{i−1})` (falls back to index ratio if the time span is 0). Pose
visibilities are interpolated too. Filled frames count as present.

## 4. Smoothing: One-Euro, causal

Per stream and per scalar coordinate (hands: 63 per slot; pose: x and y of 17 points; not `vis`), iterate frames in order:
- present and (first frame or previous frame missing) → reset: `x̂ = x`, `dx̂ = 0`.
- present otherwise: `dt = t_i − t_{i−1}` in seconds (`1/30` if ≤ 0);
  `dx = (x − x̂)/dt`; `dx̂ = α(dt, d_cutoff)·dx + (1−α)·dx̂`; `fc = min_cutoff + beta·|dx̂|`; `x̂ = α(dt, fc)·x + (1−α)·x̂`,
  where `α(dt, fc) = 1 / (1 + τ/dt)` and `τ = 1/(2π·fc)`.

## 5. Body frame (one per sequence/window)

Valid frames: pose present, `vis[11] ≥ VIS_MIN`, `vis[12] ≥ VIS_MIN`. For each: `L = P11`, `R = P12`,
`origin = (L+R)/2`, `scale = |L−R|`, `roll = atan2(L.y−R.y, L.x−R.x)`. The body frame is the **median** of each
quantity over the valid frames (average of the two middle values for even counts). No valid frames or
`scale < 1e-6` → no body frame, and location/motion features are zero.

`toBody(p) = R(−roll)·(p − origin)/scale`, i.e. with `c = cos roll`, `s = sin roll`, `d = p − origin`:
`bx = (d.x·c + d.y·s)/scale`, `by = (−d.x·s + d.y·c)/scale`. Body x points to the signer's left
(image right), body y points down, and the unit is one shoulder width.
3-D hand vectors are rotated the same way in x/y (z unchanged; roll = 0 without a body frame).

## 6. Anchors (per frame, body coordinates)

Face is visible when the pose is present and `vis` ≥ `VIS_MIN` for points 0, 2, 5, 9 and 10.
With `eyes = (P2+P5)/2`, `mouth = (P9+P10)/2`, `up = eyes − mouth`:

| # | anchor | definition |
|---|---|---|
| 0 | forehead | `eyes + 0.8·up` |
| 1 | eye_d | P5 |
| 2 | eye_nd | P2 |
| 3 | nose | P0 |
| 4 | mouth | `mouth` |
| 5 | chin | `mouth − 0.7·up` |
| 6 | cheek_d | `(P10 + P8)/2` |
| 7 | cheek_nd | `(P9 + P7)/2` |
| 8 | ear_d | P8 |
| 9 | ear_nd | P7 |
| 10 | chest | body point `(0, 0.6)` (needs only the body frame) |

`_d` = dominant side, `_nd` = non-dominant side.

## 7. Per-frame vector (F = 270)

### Hand block, 128 values; slot 0 at offset 0, slot 1 at offset 128 (all zeros if the hand is absent)

`p` = smoothed canonical points, `r = rotXY(p − p0, −roll)`, `s_h = max(|r9|, 1e-6)`, chirality `c = +1` (slot 0) / `−1` (slot 1).

| offset | n | content |
|---|---|---|
| 0 | 1 | present flag |
| 1 | 63 | `r / s_h`, point-major (x0,y0,z0,x1,…) |
| 64 | 15 | flexion angle / π at joints 1, 2, 3 of the chains thumb [0,1,2,3,4], index [0,5,6,7,8], middle [0,9,10,11,12], ring [0,13,14,15,16], pinky [0,17,18,19,20]; angle between `prev−joint` and `next−joint`. All angles use `acos(clamp(u·v / max(|u||v|, 1e-12), −1, 1))`. |
| 79 | 4 | spread / π between finger directions thumb `r4−r2`, index `r8−r5`, middle `r12−r9`, ring `r16−r13`, pinky `r20−r17` (adjacent pairs) |
| 83 | 5 | `|r4−r8|, |r4−r12|, |r4−r16|, |r4−r20|, |r8−r12|` / `s_h` |
| 88 | 3 | palm normal `c · unit(r5 × r17)` (zeros if degenerate) |
| 91 | 3 | hand direction `unit(r9)` |
| 94 | 6 | body coords of wrist (p0), index tip (p8), palm centre (mean of p0, p5, p9, p13, p17) |
| 100 | 22 | distance from index tip (11 values), then palm centre (11 values), to anchors 0–10 (face anchors are 0 when the face isn't visible) |
| 122 | 6 | velocity (body units/s) of wrist, index tip, palm centre: `(loc_i − loc_{i−1})/dt`; 0 if the hand is missing at i−1 or i == 0 |

Offsets 94–127 are 0 without a body frame.

### Global block, 14 values at offset 256

| offset | n | content |
|---|---|---|
| 256 | 1 | body flag (body frame exists, pose present, shoulders visible in this frame) |
| 257 | 1 | face flag |
| 258 | 8 | body coords of dominant shoulder P12, non-dominant shoulder P11, dominant elbow P14, non-dominant elbow P13 (elbows 0 if `vis < VIS_MIN`); all 0 when the body flag is 0 |
| 266 | 4 | both hands + body: `wrist0 − wrist1` (2), its length (1), `|indextip0 − palm1|` (1) |

### Component groups (used by scoring)
- handshape: hand offsets 1–87; orientation: 88–93; location: 94–121 + 258–265; movement: 122–127; two-hand: 266–269.

## 8. Resampling to T = 32

1. Keep frames from the first to the last frame with any hand present; fewer than 4 hand-present frames → *insufficient*.
2. `u_i = (t_i − t_first)/(t_last − t_first)` (index ratio if the span is 0); targets `u_k = k/(T−1)`.
3. For each target, find the bracketing frames `i ≤ k ≤ i+1` (the last `i` with `u_i ≤ u_k`, clamped to `N−2`).
   If the flags at offsets 0, 128, 256 and 257 are all equal in `i` and `i+1`, interpolate linearly; otherwise take
   the nearer frame (`α < 0.5` → `i`).

## 9. Static (alphabet) vector, 93 values

From one per-frame vector: use slot 0 if present, else slot 1 **mirrored to right chirality**.
Mirroring negates the x components of the 21 shape points, the normal and the direction. Value = hand block offsets `1..93`.
Frames with no hand produce no static vector.
