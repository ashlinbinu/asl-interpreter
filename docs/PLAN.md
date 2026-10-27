# Handspeak → ASL learning coach: audit and implementation plan

Status (2026-10-27), on branch `feature/learning-coach`:
- **P0 done.** Python↔JS feature parity is tested exactly.
- **P1 code-complete. Its quality gates are not yet measured,** because no real multi-signer alphabet data exists
  yet. The full recorder → train → serve loop was verified end to end on scratch data. **Change of plan:** the
  alphabet will be built from video, not photos ([§7](#7-alphabet-from-video)).
- **P2 (ASL Citizen word pipeline) code-complete,** tested on a fake archive (50 tests pass). Nothing has been
  downloaded: the inspection and the ingest wait for your OK ([§6](#6-open-questions-ask)).

---

## 1. Audit of the current repo (commit `98b97f1`)

### 1.1 What exists

| Area | Files | What it does today |
|---|---|---|
| API | `backend/app/main.py` | FastAPI: `/api/status`, `/api/signs`, `/api/recognize`, `/api/score`, `/api/text-to-sign`, `POST /api/signs`. Loads every sign into an in-memory dict at import time. |
| DB | `backend/app/db.py` | SQLAlchemy, two tables (`signs` with a 32×126 JSON `frames` column; `attempts` with `sign_id, score`). `create_all` at import, no migrations. |
| Features | `backend/app/features.py` | Hands only, sorted by image x, wrist-centred, scaled by max radius, resampled to 32 frames → 126 dims. |
| Scoring | `backend/app/scoring.py` | Pure-Python O(n²) DTW on cosine distance; score = `(sim−0.6)/0.4`. Four fixed feedback strings. |
| Image model | `backend/app/model.py`, `asl_model.joblib` | HOG(64×64) + calibrated LinearSVC on uploaded images. |
| References | `backend/app/sign_data.py` | 26 letters + 10 words generated procedurally (hand-coded joint positions + sine-wave "motion"). |
| Scripts | `scripts/train_and_seed.py`, `ingest_how2sign.py`, `test_system.py` | Seed synthetic signs + train HOG; How2Sign sentence ingest (legacy `mp.solutions`); a print-based "test". |
| Frontend | `frontend/src/**` | React 18 + Vite 5; MediaPipe `HandLandmarker` only; 3-second fixed recording; `SkeletonPlayer` animates the 126-dim vectors. |

Local data found: `PYTHON/Data/A..Z` (10,676 JPEGs). The DB holds 26 synthetic letters, 10 synthetic words, 2 user signs and 0 attempts.

### 1.2 Problems (verified, not just the ones listed in the brief)

1. **No real references.** Every practice and recognition comparison is against `sign_data.py` templates. That file contains no signer data at all.
2. **The alphabet dataset can't be used for landmarks.** `Data/` contains 300×300 crops on which cvzone has **already drawn a red-dot skeleton**. It's one signer, and consecutive files are near-duplicate frames. MediaPipe HandLandmarker detected a hand in **1 of 255** sampled images (raw or padded). The HOG model's 99.9% comes from a random split of near-duplicate frames of one person with a baked-in overlay, so it tells us nothing about webcam accuracy. `App.jsx` and `/api/status` hard-code "99.9%/99.94".
3. **Features lose information.** Sorting hands by x discards which hand is which and breaks under mirroring. There's no face or body context, so MOTHER (chin) and FATHER (forehead) are identical. Per-frame scaling by max radius also removes all movement except finger articulation.
4. **Scoring.** O(n²) Python loops per reference. The 0.6–1.0 stretch is a magic number. One reference per sign, no component breakdown.
5. **Recognition "fusion"** adds 0.5×HOG % to 0.5×DTW %, which are unrelated scales, keyed by upper-cased label.
6. **Camera** records a fixed 3 s after a click and sends only hands, with no timestamps, handedness or image size.
7. **Infra.** No tests (`test_system.py` hard-codes `c:\Users\hp\...\Data\A\1.jpg`, and so does `train_and_seed.py --data-dir`). `create_all` at import means no migration path. CORS is `*`, there's no payload limit, and every sign's frames are loaded at import, including on the test path. `ingest_how2sign.py` imports `app.main` (which triggers a DB load) just for `norm`.
8. **Environment constraints.** The repo lives in **OneDrive** and C: has **≈9 GB free**. The same NVMe SSD also has partitions **D: (145 GB free) and E: (146 GB free)**, which OneDrive doesn't sync. Large datasets go to `HANDSPEAK_STORE_DIR` on D:, and are still fetched selectively rather than whole. ffmpeg isn't installed (use the `imageio-ffmpeg` wheel). Python 3.12.10 venv with mediapipe 0.10.14, numpy 2.5, scikit-learn 1.9. Node 22.

---

## 2. Architecture decisions

| Decision | Choice | Why |
|---|---|---|
| Landmark models | MediaPipe Tasks **HandLandmarker (2 hands)** + **PoseLandmarker lite**, the same `.task` files in browser (`@mediapipe/tasks-vision@0.10.14`) and Python (`mediapipe==0.10.14`). SHA-256 pinned in `scripts/download_models.py`. | One model family on both sides, so parity is possible. |
| Face anchors | **Derived from Pose** (nose, eyes, ears, mouth corners → forehead/chin/cheeks by fixed geometry) in v1. FaceLandmarker is *not* in the v1 feature vector. | Pose is already running; it's more robust than face mesh when the hand covers the face (MOTHER/FATHER), and it saves ~8 ms/frame. Face mesh is reserved for non-manual markers later (feature spec v2). |
| Handedness | Assign hands to person-left/right by nearest **pose wrist**, falling back to the *flipped* MediaPipe label (labels assume a mirrored image; our frames are un-mirrored). Canonical slots = dominant, non-dominant. Left-handed signers are mirrored. | Survives mirroring, and preserves the dominant hand. |
| Feature pipeline | One spec (`docs/FEATURE_SPEC.md`), two implementations (`backend/app/features.py`, `frontend/src/lib/features.js`), **exact parity test** in pytest (runs Node on shared fixtures). Batch function over a window, so live sliding windows and offline clips run the identical code path (causal One-Euro smoothing, ≤3-frame gap fill). | Guarantees train/serve parity on the feature side. Landmark-level differences (WASM vs CPU) are measured separately with the parity-export tool. |
| Static model | RandomForest vs MLP on 93-dim hand-shape features, signer-grouped split, temperature scaling + reject threshold. Exported as compact JSON (MLP weights / RF tree arrays) for in-browser inference. | Tiny model; JSON avoids shipping onnxruntime for P1. |
| Dynamic model (P3) | PyTorch 1D-CNN → BiGRU → (Transformer only if better), T=32×270. ONNX → `onnxruntime-web` client-side; `onnxruntime` server-side. | Spec §5. |
| DTW (P3) | `numba`-jitted DTW with Sakoe-Chiba band on per-component feature groups; references precomputed at startup. | ~100× faster than today; per-component distances power the hints. |
| Storage | `HANDSPEAK_STORE_DIR` (default `backend/data`, **recommend a non-OneDrive path**): `landmarks/<source>/<clip>.npz`, `media/videos/<sign>/<clip>.mp4`, `recordings/`. Served via `/media` with HTTP Range (Starlette `FileResponse`). | Local now, S3-compatible later behind a `Storage` interface. |
| DB | Alembic migrations. `0001` = legacy baseline (auto-stamped for the existing `asl.db`), `0002` = new schema. | No data loss for the existing DB. |

### 2.1 Data model (migration 0002)

- `signs`: existing columns kept, plus `gloss`, `category` (`alphabet|word|phrase|user`), `kind` (`static|dynamic`), `hands` (`one|two`), `description`, `hint_tips` (JSON), `created_at`. `frames` becomes nullable and is **legacy only**: `frames_source` = `synthetic|legacy_user`, used only when `HANDSPEAK_DEV_SYNTHETIC=1`.
- `reference_clips`: `sign_id`, `source`, `signer_id`, `split`, `landmarks_path` (NPZ), `video_path`, `fps`, `num_frames`, `start_ms`, `end_ms`, `quality` (JSON), `license`, `attribution`, `source_url`, `is_canonical`, `created_at`.
- `reference_videos`: one canonical clean clip per sign (`sign_id` unique, `clip_id`, `path`, `width`, `height`, `duration_ms`, `license`, `attribution`, `source_url`).
- `learners`: anonymous id from localStorage, `settings` (left-handed …), `created_at`, `last_seen`.
- `attempts`: existing columns plus `learner_id`, `mode`, `components` (JSON), `hints` (JSON), `duration_ms`, `model_version`.

---

## 3. Phases

Each phase ends with the pytest suite passing, a summary, and a commit on the feature branch. Existing routes keep working until their replacements are tested.

**Change from the suggested order:** the own-data **recorder moves from P2 to P1**. The local alphabet data is unusable (§1.2-2), and signer-split evaluation plus the "≥90% on my own recordings" gate both need multi-signer alphabet data, which the recorder is the only immediate source of.

### P0: cleanup, test infra, feature spec
- `backend/app/config.py` (new): all paths, CORS origins, payload limits and flags from env/`.env`. No user paths anywhere.
- `backend/app/db.py`: new models, `init_db()` runs Alembic (no import-time side effects). `backend/alembic.ini`, `backend/migrations/` with `0001_legacy_baseline`, `0002_coach_schema`.
- `backend/app/features.py`: **rewrite** to FEATURE_SPEC v1. The old 126-dim code moves to `app/legacy/features_v0.py`; `sign_data.py` moves to `app/legacy/synthetic.py` (dev flag only).
- `frontend/src/lib/features.js` (new): JS port. `frontend/scripts/features_cli.mjs` for the parity test.
- `docs/FEATURE_SPEC.md` (new).
- `backend/app/main.py`: lifespan startup (migrate, load references/models once), env CORS, body-size limit middleware, `/api/status` from `metrics.json`, Pydantic schemas in `app/schemas.py`.
- `backend/tests/`: `conftest.py` (temp SQLite, `TestClient`), `test_features.py`, `test_feature_parity.py`, `test_api.py`, `test_migrations.py`. Delete `scripts/test_system.py`.
- `requirements.txt` pinned plus `requirements-dev.txt`; `run_tests.bat`; README updates.

### P1: alphabet with real landmarks, live feedback, recorder
- `scripts/download_models.py`: MediaPipe `.task` files, SHA-256 verified.
- `app/extract.py`: offline MediaPipe (Tasks API) for images/videos → `RawFrame` (same JSON as the browser).
- `scripts/datasets/common.py`: NPZ writer + `reference_clips` row. `ingest_alphabet_images.py` (folder-per-letter: Kaggle ASL Alphabet, Surrey/Massey layouts; `--signer-id`, `--license`; per-class detection-rate log). `ingest_recordings.py` (own recordings).
- `scripts/train_alphabet.py`: RF vs MLP, leave-signer-out when ≥2 signers (otherwise a clearly labelled within-signer block split), augmentation, per-class P/R, confusion matrix, confusable-pair report (M/N/S/T/A/E, U/V/R, K/P, G/Q), temperature scaling, reject threshold. Writes `models/alphabet/<version>/{model.joblib, browser.json, metrics.json}`. J/Z are excluded from the static model and flagged as dynamic.
- `app/recognition.py`: `/api/recognize` → top-5 `{label, confidence, sign_id, reference_video_url}`; image upload → backend MediaPipe → the same landmark model. HOG and fusion are removed.
- `/api/score` for static letters: calibrated target probability plus letter-specific handshape hints (component scoring arrives in P3).
- Frontend: `lib/vision.js` (Hand + Pose), new `Camera.jsx` (continuous tracking, hands + upper-body + face-anchor overlay, quality warnings, **hold-to-record** and **auto-segmentation**), `lib/staticModel.js` (in-browser letter inference), live top-5 on Sign→Text, live letter meter in Practice, **Record dataset** page (prompts, countdown, N reps, quality gate, saves landmarks + WebM), `scripts/record_cli.py` (OpenCV recorder).
- **Gate:** signer-held-out alphabet accuracy ≥95% and a live test ≥90% on the user's own recordings. **This can't be measured until real multi-signer data is ingested (see ASK-2).** The phase is "code-complete" until then, not "done".

### P2: word data + reference videos
Implemented (code + tests on a fake archive; nothing downloaded yet):
- `app/remote_zip.py`: seekable HTTP-range file for stdlib `zipfile` (no new dependency). It refuses servers that ignore `Range` before reading any body, and enforces a byte budget.
- `scripts/datasets/inspect_asl_citizen.py`: reads only the zip index + `splits/{train,val,test}.csv` (est. ~10 MB of the 45.9 GB). Reports per-gloss clips/signers by split, letter glosses A–Z, coverage of the 50 words, and the exact MB an ingest would transfer.
- `scripts/datasets/ingest_asl_citizen.py`: fetches only the chosen glosses' videos, extracts landmarks in parallel (`--workers`), keeps the official signer split, license and attribution per clip. `--dry-run`, `--max-gb` cap, resumable.
- `scripts/build_reference_videos.py`: one canonical clip per sign, chosen from **training-split** signers by capture quality. Copied to `<store>/media/videos/<sign_id>/`. Non-H.264 files are reported (re-encoding moves to P4).
- `scripts/dataset_report.py --write-docs`: regenerates the counts in `docs/DATASET.md`.
- Measured runtime on this laptop: ~60 ms/frame per process (hand 33 + pose 22). Throughput is 18 / 29 / 38 / 41 frames/s with 1 / 2 / 3 / 4 workers. **6 workers collapsed to ~5 frames/s:** each MediaPipe process runs ~67 threads and holds ~740 MB, so memory ran out. Use `--workers 3`. The 1,557 clips (118,667 frames) take ~55 min.

Still to do in P2:
- `scripts/datasets/ingest_wlasl.py`: downloader that skips dead URLs, trims to the annotated frame range, and reports coverage per gloss. Keeps C-UDA attribution.
- `scripts/datasets/ingest_islr.py` (optional): maps Kaggle ISLR's 543-point Holistic parquet layout (face 0–467, left hand 468–488, pose 489–521, right hand 522–542; labels already from the signer's view) into `RawFrame`.
- Word recorder prompts use the confirmed vocabulary.
- Already in place from P1: `GET /api/signs?has_video=`, `GET /api/signs/{id}` with `reference_video_url`, attribution and license; `/media` static with Range.

### P3: dynamic model, scoring, hints
Implemented (2026-10-27; tests on synthetic signs, real numbers come from `scripts.train_words`):
- `app/train_words.py` + `scripts/train_words.py`: 1D-CNN and BiGRU (torch, CPU), official ASL Citizen signer split.
  Label smoothing; augmentation by time warp, shift, time mask and noise. Validation signers choose the model and
  fit the temperature, reject threshold and score weights; test signers are only used for reporting. Weights are
  exported to JSON; `app/word_model.py` runs them in numpy (parity test vs torch: 1e-4), so torch is training-only
  and no ONNX packages are needed.
- Scoring: banded DTW against 5 references per sign from distinct *training* signers. Per-component distances
  (handshape, orientation, location, movement, two-hand) become percentiles of real same-sign attempts. A logistic
  model fitted on validation signers combines them with the classifier's target probability; its test AUC
  (correct sign vs a different sign) is in `metrics.json`. Hints come from the weakest components, with direction
  (too high/low, across the body, bigger/smaller movement).
- API: `/api/recognize` `mode: "word"` (top-k), `/api/score` for words. The UI has a live Words tab on Sign → text,
  word scoring in Practice, and a Words badge with held-out accuracy.
- Not done yet: in-browser word inference (the browser calls the API for words), and mirror augmentation.

Original plan:
- `scripts/train_words.py` (PyTorch): 1D-CNN baseline, BiGRU, optional Transformer; signer-split; label smoothing, class-balanced sampling, augmentation (rotation/scale/translation, mirror with consistent handedness swap, time-warp, landmark dropout/noise). Temperature scaling, "none" rejection threshold. ONNX export with `metrics.json`.
- `app/dtw.py` (numba, banded) and `app/scoring.py`: component distances (handshape, orientation, location, movement, two-hand symmetry) vs K references from different signers (best-of-K + mean). Final score = weighted blend + classifier target probability. Weights/thresholds fitted on the validation signers (logistic fit of "correct sign vs other sign"); calibration documented in `docs/SCORING.md` with AUC.
- Hint engine: weakest component → templated, sign-specific text ("Your hand is too low: THANK-YOU starts at the chin"), plus offending-hand/finger indices for red highlighting.
- Browser: `onnxruntime-web` word model + JS DTW on a ~1 s sliding window every ~150 ms. Live score bar + component meters.
- `POST /api/score` returns `{score, components, hints, highlight}`. `POST /api/attempts`.
- **Gates:** top-1/top-5 on held-out signers per class (reported as-is, not inflated); scoring AUC of correct vs other-sign attempts; ≤200 ms per live update, FPS documented.

### P4: Text → Sign player + sentence practice
- `app/text_plan.py`: tokenise → optional approximate English→gloss normalisation (drop articles/be-verbs/"to", light lemmatiser, multi-word glosses like THANK-YOU, clearly flagged) → known sign / fingerspelling fallback with alphabet clips → ordered entries `{gloss, word, sign_id, video_url, fingerspelled, letters[]}`.
- `VideoSequencePlayer.jsx`: two stacked `<video>` elements with ~150 ms crossfade, 0.5/0.75/1× speed, loop, timeline chips, click-to-replay, captions. Banner: "Word-by-word glossing, not ASL grammar…".
- Practice: reference video (+ skeleton overlay aligned to the learner's shoulders) | live camera; sentence mode practises word by word, then the whole sentence; side-by-side replay of the attempt.

### P5: progress, polish, evaluation
- `GET /api/progress`, Progress page (history, mastered = score ≥ 80 on 3 of the last 4 attempts, Leitner-style review queue).
- Accessibility pass (captions, keyboard: Space = hold-to-record, R = replay, ←/→ = words; reduced motion; high-contrast toggle).
- `docs/EVALUATION.md`: all metrics, confusion matrices, latency/FPS, limitations. One integration test on a sample video.

---

## 4. Proposed vocabulary (ASK-1)

Fifty words. Bold ones are minimal or near-minimal pairs that differ mainly by **location** or **movement**, which is exactly what the new features must separate.

> hello, goodbye, thank-you, please, sorry, yes, no, help, **eat**, drink, water, **mother**, **father**, friend, family, love, name, good, bad, more, finish, want, need, where, what, who, how, why, again, stop, go, come, **home**, school, work, teacher, book, time, like, know, learn, understand, **happy**, sad, tired, hungry, **deaf**, **hearing**, bathroom, sign

Pairs: MOTHER/FATHER (chin vs forehead), EAT/HOME (mouth vs mouth→cheek), HAPPY/PLEASE (handshape at the chest), DEAF/HEARING, KNOW/WHY (forehead), THANK-YOU/GOOD (chin, then forward vs to the palm). Most are in WLASL-100/300. P2 checks presence in ASL Citizen and WLASL and swaps out any word with fewer than 15 clips or fewer than 8 signers.

---

## 5. Risks

| Risk | Impact | Mitigation |
|---|---|---|
| **Alphabet data**: local set unusable. Kaggle ASL Alphabet is one signer and needs a Kaggle token. Multi-signer static sets (Surrey/Massey) may be hard to obtain. | P1 gate unmeasurable | Recorder in P1; leave-signer-out once ≥2 signers; honest "within-signer" label otherwise. |
| **Disk**: ~9 GB free on C:, repo in OneDrive | OneDrive would sync GBs; C: would fill | `HANDSPEAK_STORE_DIR` on D: (145 GB free); partial remote-zip ingest with a `--max-gb` cap; data dirs git-ignored. |
| **Licenses**: ASL Citizen (MS research, non-commercial), WLASL (C-UDA, computational/academic only), How2Sign (CC BY-NC 4.0), Kaggle sets (verify per page) | Can't be deployed commercially or publicly as-is | License + attribution stored per clip and shown in the UI. README states non-commercial. Own recordings are unrestricted. |
| Dataset video → webcam domain gap (lighting, framing, fps) | Lower live accuracy than offline | Train with own recordings; augmentation; report live metrics separately. |
| Pose-derived face anchors are approximate | Location hints off by ~½ face height | Hints use coarse regions (forehead/eyes/mouth/chin/cheek/chest). Face mesh in spec v2 if needed. |
| Browser performance (Hand + Pose + model ≥15 FPS) | Laggy feedback | GPU delegate, pose-lite, inference throttled to 150 ms, FPS shown in the UI. |
| Few signers → high-variance metrics | Misleading numbers | Report per-signer folds and confidence intervals. |
| PyTorch on Windows (~200 MB CPU wheel) | Setup friction | `requirements-train.txt`, optional. Runtime only needs onnxruntime. |

---

## 6. Open questions (ASK)

Resolved: the vocabulary is the 50 words in §4 (subject to the dataset check), `PYTHON/Data` is retired, the
alphabet comes from video with Kaggle photos as train-only supplement (§7).

Answered 2026-10-27:
- Datasets and licenses are approved, and the store is `D:\handspeak_data` (`backend/.env`).
- `h5py` and CPU PyTorch are installed (`requirements-train.txt`).
- You can't record videos, so the alphabet stays "unverified": no full-frame multi-signer letter video exists.

No open questions.

---

## 7. Alphabet from video

Goal: all 26 letters from **multi-signer video**. 24 static letters come from hold frames, and J and Z are treated
as dynamic signs. Evaluation is by held-out signer.

### 7.1 Sources, in order of use

| Source | Letters | Signers | Role |
|---|---|---|---|
| Own recordings (Record page saves WebM + landmarks) | A–Z incl. J, Z | each person you record, with consent | train + **primary test** (closest to live use) |
| ChicagoFSVid (release = 128×128 hand crops with per-frame letter labels, inside words only) | 24 static, coarticulated; J only as a hand-shape/orientation change, Z's path lost | 4 | static train + **cross-signer test of handshape only**, reported separately; needs `h5py` and a MediaPipe detection check on the crops |
| ASL Citizen | **none** (inspected: 0 letter glosses) | – | – |
| Kaggle ASL Alphabet photos | 24 static | 1 | **supplementary static train only**, never val/test |
| ChicagoFSWild(+), FSboard | phrases without letter timing | 160–260 / 147 | not now: would need forced alignment with a trained model |

### 7.2 Hold-frame extraction (`app/holds.py`, new)
1. MediaPipe per frame → RawFrames (`extract.video_frames`, unchanged).
2. Per frame: wrist speed in body units (the segmenter's measure) and **shape speed** = ‖Δ `static_vector`‖ / Δt on the
   dominant hand.
3. A **hold** is a run where both speeds are below thresholds for ≥120 ms with the hand detected throughout.
   Thresholds are fitted on own recordings (known holds) and fixed before any test-set use.
4. Isolated-letter clips: the longest hold; up to 5 frames sampled from its middle 60%. All frames of one clip share a
   group id, so augmentation and splits can't leak near-duplicates.
5. Annotated peaks (ChicagoFSVid): peak ± 2 frames at 60 fps, kept only if they pass the hold test.
6. Quality gates: hand score ≥0.7, not touching the frame edge, stable handedness. Per-letter yield is reported.

### 7.3 J and Z
Each is a whole motion segment: the isolated clip trimmed by the segmenter's motion start/stop, or, inside words, from
the midpoint after the previous letter's peak to the midpoint before the next. They become T=32 sequences for the
dynamic model (P3) next to the words. Until then, Practice shows their reference and says "not scored yet".

### 7.4 Splits, metrics and the "unverified" rule
- Groups = `source:signer`. Test signers come from video sources only.
- ≥5 video signers: a fixed signer split. 3–4: leave-one-signer-out, reported as mean and range.
- Photo sources carry `supplementary=true` and are never used for validation or test. Metrics are reported with and
  without them.
- `metrics.json` gets `verified = signer-held-out protocol and ≥3 distinct video signers`. Below that, `/api/status`
  and the UI badge say **"unverified"** next to any number.
- Reported separately: per source, per signer, and own recordings (the live ≥90% gate).

### 7.5 Implementation steps
1. **Done:** `train_static.py` keeps supplementary sources train-only, refuses a photo signer as test signer,
   adds `verified`, per-signer accuracy, leave-one-signer-out (3–6 video signers) and a without-photos ablation to
   `metrics.json`. The UI badge says "unverified".
2. **Done:** `app/holds.py` (centred ±50 ms speeds, so jitter isn't mistaken for motion) + tests. Training now
   learns static letters only from the middle 60% of each hold in a video clip.
3. **Done:** Kaggle photos ingested as supplementary (`ingest_alphabet_images` marks every photo clip).
4. **Done:** `ingest_chicago_fsvid.py`.
   - Probe on raw crops: MediaPipe detects a hand in 14% of them, with inconsistent left/right labels.
   - Fix: tint each crop to a skin tone on a light background, upscale ×2 and pad. Detection rises to 49%.
   - Landmarks are visibly right for some letters (F, S, O, E, M) and wrong for others (G, N, I).
   - So clips are tagged `domain=hand_crop`, filtered (handedness score ≥0.8, majority label per signer), and
     trained on. Their cross-signer accuracy is reported separately and never counts toward "verified".
5. **Not possible now:** recordings of at least 3 people (you can't record). That is the only source of
   full-frame multi-signer letter video, and of J/Z motion, so the alphabet badge stays "unverified".
6. J/Z sequences go into the P3 dynamic dataset.
