# Handspeak: ASL learning coach

Handspeak is a local web app for **learning** American Sign Language. You can type text and see each sign,
practise with live feedback from your webcam, and sign to the camera to get the nearest matches.

> **Honest scope.** Handspeak is a learning aid, not a certified ASL translator. Scores are estimates. Text → sign
> is word-by-word glossing, not ASL grammar, and facial grammar (non-manual markers) is only partly captured. Several
> datasets it can use are **non-commercial** (see [docs/DATASET.md](docs/DATASET.md)).

Status: phases P0–P1 of [docs/PLAN.md](docs/PLAN.md) are implemented. The alphabet pipeline needs real data before
its accuracy can be measured (see "Getting data").

## How it works

- The **browser** runs MediaPipe Hand (2 hands) + Pose landmarkers and computes features per
  [docs/FEATURE_SPEC.md](docs/FEATURE_SPEC.md). The features cover hand shape, handedness, location relative to
  face/body, and motion. Letters are recognised live in the browser.
- The **backend** (FastAPI) uses the same MediaPipe models and the *same* feature spec (a parity test keeps the
  Python and JS pipelines identical). It stores real reference clips, scores attempts and tracks progress.
- **References are real signer data only**: your own recordings, or ingested datasets. The old procedurally generated
  templates survive only as a dev fallback (`HANDSPEAK_DEV_SYNTHETIC=1`).

| Page | What it does now |
|---|---|
| Sign → text | Live fingerspelling recognition (top-5 with calibrated confidence), image upload, "Learn" → reference + practice |
| Text → sign | Word/gloss timeline with fingerspelling fallback; shows the real reference for each item |
| Practice | Pick a sign → reference (real landmarks/video) beside your camera; live match meter for letters; score + specific hints |
| Record dataset | Guided recorder: prompts, countdown, N repetitions, quality check; saves landmarks + video |
| Add a sign | Record a custom sign (e.g. a name sign) as a reference clip |

Moving signs (J, Z, words) are scored in phase P3, once real word clips and the dynamic model exist. The stitched
reference-video player is phase P4.

## Requirements

- Windows, macOS or Linux. **Python 3.12** (3.10–3.12 work; `mediapipe==0.10.14` has no wheels for newer versions).
- Node.js 18+ and npm.
- A webcam. The browser needs camera permission on `localhost` or HTTPS, plus internet on first load
  (MediaPipe WASM/models come from pinned CDN URLs).

## Quick start (Windows PowerShell, from the repo root)

```powershell
# 1. Backend
cd backend
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1          # if blocked: Set-ExecutionPolicy -Scope Process Bypass
python -m pip install -r requirements-dev.txt
python -m scripts.download_models     # MediaPipe .task files (SHA-256 verified) for server-side extraction
python -m uvicorn app.main:app --reload   # http://127.0.0.1:8000  (docs at /docs)

# 2. Frontend (second terminal)
cd frontend
npm install
npm run dev                           # http://localhost:5173
```

`run_all.bat` starts both after installation, and `run_tests.bat` runs the test suite.

On first start the backend migrates the database to the latest schema (Alembic). An existing `asl.db` from the
previous version is upgraded in place: the old templates are kept but tagged `synthetic`.

## Getting data (needed for real accuracy)

1. **Record yourself (best data).** Open *Record dataset*, choose a signer id (e.g. `me_01`), pick letters, set
   5–10 repetitions and press Space for each take. Record other people under their own ids, with their consent.
   For a CLI alternative: `python -m scripts.record_cli --signer-id me_01 --alphabet --reps 5`.
2. **Optional public alphabet images.** Download a folder-per-letter dataset, e.g. Kaggle *ASL Alphabet*
   (`grassknoted/asl-alphabet`; confirm its license on the page), then:

   ```powershell
   python -m scripts.datasets.ingest_alphabet_images --root D:\datasets\asl_alphabet_train `
       --source kaggle_asl_alphabet --signer-id kaggle_1 --license "<license from the dataset page>" --per-class 500
   ```

   This prints the per-class hand-detection rate and drops images without a detected hand.
   *Images with landmarks already drawn on them (e.g. cvzone exports) don't work: MediaPipe can't detect hands in them.*
3. **Train the alphabet model:**

   ```powershell
   python -m scripts.train_alphabet --test-signers own:me_01
   ```

   It compares RandomForest and MLP on **signer-held-out** data, calibrates the probabilities and writes
   `models/alphabet/<version>/{model.joblib,browser.json,metrics.json}`. With a single signer it falls back to a block
   split and says so (that number overstates live accuracy). Restart the API to load the new model.

Keep large data outside OneDrive: set `HANDSPEAK_STORE_DIR` (see below).

## Configuration (`backend/.env`, all optional)

| Variable | Default | Purpose |
|---|---|---|
| `DATABASE_URL` | SQLite `backend/asl.db` | or a PostgreSQL/Supabase URI |
| `HANDSPEAK_STORE_DIR` | `backend/data` | landmark NPZ files, media, recordings (put it outside OneDrive) |
| `HANDSPEAK_MODELS_DIR` | `backend/models` | trained models + MediaPipe `.task` files |
| `CORS_ORIGINS` | `http://localhost:5173,http://127.0.0.1:5173` | allowed browser origins |
| `HANDSPEAK_MAX_BODY_BYTES` / `HANDSPEAK_MAX_UPLOAD_BYTES` | 8 MB / 60 MB | request size limits |
| `HANDSPEAK_DEV_SYNTHETIC` | `0` | dev-only fallback to the old synthetic templates |
| `DATA_DIR` | – | optional default raw dataset folder |

Frontend: `VITE_API_URL` (default `/api`, proxied by Vite together with `/media`). `HANDSPEAK_BACKEND` changes the
proxy target.

## Tests

```powershell
cd backend; python -m pytest
```

The suite covers feature geometry (invariance, handedness, mirroring, gap fill, location), **Python↔JS feature
parity**, JS↔Python model parity, migrations (including upgrading the legacy DB), the training splits (no signer
leakage), API schemas/limits/CORS/Range requests, recordings, progress, and auto-segmentation. Node-based tests are
skipped if `node` isn't on PATH. MediaPipe tests are skipped until `scripts.download_models` has run.

## API (all under `/api`; full schemas at `/docs`)

| Method | Route | Purpose |
|---|---|---|
| GET | `/status` | DB counts, model versions + held-out metrics from `metrics.json`, feature spec version |
| GET | `/signs?q=&category=&kind=&has_video=` | vocabulary |
| GET | `/signs/{id}` | sign + reference video/landmarks, license, attribution |
| POST | `/signs` | add a custom sign (RawFrames) |
| POST | `/recognize` | `{frames}` or `{image}` → top-k `{label, confidence, sign_id, reference_video_url}` |
| POST | `/score` | `{sign_id, frames}` → `{score, components, hints, predicted}` |
| POST | `/text-to-sign` | `{text}` → ordered entries (gloss, sign id, video, fingerspelling) |
| POST | `/recordings` | multipart recorder upload (meta + landmarks JSON + optional video) |
| GET | `/recordings/summary?signer_id=` | takes per sign for the recorder |
| POST | `/attempts`, GET `/progress?learner_id=` | attempt history and mastery |
| GET | `/models/alphabet` | browser model file |

Frames follow the `RawFrame` format in [docs/FEATURE_SPEC.md](docs/FEATURE_SPEC.md) §1. Media files are served from
`/media` with HTTP Range support.

## Project structure

```text
docs/            PLAN.md · FEATURE_SPEC.md · DATASET.md
backend/
  app/           main.py (API) · features.py (spec) · extract.py (MediaPipe offline) · recognition.py ·
                 train_static.py · registry.py · text_plan.py · clipstore.py · content.py (hints) · config.py · db.py
  app/legacy/    old synthetic templates, v0 features, v0 DTW (dev fallback only)
  migrations/    Alembic (0001 legacy baseline, 0002 coach schema)
  scripts/       download_models · train_alphabet · record_cli · seed_dev_synthetic · datasets/ (ingest_*)
  tests/         pytest suite
frontend/src/
  lib/           features.js (spec) · vision.js (MediaPipe) · staticModel.js · segmenter.js · identity.js
  components/    Camera.jsx · SkeletonPlayer.jsx · ReferencePanel.jsx
  pages/         Recognize · TextToSign · Practice · Record · Library
```

## Troubleshooting

- **`pip` fails with `cannot import name 'RequirementInformation'`:** the venv's pip is a mix of two versions.
  Delete `.venv\Lib\site-packages\pip*` and run `python -m ensurepip`.
- **"Alphabet model: not trained" badge:** record or ingest data, then run `python -m scripts.train_alphabet`.
- **Location feedback missing:** keep your face and shoulders in view; the camera warns when they aren't visible.
- **Camera unavailable:** grant permission, use `localhost`, close other apps using the camera.
- **`GET /` is 404:** expected; use `/docs`, `/api/status` or the frontend.

## Licenses and attribution

Each reference clip stores its license and attribution, and the UI shows them next to references. Your own
recordings are yours. How2Sign is CC BY-NC 4.0. ASL Citizen (Microsoft) is licensed for non-commercial research only.
WLASL is C-UDA (computational/academic use only). Check each Kaggle dataset's page. Don't deploy publicly or
commercially with non-commercial data.
