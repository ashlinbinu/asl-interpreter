# Handspeak: ASL learning coach

Handspeak is a local web app for **learning** American Sign Language. You can type text and see each sign,
practise with live feedback from your webcam, and sign to the camera to get the nearest matches.

> **Honest scope.** Handspeak is a learning aid, not a certified ASL translator. Scores are estimates. Text → sign
> is word-by-word glossing, not ASL grammar, and facial grammar (non-manual markers) is only partly captured. Several
> datasets it can use are **non-commercial** (see [docs/DATASET.md](docs/DATASET.md)).

Status: phases P0–P5 of [docs/PLAN.md](docs/PLAN.md) are implemented. Measured results on held-out signers are in
[docs/EVALUATION.md](docs/EVALUATION.md):
- **50 words:** 84.0% top-1 and 95.3% top-5 on 11 ASL Citizen signers the model never saw (633 clips).
- **Alphabet (24 static letters): unverified.** 75.0% on one held-out ChicagoFSVid signer, but those are studio hand
  crops, not webcam video. No full-frame multi-signer letter video exists yet. J and Z are not recognised.
- The browser pages were built and unit-tested, but **not tried with a real webcam** (none was available).

## How it works

- The **browser** runs MediaPipe Hand (2 hands) + Pose landmarkers and computes features per
  [docs/FEATURE_SPEC.md](docs/FEATURE_SPEC.md). The features cover hand shape, handedness, location relative to
  face/body, and motion. Letters and words are recognised live in the browser.
- The **backend** (FastAPI) uses the same MediaPipe models and the *same* feature spec (a parity test keeps the
  Python and JS pipelines identical). It stores real reference clips, scores attempts (classifier + DTW against real
  signers, per component) and tracks progress.
- **References are real signer data only**: ASL Citizen videos for words, your own recordings, or ingested datasets.
  The old procedurally generated templates survive only as a dev fallback (`HANDSPEAK_DEV_SYNTHETIC=1`).

| Page | What it does |
|---|---|
| Sign → text | Live letters (in-browser) and live **words** (in-browser model, plus a server check per finished sign): top-5 with calibrated confidence. Image upload for letters. "Learn" → reference + practice |
| Text → sign | Gloss timeline with fingerspelling fallback and optional approximate ASL-style glossing; **sequence video player** (crossfade, 0.5/0.75/1×, loop, captions, ←/→, R, P) |
| Practice | Reference (real video or landmarks) beside your camera; live match meter; score with handshape / orientation / location / movement components and directional hints; replay your attempt (R); **sentence practice** (word by word, or whole sentence) |
| Progress | Signs practised, mastered (≥80 in 3 of the last 4), Leitner review queue, recent attempts |
| Record dataset | Guided recorder for A–Z (J/Z as motion takes) and words: prompts, countdown, N reps, quality check |
| Add a sign | Record a custom sign (e.g. a name sign) as a reference clip |

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
npm run dev                           # http://127.0.0.1:5173
```

`run_all.bat` starts both after installation, and `run_tests.bat` runs the test suite.

On first start the backend migrates the database to the latest schema (Alembic). An existing `asl.db` from the
previous version is upgraded in place: the old templates are kept but tagged `synthetic`.

## Getting data (needed for real accuracy)

1. **Record yourself (best data).** Open *Record dataset*, choose a signer id (e.g. `me_01`), pick letters, set
   5–10 repetitions and press Space for each take. Record other people under their own ids, with their consent.
   For a CLI alternative: `python -m scripts.record_cli --signer-id me_01 --alphabet --reps 5`.
2. **Optional public alphabet images (supplementary training data only, never used for testing).** Download a
   folder-per-letter dataset, e.g. Kaggle *ASL Alphabet*
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
   Trained models are **not in git** (`.gitignore`): they are derived from non-commercial / research-only data, so
   each machine trains its own.

4. **Word clips from ASL Citizen** (Microsoft; non-commercial research license, read it first). Only the clips for
   the 50-word vocabulary are fetched from the 45.9 GB archive, using HTTP range requests:

   ```powershell
   python -m scripts.datasets.inspect_asl_citizen            # index + split CSVs only, no videos
   python -m scripts.datasets.ingest_asl_citizen --dry-run   # shows the MB it would fetch
   python -m scripts.datasets.ingest_asl_citizen --word goodbye=BYE --max-gb 1 --workers 3   # 1,557 clips, 791 MB, ~1 h
   python -m scripts.build_reference_videos --category word
   python -m scripts.dataset_report --write-docs
   ```

5. **Train the word model** (needs the training extras: `python -m pip install -r requirements-train.txt
   --extra-index-url https://download.pytorch.org/whl/cpu`):

   ```powershell
   python -m scripts.train_words          # 1D-CNN vs BiGRU on ASL Citizen's official signer split, ~5-10 min on CPU
   python -m scripts.evaluation_report    # regenerates docs/EVALUATION.md
   ```

6. **Optional: ChicagoFSVid letter crops** (4 signers, research-only, no license stated). Download
   `home.ttic.edu/~klivescu/chicago-fsvid.tgz` (1.7 GB), extract it, then
   `python -m scripts.datasets.ingest_chicago_fsvid --root <extracted>\fsvid`. These are studio hand crops,
   recoloured so MediaPipe can detect them. They help train the alphabet but never count as verification.

Keep large data outside OneDrive: set `HANDSPEAK_STORE_DIR` (see below), e.g. `D:\handspeak_data`.

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

The suite covers:
- feature geometry (invariance, handedness, mirroring, gap fill, location) and **Python↔JS feature parity**;
- JS↔Python parity of both models (alphabet, and the word CNN/BiGRU), plus torch↔numpy word-model parity;
- migrations (including upgrading the legacy DB) and training splits (no signer leakage, photo sources train-only,
  the "verified" rule);
- hold detection, DTW scoring and hints;
- the ASL Citizen pipeline on a fake archive and a local range server;
- text → sign glossing, API schemas/limits/CORS/Range requests, recordings, progress/review, auto-segmentation;
- an **integration test** that runs real held-out ASL Citizen videos through MediaPipe → features → word model →
  API.

Skips: Node-based tests without `node` on PATH, MediaPipe tests until `scripts.download_models` has run, torch tests
without `requirements-train.txt`, and the integration test without local data and a trained word model.

## API (all under `/api`; full schemas at `/docs`)

| Method | Route | Purpose |
|---|---|---|
| GET | `/status` | DB counts, model versions + held-out metrics from `metrics.json`, feature spec version |
| GET | `/signs?q=&category=&kind=&has_video=` | vocabulary |
| GET | `/signs/{id}` | sign + reference video/landmarks, license, attribution |
| POST | `/signs` | add a custom sign (RawFrames) |
| POST | `/recognize` | `{frames, mode: "alphabet"\|"word"}` or `{image}` → top-k `{label, confidence, sign_id, reference_video_url}` |
| POST | `/score` | `{sign_id, frames}` → `{score, components, hints, predicted}` (letters: calibrated probability; words: classifier + DTW components) |
| POST | `/text-to-sign` | `{text, normalize}` → ordered entries (gloss, sign id, video, attribution, fingerspelling), dropped/changed words |
| POST | `/recordings` | multipart recorder upload (meta + landmarks JSON + optional video) |
| GET | `/recordings/summary?signer_id=` | takes per sign for the recorder |
| POST | `/attempts`, GET `/progress?learner_id=` | attempt history, mastery, Leitner boxes and review queue |
| GET | `/models/alphabet`, `/models/words` | browser model files |

Frames follow the `RawFrame` format in [docs/FEATURE_SPEC.md](docs/FEATURE_SPEC.md) §1. Media files are served from
`/media` with HTTP Range support.

## Project structure

```text
docs/            PLAN.md · FEATURE_SPEC.md · DATASET.md · EVALUATION.md (generated)
backend/
  app/           main.py (API) · features.py (spec) · extract.py (MediaPipe offline) · recognition.py ·
                 train_static.py · holds.py (letter holds in video) · train_words.py · word_model.py (numpy
                 inference + DTW scoring) · registry.py · text_plan.py · clipstore.py · content.py (hints) ·
                 config.py · db.py · vocabulary.py (50 words, gloss matching) · remote_zip.py (HTTP-range zip reader)
  app/legacy/    old synthetic templates, v0 features, v0 DTW (dev fallback only)
  migrations/    Alembic (0001 legacy baseline, 0002 coach schema)
  scripts/       download_models · train_alphabet · train_words · evaluation_report · record_cli ·
                 build_reference_videos · dataset_report · seed_dev_synthetic ·
                 datasets/ (inspect_asl_citizen, ingest_asl_citizen, ingest_chicago_fsvid, ingest_alphabet_images, …)
  tests/         pytest suite
frontend/src/
  lib/           features.js (spec) · vision.js (MediaPipe) · staticModel.js · wordModel.js · segmenter.js ·
                 useLiveLetters.js · useLiveWords.js · identity.js
  components/    Camera · SkeletonPlayer · ReferencePanel · VideoSequencePlayer · SentencePractice
  pages/         Recognize · TextToSign · Practice · Progress · Record · Library
```

## Troubleshooting

- **`pip` fails with `cannot import name 'RequirementInformation'`:** the venv's pip is a mix of two versions.
  Delete `.venv\Lib\site-packages\pip*` and run `python -m ensurepip`. This happened twice with the venv inside
  OneDrive. If it recurs, create the venv outside OneDrive (e.g. `py -3.12 -m venv D:\handspeak_venv`).
- **Landmark extraction is slow / the PC freezes:** use `--workers 3`. Each MediaPipe process holds ~740 MB.
- **"Words model: not trained" badge:** run `python -m scripts.train_words`, then restart the API.
- **"Alphabet model: not trained" badge:** record or ingest data, then run `python -m scripts.train_alphabet`.
- **Location feedback missing:** keep your face and shoulders in view; the camera warns when they aren't visible.
- **Camera unavailable:** grant permission, use `127.0.0.1` or `localhost`, close other apps using the camera.
- **"Waiting for the backend…" / "Backend unavailable":** the page retries every 2 s and loads by itself once the
  backend answers. If it never does, read the "ASL Backend" window. If you clicked inside a server window, Windows
  pauses that program while text is selected: press Esc in it.
- **Low frame rate:** the camera badge shows FPS and the tracker (`GPU` or `CPU`). On this laptop (Ryzen 5 5500U,
  Radeon graphics) Chrome measured ~25 fps with a real signer video before pose was thinned to every second frame,
  ~37 after (camera-limited at 30 with a webcam). Close other heavy tabs and programs; `CPU` means the GPU couldn't
  be used.
- **`GET /` is 404:** expected; use `/docs`, `/api/status` or the frontend.

## Licenses and attribution

Each reference clip stores its license and attribution, and the UI shows them next to references. Your own
recordings are yours. How2Sign is CC BY-NC 4.0. ASL Citizen (Microsoft) is licensed for non-commercial research only.
WLASL is C-UDA (computational/academic use only). Check each Kaggle dataset's page. Don't deploy publicly or
commercially with non-commercial data.
