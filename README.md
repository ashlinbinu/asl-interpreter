# Handspeak: ASL Interpreter and Practice Studio

Handspeak is a local full-stack web application for experimenting with American Sign Language (ASL) alphabet recognition, sign-library lookup, text-to-sign display, and camera-based practice.

It consists of:

- A **React + Vite** browser app.
- A **FastAPI** backend.
- A **SQLite** database by default, with optional PostgreSQL support.
- **MediaPipe Tasks Vision** in the browser for live hand landmarks.
- A backend **HOG image classifier** for uploaded images when its trained model artifact is available.
- Landmark-sequence matching with Dynamic Time Warping (DTW) for recognition and practice.

> **Important:** This is a learning/demo application, not a production ASL translator. Its generated reference poses are templates, and its text-to-sign behavior is a database lookup with fingerspelling fallback—not ASL grammar translation. Scores and image-classifier confidence are estimates, not guarantees of correctness.

## Features

- **Sign to text**
  - Use a webcam to capture a short sequence of hand landmarks.
  - Or upload an image for alphabet classification, if the trained model is available.
  - Results show up to five matches. When a matching sign is in the library, its reference animation is displayed.
  - The recognized label can be read aloud using the browser's speech synthesis.
- **Text to sign**
  - Looks for matching phrases (up to four words) and individual labels in the sign library.
  - Uses a fuzzy match for individual words.
  - If a word is not found, it attempts to display it as individual alphabet signs. Unsupported characters are reported as missing.
- **Practice**
  - Select signs from alphabet, common vocabulary, or the whole library.
  - Compare a camera capture with a stored sequence; see an estimated score and feedback.
  - Tracks mastered signs and a streak for the current browser session.
- **Add a sign**
  - Record a labeled sign using the camera. It is saved in the database and loaded into the backend's in-memory library.
- **Status and API documentation**
  - The app polls the backend status endpoint every eight seconds.
  - FastAPI exposes interactive API documentation at `/docs`.

## Requirements

- **Python 3.10–3.12** is the intended backend range for the pinned `mediapipe==0.10.14` dependency. Python 3.12 is recommended on Windows.
- **Node.js and npm** for the frontend.
- A modern browser. Camera features require camera permission and work on `localhost` or a secure HTTPS origin.
- Internet access on first use of the camera hand tracker: the frontend loads MediaPipe WASM and the hand-landmarker task model from public CDNs.

## Quick start on Windows

Open a PowerShell terminal in the project root.

### 1. Set up and start the backend

```powershell
cd "C:\Users\hp\OneDrive\Desktop\PYTHON\asl-interpreter\backend"
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m uvicorn app.main:app --reload
```

Keep this terminal open. The API listens at `http://127.0.0.1:8000`.

If PowerShell blocks virtual-environment activation, allow scripts for just the current terminal, then activate again:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
```

To check that the API and database are responding, open `http://localhost:8000/api/status`. The interactive API reference is at `http://localhost:8000/docs`.

### 2. Set up and start the frontend

Open a **second** terminal:

```powershell
cd "C:\Users\hp\OneDrive\Desktop\PYTHON\asl-interpreter\frontend"
npm install
npm run dev
```

Open the local URL Vite prints, normally `http://localhost:5173`. Keep both terminals running while using the app. Vite forwards frontend `/api` requests to `http://localhost:8000`.

### One-click launcher

After installing the backend and frontend dependencies as above, you can run `run_all.bat` from the project root. It opens separate command windows for the API and frontend. Open `http://localhost:5173`.

Alternatively, run `start_backend.bat` and `start_frontend.bat` separately. The backend launcher activates `backend\.venv` if it exists.

## Verify the installation

In PowerShell, query the backend:

```powershell
Invoke-RestMethod http://localhost:8000/api/status
```

The response includes:

- `ok`: API status.
- `database.connected`, `database.database_type`, `database.signs_count`, and `database.attempts_count`.
- `model.ready`: whether the image-classifier model loaded.
- `signs_in_cache`: number of signs loaded into the running API.

An HTTP `404` at `http://localhost:8000/` is expected: the app does not define a homepage route. Use `/docs`, `/api/status`, or the Vite frontend instead.

## Configuration

### Database

Without a `DATABASE_URL` environment variable, the backend uses SQLite at `backend/asl.db`. Database tables are created automatically when the backend starts.

To use PostgreSQL (including Supabase):

1. Copy `backend\.env.example` to `backend\.env`.
2. Replace the example `DATABASE_URL` with your PostgreSQL connection URI, for example:

   ```env
   DATABASE_URL=<your PostgreSQL connection URI>
   ```

3. Restart the backend.

Keep real credentials private; do not commit `backend\.env`. The PostgreSQL driver is included in `backend/requirements.txt`.

### Frontend API URL

The frontend defaults to `/api`; the Vite development server proxies that path to `http://localhost:8000` using `frontend/vite.config.js`. To use a different backend origin, set `VITE_API_URL` in the frontend environment, for example:

```env
VITE_API_URL=http://localhost:8000/api
```

Restart Vite after changing environment variables.

### Camera and browser permissions

Allow camera access when prompted. If the camera does not start, confirm that another application is not using it and reload the page. Camera access is generally restricted to `localhost` or HTTPS. The browser hand tracker downloads its runtime and model assets from the internet.

## How the recognition and scoring work

### Camera landmark pipeline

The frontend samples hand landmarks from MediaPipe. A frame can contain up to two hands, each represented by 21 landmarks with x, y, and z coordinates. Empty frames are discarded; at least four frames containing a hand are required. Backend preprocessing:

1. Orders hands from left to right in the image.
2. Shifts each hand relative to its wrist and scales it by hand size.
3. Resamples the clip to 32 frames.
4. Produces 126 values per frame (two hands × 21 points × three coordinates).

### Recognition modes

- **Uploaded image:** The backend converts the image to grayscale, extracts HOG features, then uses the saved classifier in `backend/app/asl_model.joblib`. This predicts alphabet classes. If the artifact is missing or cannot load, `model.ready` is false and image classification is unavailable.
- **Camera sequence:** The backend compares the normalized sequence against signs currently in the database using DTW over cosine distance.
- **Hybrid:** If both image and landmark input are sent and both produce predictions, the backend combines their scores.

### Practice scores

The score is based on DTW sequence similarity mapped to a 0–100 scale. It is a rough comparison against the stored reference, not a measure of ASL fluency. The API attempts to save practice attempts in the database.

### Reference signs and library contents

`backend/app/sign_data.py` generates the 26 alphabet templates plus these 10 common vocabulary templates: `hello`, `thank you`, `yes`, `no`, `please`, `help`, `love`, `goodbye`, `water`, and `friend` (36 canonical entries total). These are generated landmark sequences, not recordings from a qualified ASL signer. The database may contain additional user-created signs or imported signs.

## Seed the database and train the image model

The setup script seeds or updates the 36 canonical templates. By default, it also tries to train the image classifier using an image dataset.

To seed templates without training:

```powershell
cd "C:\Users\hp\OneDrive\Desktop\PYTHON\asl-interpreter\backend"
.\.venv\Scripts\Activate.ps1
python -m scripts.train_and_seed --skip-training
```

To train the classifier, provide an image dataset with one folder per uppercase letter (`A` through `Z`), containing `.jpg` images:

```text
Data/
├── A/
│   ├── 1.jpg
│   └── ...
├── B/
└── ...
```

Run the script with the dataset path:

```powershell
python -m scripts.train_and_seed --data-dir "C:\path\to\Data"
```

Optional `--max-per-class N` limits the number of images loaded from each letter folder. Training uses a stratified 85/15 train/test split and prints the measured holdout accuracy. The resulting model is written to `backend/app/asl_model.joblib`. A missing or incomplete dataset prevents training; the script can still seed the database. Restart the API after replacing the model so it loads the new artifact.

## Import signs from How2Sign

The optional `backend/scripts/ingest_how2sign.py` script extracts hand landmarks from downloaded video clips and adds short labels to the database.

1. Download a How2Sign translation TSV and the matching RGB front-view `.mp4` clips from [How2Sign](https://how2sign.github.io/).
2. From `backend`, run:

   ```powershell
   python -m scripts.ingest_how2sign --csv "C:\path\to\translations.tsv" --videos "C:\path\to\videos" --max-words 3 --limit 300
   ```

The script expects the TSV to have `SENTENCE` and `SENTENCE_NAME` columns and each video to be named `<SENTENCE_NAME>.mp4`. It extracts each clip's frames with MediaPipe; processing can take time. Restart the API after importing so the new entries are loaded into its in-memory cache.

How2Sign is sentence-level data and is licensed **CC BY-NC 4.0**; review its terms and attribution requirements, especially before any commercial use.

## API reference

All routes are prefixed with `/api`. Request and response schemas can also be inspected at `http://localhost:8000/docs`.

| Method | Route | Purpose |
|---|---|---|
| `GET` | `/api/health` | Alias for the service and database status response. |
| `GET` | `/api/status` | API, database, model, and sign-cache status. |
| `GET` | `/api/signs?q=<text>&category=<source>` | List signs, optionally filtered by label substring and source (`alphabet`, `common`, `user`, or imported sources such as `how2sign`). |
| `GET` | `/api/signs/{sign_id}` | Get one sign, including its frame sequence. |
| `POST` | `/api/signs` | Add a sign with a label and camera landmark frames. |
| `POST` | `/api/recognize` | Recognize an image, landmark sequence, or both. |
| `POST` | `/api/score` | Compare camera frames with a stored sign and record an attempt. |
| `POST` | `/api/text-to-sign` | Find matching phrase/word signs and fingerspell where possible. |

Request bodies:

- **Add a sign:** `{"label": "welcome", "frames": [...]}`.
- **Recognize:** `{"image": "data:image/jpeg;base64,<base64-encoded-image>"}`, `{"frames": [...]}`, or an object containing both properties.
- **Score:** `{"sign_id": 1, "frames": [...]}`.
- **Translate text:** `{"text": "hello"}`.

For landmark requests, `frames` is an array of frames. Each frame contains zero or more hands; each hand is an array of 21 `[x, y, z]` landmark points. Coordinates are numeric. At least four frames must contain a hand for sequence matching, scoring, and sign creation. The browser generates this structure automatically.

Successful recognition returns `matches` and a `method` (`image_classifier`, `dtw_landmarks`, or `hybrid_fusion`). A practice response includes `score`, `feedback`, and `sign_label`. Text-to-sign returns `signs` and `missing`.

## Project structure

```text
asl-interpreter/
├── README.md
├── run_all.bat
├── start_backend.bat
├── start_frontend.bat
├── backend/
│   ├── .env.example
│   ├── requirements.txt
│   ├── asl.db                         # Created/used by SQLite at runtime
│   ├── app/
│   │   ├── main.py                    # FastAPI routes and application logic
│   │   ├── db.py                      # SQLAlchemy models and database setup
│   │   ├── features.py                # Landmark normalization and resampling
│   │   ├── scoring.py                 # DTW similarity and practice feedback
│   │   ├── model.py                   # HOG image classification
│   │   ├── sign_data.py               # Generated alphabet and common templates
│   │   └── asl_model.joblib           # Optional trained classifier artifact
│   └── scripts/
│       ├── train_and_seed.py          # Seed templates and optionally train model
│       ├── ingest_how2sign.py         # Optional video dataset importer
│       └── test_system.py             # System verification script
└── frontend/
    ├── package.json
    ├── vite.config.js                 # Dev server and API proxy
    ├── index.html
    └── src/
        ├── App.jsx                    # App navigation and status display
        ├── api.js                     # Frontend API client
        ├── main.jsx                   # React entry point
        ├── styles.css
        ├── components/
        │   ├── Camera.jsx             # Webcam capture and MediaPipe tracking
        │   └── SkeletonPlayer.jsx     # Landmark sequence animation
        └── pages/
            ├── Recognize.jsx
            ├── TextToSign.jsx
            ├── Practice.jsx
            └── Library.jsx
```

## Troubleshooting

- **`Could not find a version ... mediapipe==0.10.14`:** Use Python 3.10–3.12. Check available versions with `py --list`, then create the virtual environment explicitly with `py -3.12 -m venv .venv`.
- **`GET /` returns 404:** Expected; use `/docs`, `/api/status`, or the frontend at port 5173.
- **Frontend says backend is unavailable:** Confirm the API terminal is still running on port 8000 and that the frontend is using Vite's default `/api` proxy.
- **No signs appear:** Check `/api/status` for `database.connected` and `database.signs_count`. Run `python -m scripts.train_and_seed --skip-training` from `backend` to add the canonical templates.
- **Image recognition is unavailable:** Check `model.ready` at `/api/status`. Ensure `backend/app/asl_model.joblib` exists and loads; train it using the dataset instructions above.
- **Camera is unavailable:** Grant camera permission, use `localhost` or HTTPS, close applications that may be using the camera, and make sure the browser can access the MediaPipe CDN assets.
- **Text-to-sign reports missing symbols:** The required phrase, word, or alphabet template may not be present in the current database.
- **Port is already in use:** Stop the other process using port 8000 or 5173, or configure a different port and update the frontend API proxy accordingly.

## Security and deployment notes

The included configuration is intended for local development. The API currently allows cross-origin requests from any origin, and the development server uses hot reload. Before exposing it publicly, configure restrictive CORS origins, run behind an appropriate production server/proxy, secure database credentials, and review data privacy and camera-consent requirements.
