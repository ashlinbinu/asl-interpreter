# Deployment (free)

Two ways to put Sign-ify on the internet without paying. Nothing here deploys by itself: every step below is one you
run. **Read [§5 Licenses](#5-dataset-licenses-what-may-be-served-publicly) before sharing any link**: some of the
reference media in the local store may not be shown to the public.

| | Route A: laptop + Cloudflare quick tunnel | Route B: Cloudflare Pages + Render + Neon + R2 |
|---|---|---|
| Runs on | your laptop (must stay on and awake) | free hosted services |
| URL | random `https://….trycloudflare.com`, new every start | fixed `https://….pages.dev` |
| Setup | ~10 minutes, no accounts | ~1 hour, four accounts |
| Data | your local database and store | Neon (database) + R2 (files), copied with `scripts.publish_store` |
| First visit after idle | instant | ~1 minute while the free server wakes up |

Both give HTTPS, which browsers require before they allow the camera on any address other than `localhost`.

## 1. Before you start

- **No login.** Anyone with the link can use every page: add or delete custom signs, upload recordings (up to 60 MB
  each) and see every custom sign's video. Share the link only with people you trust.
- **Route B builds from GitHub**, so the code has to be committed and pushed first (resolve the conflict in
  `README.md` before committing). The repo belongs to `ashlinbinu`: if you can't give Render and Cloudflare access to
  it, fork it and deploy the fork.

Commands below assume:

```powershell
$repo = "C:\Users\hp\OneDrive\Desktop\PYTHON\asl-interpreter"
```

## 2. Route A: laptop + Cloudflare quick tunnel

The built frontend is served by `vite preview` on `127.0.0.1:4173`, which proxies `/api` and `/media` to the backend
on `127.0.0.1:8000` (`frontend/vite.config.js`). The tunnel publishes port 4173 only, so the page and the API share
one HTTPS origin and no CORS or `VITE_API_URL` is needed.

**Once:** install `cloudflared`, then open a **new** PowerShell window so it is on `PATH`.

```powershell
winget install --id Cloudflare.cloudflared
cloudflared --version
```

**1. Build the frontend** (again after every frontend change). `VITE_API_URL` must be unset for this route; also
delete `frontend\.env.production.local` if you created one for route B.

```powershell
cd $repo\frontend
npm install
Remove-Item Env:VITE_API_URL -ErrorAction SilentlyContinue
npm run build
```

**2. Window 1: backend**, on the loopback address only. Ready when it prints `Application startup complete`.

```powershell
cd $repo\backend
.\.venv\Scripts\python.exe -m uvicorn app.main:app --host 127.0.0.1 --port 8000 --no-access-log
```

**3. Window 2: the built frontend.** Open <http://127.0.0.1:4173> to check it before going public.

```powershell
cd $repo\frontend
npm run preview
```

**4. Window 3: the tunnel.**

```powershell
cloudflared tunnel --url http://127.0.0.1:4173
```

It prints a line like `https://random-words-here.trycloudflare.com`: that is the link to share. Ctrl+C in each of
the three windows stops everything; the next start prints a new link.

**Camera over HTTPS.** The tunnel URL has a valid certificate, so the camera works on phones and other computers. The
browser asks for permission on the first visit; if it was refused, allow it again under the padlock icon in the
address bar. `http://<laptop-IP>:4173` on your Wi-Fi does **not** work: browsers block the camera on plain HTTP.

**Serving only the public subset.** As set up above, the tunnel serves your whole local store, including the ASL
Citizen word videos that may not be redistributed (§5). To serve only what route B would publish, copy that subset
to a separate database and folder once, and point the backend (window 1) at it. Variables set in the shell take
precedence over `backend\.env`.

```powershell
cd $repo\backend
New-Item -ItemType Directory -Force D:\handspeak_public | Out-Null
.\.venv\Scripts\python.exe -m scripts.publish_store --to-db "sqlite:///D:/handspeak_public/public.db" --to-storage D:\handspeak_public --apply
$env:DATABASE_URL = "sqlite:///D:/handspeak_public/public.db"
$env:HANDSPEAK_STORE_DIR = "D:\handspeak_public"
.\.venv\Scripts\python.exe -m uvicorn app.main:app --host 127.0.0.1 --port 8000 --no-access-log
```

**Limits.** Quick tunnels are meant for testing: up to 200 requests in flight, no uptime promise, and the link only
works while the laptop is awake and all three windows are open (Settings > System > Power & battery > Screen, sleep
& hibernate timeouts: never sleep when plugged in). A fixed hostname needs a named tunnel and a domain on your Cloudflare account; add that hostname to
`preview.allowedHosts` in `frontend/vite.config.js`.

**Troubleshooting.** `Blocked request. This host is not allowed`: you tunnelled `npm run dev` (port 5173) instead of
`npm run preview`. Cloudflare error 502: window 2 isn't running. The page says "Waiting for the backend": window 1
isn't running or hasn't finished starting.

## 3. Route B: static frontend + Render + Neon + R2

```
browser ── https://PROJECT.pages.dev ─────────── static files (Cloudflare Pages or Netlify)
   │
   └──── https://SERVICE.onrender.com/api ────── FastAPI in Docker (Render, free) ── Neon Postgres (free)
              /media/... ── 307 redirect ──────── signed URL to a private Cloudflare R2 bucket (free)
```

The free tiers this route relies on, as stated by each provider in October 2026 (check before relying on them):

| Service | Free tier | What it means here |
|---|---|---|
| [Render](https://render.com/docs/free) web service | 512 MB RAM, 0.1 CPU ([compute plans](https://render.com/docs/compute-plans)); sleeps after 15 min without inbound traffic, ~1 min to wake; disk wiped on every restart, redeploy and spin-down; 750 instance-hours per workspace per month | the API: ~270 MB working set measured locally after startup, so it fits. Files must live in R2 and data in Neon. One always-on service fits in 750 h. |
| Render Postgres | expires 30 days after creation | not used: Neon instead |
| [Neon](https://neon.com/pricing) | 0.5 GB storage and 100 CU-hours per project per month; compute sleeps after 5 min idle | the database (a few MB). 100 CU-hours is about 400 awake hours at the smallest size (0.25 CU); past the limit compute is suspended until next month |
| Cloudflare R2 | 10 GB-month storage, 1 M writes and 10 M reads per month, free egress; may ask for a card or PayPal before the first bucket (not charged within the free tier) | landmarks, recordings and reference media (1.9 MB to start) |
| Cloudflare Pages | 500 builds per month, unmetered static bandwidth | the frontend |
| Netlify (alternative) | new accounts: 300 credits per month; each production deploy costs credits | the frontend, if you prefer Netlify |

Each service needs a URL or secret from another, so do the steps in this order.

### B1. Neon: the database

1. Sign up at <https://neon.com> and create a project in a region near your users.
2. **Connect** > turn **Connection pooling** off (the direct connection suits Alembic migrations) > copy the
   connection string. It looks like
   `postgresql://neondb_owner:PASSWORD@ep-NAME-123456.REGION.aws.neon.tech/neondb?sslmode=require&channel_binding=require`.

You don't create any tables: the backend runs its Alembic migrations against `DATABASE_URL` every time it starts.

### B2. Cloudflare R2: the files

1. In the Cloudflare dashboard open **R2 Object Storage** (add a payment method if asked).
2. **Create bucket**, e.g. `signify-store`. Leave public access **off** and add no custom domain: the API hands out
   short-lived signed links (6 hours) for reference media only, so recordings and landmarks are never public.
3. Note the **Account ID** shown on the R2 overview page. The endpoint is
   `https://ACCOUNT_ID.r2.cloudflarestorage.com`.
4. **Manage API tokens** > **Create API token**: permission **Object Read & Write**, applied to this bucket only.
   Copy the **Access Key ID** and **Secret Access Key** (the secret is shown once).

No CORS rule is needed on the bucket: the app loads media with plain `<video>` and `<img>` tags.

### B3. Publish the vocabulary and public media from the laptop

A fresh database only gets the letters A-Z; the 51 words come from the ingest scripts, which run on the laptop. This
copies every letter and word, plus the reference media whose licenses allow it (§5), into Neon and R2. Learners,
attempts, custom signs and recordings are not copied.

```powershell
cd $repo\backend
.\.venv\Scripts\python.exe -m pip install -r requirements-deploy.txt          # adds boto3
$env:HANDSPEAK_S3_BUCKET = "signify-store"
$env:HANDSPEAK_S3_ENDPOINT = "https://ACCOUNT_ID.r2.cloudflarestorage.com"
$env:HANDSPEAK_S3_ACCESS_KEY_ID = "R2_ACCESS_KEY_ID"
$env:HANDSPEAK_S3_SECRET_ACCESS_KEY = "R2_SECRET_ACCESS_KEY"
$neon = 'postgresql://neondb_owner:PASSWORD@ep-NAME-123456.REGION.aws.neon.tech/neondb?sslmode=require&channel_binding=require'
.\.venv\Scripts\python.exe -m scripts.publish_store --to-db $neon --to-storage s3            # dry run: reports only
.\.venv\Scripts\python.exe -m scripts.publish_store --to-db $neon --to-storage s3 --apply
```

Keep the connection string in **single** quotes so PowerShell leaves its `&` and any `$` alone. With the current store
the dry run reports `77 signs, 50 clips, 26 videos, 24 photos: 100 files, 1.9 MB`. Re-run it with `--apply` after
ingesting new data; it updates rows in place and uploads only missing files.

On the hosted site the letters have their videos, photos and skeletons, and all 51 words are recognised, scored and
used by text-to-sign, but words have **no reference video or skeleton**: those come from ASL Citizen (§5).

### B4. Render: the API

1. Push the code to GitHub (§1).
2. <https://render.com> > **New** > **Web Service** > connect the GitHub repository.
3. Settings: **Language** Docker, **Branch** `main`, **Root Directory** `backend`, **Dockerfile Path**
   `./Dockerfile`, **Instance Type** Free, **Health Check Path** `/api/health` (under Advanced).
4. **Environment Variables**:

   | Key | Value |
   |---|---|
   | `DATABASE_URL` | the Neon connection string |
   | `CORS_ORIGINS` | the Pages URL from B5, e.g. `https://signify.pages.dev` (no trailing slash). Before B5 exists, put any placeholder and change it after B5 |
   | `HANDSPEAK_STORAGE` | `s3` |
   | `HANDSPEAK_S3_BUCKET` | `signify-store` |
   | `HANDSPEAK_S3_ENDPOINT` | `https://ACCOUNT_ID.r2.cloudflarestorage.com` |
   | `HANDSPEAK_S3_ACCESS_KEY_ID` | the R2 Access Key ID |
   | `HANDSPEAK_S3_SECRET_ACCESS_KEY` | the R2 Secret Access Key |

5. **Create Web Service**. The first build takes several minutes (MediaPipe and OpenCV wheels). Render sets `PORT`
   itself; the image listens on it.
6. Check `https://SERVICE.onrender.com/api/health`: `"database": {"connected": true, "database_type": "postgresql",
   "signs_count": 77, …}` and both models `"ready": true`.

The image (`backend/Dockerfile`) has no PyTorch and no onnxruntime: the word network runs in numpy and the alphabet
model in scikit-learn. It copies the trained models from `backend/models` in the repo, and never the local database,
`.env` or data store (`backend/.dockerignore`). It also compiles the DTW kernel at build time, so a waking server
doesn't spend the first scored attempt compiling it.

### B5. Cloudflare Pages: the frontend

1. Cloudflare dashboard > **Workers & Pages** > **Create** > **Pages** > **Connect to Git** > the repository.
2. **Production branch** `main`, **Framework preset** None, **Build command** `npm run build`, **Build output
   directory** `dist`, **Root directory** (advanced) `frontend`.
3. **Environment variables**: `VITE_API_URL` = `https://SERVICE.onrender.com/api` and `NODE_VERSION` = `22`.
4. **Save and Deploy**. The site is `https://PROJECT.pages.dev`.

Without Git access, build locally and upload `frontend\dist` under **Pages > Create > Upload assets**:

```powershell
cd $repo\frontend
$env:VITE_API_URL = "https://SERVICE.onrender.com/api"
npm run build
```

**Netlify instead:** **Add new site** > **Import an existing project**, **Base directory** `frontend`, **Build
command** `npm run build`, **Publish directory** `frontend/dist`, and the same two environment variables.

`VITE_API_URL` is read at build time: change it, then rebuild. `frontend/.env.example` lists it.

### B6. Finish

1. On Render, set `CORS_ORIGINS` to the exact Pages URL (comma-separate several, e.g. a Netlify URL too) and save;
   the service restarts. Pages preview URLs (`https://HASH.PROJECT.pages.dev`) are not allowed unless you list them.
2. Open the Pages URL. After an idle spell the page shows **"Waking up the server…"** for about a minute, then opens.

**How the sleeping server is handled.** When `VITE_API_URL` is a full `https://` URL, the frontend knows the API is
remote. Until the API has answered once, requests that fail the way a waking host fails (a network error, or a
502/503/504 that isn't the API's own JSON) are retried for about 2.5 minutes, and then the app keeps polling
`/api/status` every 2 s. Meanwhile it shows the waking-up screen instead of the local "start the backend" help. While
a tab is open, the 15 s status poll keeps the server awake. Local runs and route A are unaffected.

**Without R2** (no card on file): leave `HANDSPEAK_STORAGE` unset. The API still runs, but reference videos and photos
are missing, and custom signs and recordings disappear whenever the server sleeps or redeploys while their rows stay
in Neon. Not recommended.

**Updates.** Pushing to `main` rebuilds Render and Pages. New data: re-run B3. New models: they are copied from
`backend/models` in the repo, so they must be committed (but see §5).

## 4. Values you must enter yourself

| Value | Where it comes from | Where it goes |
|---|---|---|
| Neon connection string | Neon > Connect (pooling off) | Render `DATABASE_URL`; `publish_store --to-db` |
| R2 Account ID | R2 overview page | inside `HANDSPEAK_S3_ENDPOINT` = `https://ACCOUNT_ID.r2.cloudflarestorage.com` (Render and laptop) |
| R2 bucket name | you choose it (B2) | `HANDSPEAK_S3_BUCKET` (Render and laptop) |
| R2 Access Key ID and Secret Access Key | R2 > Manage API tokens | `HANDSPEAK_S3_ACCESS_KEY_ID`, `HANDSPEAK_S3_SECRET_ACCESS_KEY` (Render and laptop) |
| Render service URL | Render, after B4 | Pages/Netlify `VITE_API_URL` = that URL + `/api` |
| Pages (or Netlify) URL | after B5 | Render `CORS_ORIGINS` |
| Tunnel URL (route A) | printed by `cloudflared` | nothing: share it |

Fixed values: `HANDSPEAK_STORAGE=s3`, `NODE_VERSION=22`. Secrets go only into Render's environment settings and your
PowerShell session, never into a committed file.

## 5. Dataset licenses: what may be served publicly

Summary of each source's terms as recorded in [DATASET.md](DATASET.md) and, for ASL Citizen, its
[license page](https://www.microsoft.com/en-us/research/project/asl-citizen/dataset-license/). This is not legal
advice: when in doubt, ask the rights holder in writing.

### Sources whose licenses forbid public redistribution

| Source | In the local store | Terms | Must not be served publicly |
|---|---|---|---|
| `asl_citizen` (Microsoft) | 51 word reference videos, 1,584 landmark clips, raw clips under `raw/asl_citizen/` | Microsoft Research License Terms: "solely for non-commercial, non-revenue generating, research purposes"; "you may not distribute the data or your modifications to the data"; you may not "share, publish, distribute or lend the Materials" or "provide the Materials as a stand-alone hosted solution for others to use". Signers' faces are personal data to be kept "in strict confidence". | the word videos (`media/videos/<id>/asl_citizen_*.mp4`); the landmark clips, which the API sends as the reference skeleton (`reference_landmarks` in `GET /api/signs/{id}`); the raw videos. `publish_store` leaves all of them out. |
| `chicago_fsvid` (Kim et al. 2017) | 3,780 landmark clips (static letters), the downloaded hand-crop archive | no license stated, so all rights are reserved by default | everything from it. None of it is served as media today (its clips are never a sign's reference). |
| not ingested: ChicagoFSWild, Surrey/Exeter | none | no license stated; the FSWild source videos belong to their creators | don't add them to a public site |
| not ingested: WLASL | none | C-UDA 1.0 (computational use) for the annotations; the videos belong to their YouTube and ASL-site owners | don't host the videos |

### Sources that may be shown, with conditions

| Source | Served | Conditions |
|---|---|---|
| `wikimedia_commons` (Travis Dougherty, filmed by Fuzheado) | 26 letter videos and their skeletons | CC BY-SA 3.0: attribution, a license link, a note of the changes (the clips are cut, cropped and mirrored), and the adapted clips under the same license. The app shows the attribution and license under each video. |
| `kaggle_asl_alphabet` (Akash / grassknoted) | 24 handshape photos and their skeletons | GPL 2 (Kaggle metadata): keep the license notice with the photos, as the app does. |
| not ingested: How2Sign (CC BY-NC 4.0), Sem-Lex (CC BY-NC-SA 4.0), FSboard (CC BY 4.0) | none | shareable non-commercially (FSboard also commercially) with attribution; check again before ingesting |

### Media recorded by people

| Source | Must not be served publicly unless |
|---|---|
| `own` (Record page, `record_cli.py`): webcam videos under `recordings/own/` | each signer agreed to public display; the consent recorded so far is for local research |
| `user` (custom signs): `media/videos/<id>/user_*`, `recordings/user/` | that person agreed. On a hosted site every visitor sees every custom sign's video, so say so to the people who record one. |

### Already public: needs your decision

- `backend/models/words/*/references.npz` holds feature sequences of ASL Citizen training signers ("modifications to
  the data"). `backend/models/words/*/model.json` holds weights trained on ASL Citizen, and the API sends them to every
  browser (`/api/models/words`). `backend/models/alphabet/*` is trained on ChicagoFSVid and the Kaggle photos and is
  sent to browsers too (`/api/models/alphabet`).
- `.gitignore` says these must not be in git, but commit `36f2fb5` ("second commit") added them and they are on the
  public GitHub `main` of `ashlinbinu/asl-interpreter`.
- Options: ask Microsoft for written permission (the dataset page lists `ASL_Citizen@microsoft.com`), or take the
  files out of the public repository and deploy from a private fork. Deleting them in a new commit leaves them in the
  history; removing them for good means rewriting history and force-pushing, which the repo owner has to coordinate.
  A public learning site with no revenue may still not count as "research", which is one more question to ask.
