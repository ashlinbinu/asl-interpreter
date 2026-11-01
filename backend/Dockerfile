# Sign-ify API for a free Render web service (docs/DEPLOYMENT.md, route B). Build context: backend/
#   docker build -t signify-api backend
#   docker run --rm -p 10000:10000 -e DATABASE_URL=... -e CORS_ORIGINS=... signify-api
# No PyTorch (training only) and no onnxruntime: inference is numpy + scikit-learn.
FROM python:3.12-slim

# OpenCV and MediaPipe (server-side photo recognition) need these shared libraries to import.
RUN apt-get update && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

ENV PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 PIP_DISABLE_PIP_VERSION_CHECK=1
# Render's free plan has 0.1 CPU and 512 MB: one BLAS/numba thread, since each extra thread only adds memory.
ENV OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMBA_NUM_THREADS=1
# The DTW kernel is compiled into the image below for a generic x86-64 CPU, so a waking server loads it instead of
# spending ~20 s (at 0.1 CPU) compiling it on the first scored attempt.
ENV NUMBA_CACHE_DIR=/app/.numba-cache NUMBA_CPU_NAME=generic
# Defaults; Render's environment overrides them. With HANDSPEAK_STORAGE=s3 nothing is written to the store dir.
ENV HANDSPEAK_STORE_DIR=/tmp/handspeak-store PORT=10000

WORKDIR /app
COPY requirements.txt requirements-deploy.txt ./
RUN pip install -r requirements-deploy.txt

COPY alembic.ini ./
COPY migrations ./migrations
COPY models ./models
COPY app ./app
RUN python -m compileall -q app migrations \
    && useradd --create-home --uid 10001 signify && mkdir -p "$NUMBA_CACHE_DIR" && chown signify "$NUMBA_CACHE_DIR"
USER signify
RUN python -c "import numpy as np; from app.word_model import dtw_path; a = np.zeros((32, 8)); dtw_path(a, a)"

EXPOSE 10000
# Migrations run at startup: app.main.startup() -> init_db() brings DATABASE_URL to the latest Alembic revision
# before the API accepts requests.
CMD ["sh", "-c", "exec uvicorn app.main:app --host 0.0.0.0 --port \"$PORT\" --no-access-log"]
