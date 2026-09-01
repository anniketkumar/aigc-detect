# Deployment plan

Not yet executed — this is the plan, per request. Scope: get the React UI +
FastAPI backend live at one public HTTPS URL. The Chrome extension and PyPI-
style distribution of `predict.py` are separate, lower-priority tracks
addressed at the end.

## What we're actually deploying

Two runtime pieces, both already working locally, neither containerized yet:

- **`app.py`** — FastAPI adapter. Loads `runs/aug.pt` / `runs/baseline.pt`
  (4 KB each, committed) plus a frozen CLIP ViT-B/16 backbone that
  `open_clip` pulls from OpenAI's own registry on first use (~350 MB,
  currently downloaded at runtime, not baked in).
- **`frontend/`** — Vite + React SPA. In dev it's served by the Vite dev
  server and proxies `/api/*` to `127.0.0.1:8000` (`frontend/vite.config.js`).
  All API calls in `App.jsx` are relative (`fetch('/api/analyze')`, no base
  URL) — confirmed by grep, no hardcoded host anywhere in the frontend.

That last point decides the architecture: **because the frontend never
hardcodes a host, the simplest deploy is one container that serves both** —
FastAPI answers `/api/*` and also serves the built `frontend/dist/` as
static files at `/`. Same-origin, zero CORS config to get right, zero
`VITE_API_BASE` env var to wire through the build. One URL, one service.

## Required code changes (not yet made)

1. **`app.py`: mount the built frontend.** Add, after the existing routes:
   ```python
   from fastapi.staticfiles import StaticFiles
   app.mount("/", StaticFiles(directory="frontend/dist", html=True), name="static")
   ```
   Must be mounted *last* — FastAPI matches routes in registration order, and
   a catch-all static mount before `/api/*` would shadow the API.

2. **`app.py`: bind host/port from the platform.** Every hosting target here
   injects the port via `$PORT` (or fixes it, HF Spaces = 7860). Add a
   `Dockerfile CMD` that reads it rather than hardcoding `--port 8000`:
   ```
   CMD uvicorn app:app --host 0.0.0.0 --port ${PORT:-8000}
   ```

3. **New `Dockerfile`** (doesn't exist yet — repo has no container config at
   all). Key points, not just "pip install":
   - Base on `python:3.13-slim` to match the dev container's Python 3.13.
   - Build the frontend in a separate stage (`node:20-slim`, `npm ci && npm
     run build`) and `COPY --from=` the `dist/` output into the final image,
     so the runtime image doesn't carry Node or `node_modules`.
   - **Bake the CLIP backbone into the image at build time**, don't leave it
     to download on first request:
     ```dockerfile
     RUN python -c "import open_clip; open_clip.create_model_and_transforms('ViT-B-16-quickgelu', pretrained='openai')"
     ```
     Without this, the *first* real user request after every cold start pays
     a ~350 MB download and can time out on a platform's request-timeout
     limit, or fail outright if the container has restricted egress.
   - `opencv-python` is in `requirements.txt` but **not imported anywhere in
     `app.py`/`predict.py`/`src/`** (grep confirms — it's only used by
     `scripts/`, which don't run in the deployed service). Two options:
     - install it anyway and add the system libs it needs (`libgl1
       libglib2.0-0`) to avoid the classic `ImportError: libGL.so.1` in slim
       images, **or**
     - ship a trimmed `requirements-serve.txt` (numpy, pillow, torch,
       torchvision, open-clip-torch, fastapi, uvicorn, python-multipart) for
       a smaller/faster-building image. Recommend trimming — it's dead
       weight in this image and the fewer heavy binary wheels, the fewer
       build surprises.
   - Add a `.dockerignore`: `data/`, `results/`, `.git/`, `frontend/node_modules/`,
     `__pycache__/`, `.pytest_cache/`, `*.ipynb_checkpoints`. Without it the
     build context includes the `data/forbidden/blocklist.json` fetch tree
     and full `results/` (audit CSVs, montages) — none of it needed at
     runtime, all of it slows every build upload.

4. **CORS (`app.py:32-37`)**: currently `allow_origins=["*"]`. Leave it —
   the Chrome extension's background script calls the API cross-origin by
   design (`extension/background.js:43`, see below), so locking CORS to the
   deployed frontend's own origin would break the extension for no real
   security gain (there's no auth/session state to protect here).

None of this touches `predict.py`, `src/`, or the eval/training pipeline —
the deploy surface is exactly `app.py` + `frontend/`.

## Hosting choice

**Correction, checked live (see below):** HF Spaces changed its pricing
since the original version of this plan. Per HF's own docs, *"creating a
Space that runs on compute (Gradio or Docker) requires a paid plan, while
Static Spaces are free for everyone."* CPU Basic hardware has no hourly
cost, but you can't even create a Docker Space without HF PRO (~$9/mo
personal, Team/Enterprise for orgs). Dropping it as the recommendation.

Re-surveyed the realistic options as they actually stand today:

| Platform | Cost | Trade-off here |
|---|---|---|
| **Cloud Run (GCP)** — recommended | Free within quota (2M requests/mo, 180,000 vCPU-s/mo, 360,000 GiB-s/mo, permanent "Always Free," not a trial) | Scale-to-zero means idle time costs nothing, so a low-traffic demo link stays $0. Since Feb 2026 it requires a billing account (card on file) to deploy at all, but nothing is charged while under quota. Most setup of the realistic options: GCP project, `gcloud` auth, Artifact Registry push, `gcloud run deploy` — all scriptable, ~15 min one-time. Cold starts after scale-to-zero (mitigated by baking the CLIP backbone into the image, item 3 above). |
| Oracle Cloud Always Free (VM) | Genuinely free forever, no request-based billing at all | Most generous specs (up to 24 GB RAM, ARM Ampere) but it's a raw VM — you install Docker, open the firewall port, and terminate TLS yourself (e.g. Caddy for auto-HTTPS). More ops work than a managed platform; picks a real long-term $0 outcome over convenience today. |
| Render (Docker web service) | Free, 750 instance-hours/mo | 512 MB RAM / 0.1 CPU — genuinely tight for torch + CLIP (realistic peak is 800 MB-1.2 GB: Python + torch import + ~350 MB weights + FastAPI). May OOM under real load; would need the $7/mo+ paid instance to be reliable. Spins down after 15 min idle, ~1 min cold start on wake. Easiest git-push deploy flow of the group if the RAM risk is acceptable for a demo. |
| Fly.io | Not free for new accounts | 2-hour/7-day trial only, then requires a card; ~$2-5/mo minimum for an always-on app. Dropped as a "free" option. |
| HF Spaces (Docker SDK) | ~$9/mo (PRO) | Otherwise the best fit (16 GB RAM CPU Basic, simple `git push` deploy, HTTPS included) — worth reconsidering if $9/mo is acceptable, since it removes the GCP/Oracle setup overhead entirely. |

**Recommendation: Cloud Run.** It's the only option here that's both
actually free on an ongoing basis *and* a managed platform (no VM/TLS work).
Given the same-day deadline noted in `HANDOFF.md`, the setup cost (~15 min,
one time) is worth it over Render's RAM risk or paying for HF PRO.

### Cloud Run deploy steps (once the Dockerfile above exists)

```bash
gcloud auth login
gcloud config set project <PROJECT_ID>
gcloud services enable run.googleapis.com artifactregistry.googleapis.com

gcloud artifacts repositories create aigc-detect --repository-format=docker \
    --location=us-central1   # a free-tier region

gcloud builds submit --tag us-central1-docker.pkg.dev/<PROJECT_ID>/aigc-detect/app

gcloud run deploy aigc-detect \
    --image us-central1-docker.pkg.dev/<PROJECT_ID>/aigc-detect/app \
    --region us-central1 \
    --memory 2Gi --cpu 2 \
    --allow-unauthenticated \
    --port 8080   # Cloud Run injects $PORT=8080; matches the app.py change in item 2
```

`--allow-unauthenticated` is required for a public demo link (Cloud Run
defaults to requiring an identity token otherwise). `--memory 2Gi` gives
headroom over the realistic ~1.2 GB peak; can be tuned down after watching
actual usage in the Cloud Run metrics tab.

## Runtime/ops notes for a public demo

- **No auth, no rate limiting** on `/api/analyze` / `/api/analyze-batch`.
  Fine for a hackathon demo link; flag as a known limitation rather than
  building rate limiting today. `MAX_BATCH_FILES = 50` and
  `MAX_UPLOAD_BYTES = 25 MB` (`app.py:27-28`) already bound the worst case
  per request.
- **Stateless** — no DB, no persistent volume needed. Redeploy = full
  rollback; there's no data migration risk.
- **Health check**: `/api/health` (`app.py:109-111`) already exists — wire
  it into whatever the hosting platform's health-check config expects.
- **Cold start**: with the backbone baked into the image (item 3 above),
  cold start is dominated by loading ~350 MB of weights into memory once
  per container start, not a network fetch — should be a few seconds, not
  the 30-60s+ a live download would cost.

## Deployment checklist (when ready to execute)

1. Trim/confirm `requirements-serve.txt` (or accept full `requirements.txt`
   + system libs for opencv).
2. Write `Dockerfile` + `.dockerignore`.
3. Add the `StaticFiles` mount to `app.py`.
4. `docker build` locally, run the container, hit `/api/health` and
   `/api/analyze` with a real image, load `/` in a browser — verify parity
   with the current `uvicorn` + `vite dev` setup before pushing anywhere.
5. Set up the GCP project/billing account, then run the Cloud Run deploy
   steps above (`gcloud builds submit` + `gcloud run deploy`).
6. Verify the live URL: health check, one clean analyze, one batch analyze,
   both checkpoints (`aug`/`baseline`).
7. Update `README.md`'s "Run it" section with the live demo link.

## Out of scope for today

- **Chrome extension**: `extension/background.js:43` hardcodes
  `http://localhost:8000`. It'll keep working for anyone running the
  backend locally, but won't hit the deployed instance until that URL is
  swapped for the public one — a one-line change, worth doing once the
  deployed URL is known, but the extension isn't going through Chrome Web
  Store review today regardless (that review process alone takes days), so
  it's not on the deploy critical path.
- **GPU/scaling**: current model is CPU-fine (frozen backbone, linear head,
  no training in the request path); no autoscaling story needed for a demo.
- **Custom domain**: the platform-issued URL is enough for a submission
  link; can be layered on later without touching the deploy itself.
