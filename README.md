# PPB Vision — PPE Compliance System

AI detection & verification pipeline: a worker's photo goes in, a **PASS / FAIL / NEEDS_REVIEW**
PPE compliance verdict comes out — with bounding boxes, missing-equipment list,
and a Gemini-generated explanation on failures.

**Status:** Phase 1 (training + benchmarking) and Phase 2 (inference + API +
tests + export) complete. Next: React Native mobile app (Phase 3).

Blueprint / single source of truth: [`agents.md`](agents.md)

---

## Quick start

```bash
git clone https://github.com/harshitshukla-github/Project_PPE_VisionApp.git
cd Project_PPE_VisionApp
uv sync                         # installs everything (Python 3.10–3.12)
cp .env.example .env            # then fill in your API keys

# Run the test suite (offline, ~20 s)
uv run pytest

# Start the API server (needs trained weights in models/ — ask Harshit)
uv run python cli.py serve --model yolo11l
# Swagger UI: http://localhost:8000/docs
```

> **Low-RAM machines:** serve the nano model instead — nearly as accurate,
> 10x smaller: `uv run python cli.py serve --model yolo26n`

---

## API — what the mobile app calls

Base URL: `http://<server>:8000/api/v1`

| Method | Endpoint | Purpose |
|---|---|---|
| POST | `/compliance/check` | **The main endpoint.** Photo + worker + job → PASS / FAIL / NEEDS_REVIEW |
| POST | `/detect` | Raw detections only (debugging / overlay drawing) |
| GET | `/jobs` | Job categories + required PPE (populate the app's job picker) |
| GET | `/health` | Liveness + loaded model |
| GET | `/model/info` | Model metadata, class list, thresholds |

### POST /compliance/check

`multipart/form-data` with three fields:

| Field | Type | Example |
|---|---|---|
| `image` | file (JPEG/PNG) | camera capture |
| `worker_id` | string | `"W-0042"` |
| `job_category` | string | `"welder"` — must be a key from `GET /jobs` |

Response (`200`):

```json
{
  "worker_id": "W-0042",
  "job_category": "welder",
  "timestamp": "2026-07-15T13:08:34Z",
  "verdict": "FAIL",
  "confidence_avg": 0.847,
  "detections": [
    {"class": "hardhat", "confidence": 0.847, "bbox": [376.1, 110.5, 518.0, 223.5]},
    {"class": "no_vest", "confidence": 0.791, "bbox": [360.0, 240.0, 540.0, 620.0]}
  ],
  "missing_equipment": [],
  "violations_detected": ["no_vest"],
  "unverifiable_equipment": ["face_shield"],
  "explanation": "You're missing your safety vest ...",
  "model_used": "yolo11l",
  "inference_time_ms": 62.3,
  "gemini_invoked": true
}
```

Field notes for the app:

- `verdict` — `"PASS"`, `"FAIL"`, or `"NEEDS_REVIEW"`.
  - `FAIL` requires **positive evidence** of a breach: a violation class
    detected by YOLO (`violations_detected`), or an item Gemini saw was not
    worn (`missing_equipment`).
  - When YOLO cannot find a required item, the photo goes to Gemini for a
    second opinion on those items only. That usually settles the check as
    `PASS` or `FAIL`.
  - `NEEDS_REVIEW` is what is left when neither model could judge an item,
    or Gemini was unavailable. It must never be rendered as a failure.
  - `verdict_source` is `"detector"` or `"detector+gemini"`; `verification`
    holds Gemini's per-item `status`, `confidence` and one-line `reason`.
  - `explanation` is non-null only on FAIL.
  - Set `PPB_GEMINI_VERIFY=false` to switch the second opinion off. It sends
    the worker's photo to Google, and adds roughly 3–8 s to checks that
    need it.
- `bbox` — `[x1, y1, x2, y2]` in pixels of the **uploaded** image; use with
  `/detect`'s `image_width`/`image_height` if you draw overlays.
- `violations_detected` — explicit violation classes seen in frame
  (`no_hardhat`, `no_vest`). A violation fails the check even if the matching
  item is also detected (multi-person frames).
- `unverifiable_equipment` — required by the job but not yet detectable by
  the model (no training data). Show as "verify manually", not as a failure.
  Today that is `face_shield`, `harness` and `ear_protection` only. `boots`
  is enforced like any other item: the detector looks first, and when it
  finds none Gemini checks the photo. Feet out of frame gives `NEEDS_REVIEW`.
- Errors: `400` undecodable image, `422` unknown `job_category` (message
  lists valid ones).

---

## Trained models (test-split benchmark)

| Model | mAP50 | mAP50-95 | Latency (RTX 3050) | Size | Role |
|---|---|---|---|---|---|
| **yolo11l** | 0.713 | **0.471** | 52 ms | 51 MB | **Production backend** |
| yolo26l | 0.705 | 0.458 | 48 ms | 106 MB | Backup / NMS-free export |
| **yolo26n** | 0.674 | 0.422 | **7.4 ms** | **5.4 MB** | **Edge / low-RAM** |
| yolo11n | 0.672 | 0.424 | 8.0 ms | 5.5 MB | Edge comparison |

Full comparison: `runs/benchmark_report.md` (generate with
`uv run python cli.py benchmark`).

Weight files (`models/*.pt`) are **not in git** (too large) — get them from
Harshit or retrain (`cli.py train`). Known model limitations: `no_vest`
accuracy is weak (AP 0.27), `boots` is modest (AP 0.61, so Gemini settles
many boots checks) and three classes (`ear_protection`, `face_shield`,
`harness`) have no training data yet.

---

## CLI reference

```bash
uv run python cli.py data pipeline        # download → remap → split → validate
uv run python cli.py train --model yolo11l --epochs 50
uv run python cli.py benchmark            # test-split comparison + report
uv run python cli.py export --model yolo26n --half --int8   # ONNX for edge
uv run python cli.py serve --model yolo11l --port 8000
uv run python cli.py info                 # pipeline status check
uv run python cli.py db setup             # create the admin system's database
uv run python cli.py db status            # check .env can connect
```

## Database (admin system)

The admin dashboard stores its data in PostgreSQL (tested on 18). On each
machine, once:

```bash
uv run python cli.py db setup
```

It asks for the password of your existing `postgres` superuser, which is
used for that one connection and never stored. It creates:

- a `ppb_app` login role with no superuser rights, its password generated
  and written only to `.env`
- `ppb`, the application database, and `ppb_test`, which the test suite may
  wipe, both owned by `ppb_app`

and adds `DATABASE_URL` and `TEST_DATABASE_URL` to `.env`. Re-running it is
safe: it reuses the password already in `.env` and repairs anything missing.
Use `--superuser`, `--host` or `--port` if your server differs from
`postgres@localhost:5432`.

Then create the tables and load the job matrix:

```bash
uv run python cli.py db migrate           # safe to re-run; also after every pull
uv run python cli.py db seed              # loads configs/job_matrix.yaml once
uv run python cli.py db status            # connection, schema revision, job count
```

**Where the job matrix comes from.** With `DATABASE_URL` set, the API reads
the matrix from Postgres, and the server refuses to start until it has been
migrated and seeded. `db seed` only fills an empty database: after that the
database is the source of truth (the admin dashboard will edit it there), and
`configs/job_matrix.yaml` is not applied again. Without `DATABASE_URL`, the
API reads the YAML file as before.

The server re-reads the matrix every `PPB_MATRIX_CACHE_S` seconds (default
30). If Postgres stops answering it keeps using the last matrix it read; if
it never read one, `/jobs` and `/compliance/check` return 503.

The database tests wipe and rebuild the database in `TEST_DATABASE_URL`
before each test, and refuse to run unless its name ends in `_test`.

## Login (admin system)

The contract is [`docs/auth-api.md`](docs/auth-api.md). In short: admins and
supervisors sign in to the dashboard with email + password (cookie session);
workers sign in to the app with a worker ID + 6-digit PIN (bearer token).
There is no sign-up. Until the dashboard exists, accounts come from the CLI:

```bash
uv run python cli.py db migrate                                        # creates the login tables
uv run python cli.py auth create-admin --email you@example.com --name "Your Name"
uv run python cli.py auth create-worker --worker-id W-0042 --name "Asha Verma"   # prints a one-time PIN
uv run python cli.py auth reset-pin --worker-id W-0042                 # new one-time PIN, unlocks
```

`PPB_WORKER_AUTH=optional` (the default) keeps today's app working: a check
without a token uses the `worker_id` form field. Switch it to `required` only
after the app ships its login screen. Set `PPB_COOKIE_SECURE=false` only to
use the dashboard over plain `http://` on a LAN address.

`tests/test_auth_api.py` calls every route without credentials; a new route
that forgets its login check fails CI until it is either protected or added
to the test's list of public routes.

## Project layout

```
configs/          data.yaml, per-model hyperparams, job_matrix.yaml
src/data/         dataset download / remap / split / validate
src/training/     unified YOLO26+YOLO11 trainer
src/evaluation/   benchmark, plots, report
src/inference/    detector, compliance logic, Gemini explainer, schemas
src/api/          FastAPI server (this is what the mobile app talks to)
src/export/       ONNX export + INT8 quantization
tests/            57 offline tests, 93% coverage — `uv run pytest`
```

## Environment variables (.env)

Required: `ROBOFLOW_API_KEY` (data download), `GOOGLE_API_KEY` (Gemini
explanations — optional; a fallback text is used without it).
Common: `PPB_MODEL_NAME=yolo11l`, `PPB_DEVICE=0` (or `cpu`).
Full list in [`.env.example`](.env.example).
