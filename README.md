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
  "unverifiable_equipment": ["boots"],
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
accuracy is weak (AP 0.24) and four classes (`boots`, `ear_protection`,
`face_shield`, `harness`) have no training data yet.

---

## CLI reference

```bash
uv run python cli.py data pipeline        # download → remap → split → validate
uv run python cli.py train --model yolo11l --epochs 50
uv run python cli.py benchmark            # test-split comparison + report
uv run python cli.py export --model yolo26n --half --int8   # ONNX for edge
uv run python cli.py serve --model yolo11l --port 8000
uv run python cli.py info                 # pipeline status check
```

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
