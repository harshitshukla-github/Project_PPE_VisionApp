# PPB Vision Module — agents.md

## Project Identity

| Field           | Value                                                        |
|-----------------|--------------------------------------------------------------|
| **Project**     | PPB — PPE (Personal Protective Equipment) Compliance System  |
| **Module**      | `ppb-vision` — AI Detection & Verification Pipeline          |
| **Owner**       | Harshit                                                      |
| **Created**     | 2026-03-19                                                   |
| **Python**      | ≥3.10, <3.13                                                 |
| **Env Manager** | `uv` (https://docs.astral.sh/uv/)                           |
| **License**     | Private / TBD                                                |

---

## 1. What This Module Does

The vision module is the **AI brain** of PPB. It receives a worker's photo (or video frame), detects every piece of safety equipment on their body, compares it against a job-specific PPE checklist, and returns a **PASS / FAIL** verdict with human-readable reasoning.

### Core Responsibilities

- **Detect** PPE items in an image: hardhat, safety vest, goggles, gloves, mask, steel-toe boots, ear protection, face shield, harness.
- **Classify** each detected item's compliance state: `worn_correctly`, `worn_incorrectly`, `missing`.
- **Map** detections against a **Job PPE Matrix** (e.g., "Welder" requires goggles + gloves + face shield + vest).
- **Return** structured JSON with bounding boxes, confidence scores, compliance status, and (optionally) a Gemini-generated natural-language explanation of what's missing and why.

---

## 2. Dual Model Strategy — YOLO26 vs YOLO11

We are building **both** YOLO26 and YOLO11 pipelines in parallel. The final production model will be selected based on benchmark results on our PPE dataset, budget constraints, and deployment target (cloud GPU vs edge device).

### 2.1 Why Both?

| Factor                  | YOLO26                                      | YOLO11                                      |
|-------------------------|---------------------------------------------|---------------------------------------------|
| **Architecture**        | NMS-free end-to-end, DFL removed            | Traditional NMS post-processing             |
| **Best at**             | Large / X-Large scales (L, X)               | Nano / Small scales (N, S)                  |
| **PPE mAP advantage**   | +1.3% to +3.1% at L/X scale                | Better at N/S scale on PPE benchmarks       |
| **CPU inference**       | Up to 43% faster (ONNX)                     | Baseline                                    |
| **Optimizer**           | MuSGD (SGD + Muon hybrid)                   | Standard SGD / AdamW                        |
| **Edge deployment**     | Simplified (no NMS node in ONNX graph)      | Mature, widely tested                       |
| **Ultralytics package** | Same `ultralytics` pip package              | Same `ultralytics` pip package              |
| **Model loading**       | `YOLO("yolo26l.pt")`                        | `YOLO("yolo11l.pt")`                        |

### 2.2 Decision Matrix (to be filled after benchmarking)

| Scenario                              | Recommended Model   | Reason                            |
|---------------------------------------|---------------------|-----------------------------------|
| Backend GPU inference (production)    | YOLO26-L or YOLO26-X | Best accuracy at large scale     |
| On-device / edge / mobile             | YOLO11-N or YOLO11-S | Better accuracy at small scale   |
| Budget-constrained (CPU-only server)  | YOLO26-M             | 43% faster CPU inference         |
| Rapid prototyping (P1)               | Either Nano variant  | Fastest iteration cycle          |

### 2.3 Model Variants We Will Train

```
YOLO26 variants:   yolo26n, yolo26s, yolo26m, yolo26l, yolo26x
YOLO11 variants:   yolo11n, yolo11s, yolo11m, yolo11l, yolo11x

Primary training targets (first pass):
  - yolo26l.pt    → production backend candidate
  - yolo26n.pt    → edge / mobile candidate
  - yolo11l.pt    → production backend comparison
  - yolo11n.pt    → edge / mobile comparison
```

---

## 3. PPE Detection Classes

These are the target classes for fine-tuning. The dataset will use YOLO-format TXT annotations.

| Class ID | Label              | Category         | Notes                                   |
|----------|--------------------|------------------|-----------------------------------------|
| 0        | `hardhat`          | Head protection  | Includes standard and full-brim styles  |
| 1        | `no_hardhat`       | Head violation   | Person's head visible, no helmet        |
| 2        | `safety_vest`      | Body protection  | High-vis vest / reflective jacket       |
| 3        | `no_vest`          | Body violation   | Torso visible, no vest                  |
| 4        | `goggles`          | Eye protection   | Safety goggles or glasses               |
| 5        | `mask`             | Face protection  | Dust mask, N95, surgical                |
| 6        | `gloves`           | Hand protection  | Safety / work gloves                    |
| 7        | `boots`            | Foot protection  | Steel-toe / safety boots                |
| 8        | `ear_protection`   | Hearing          | Earmuffs or earplugs (visible)          |
| 9        | `face_shield`      | Face protection  | Full face shield                        |
| 10       | `harness`          | Fall protection  | Safety harness for height work          |
| 11       | `person`           | Anchor class     | Person bounding box (reference)         |

---

## 4. Dataset Strategy

### 4.1 Primary Sources

| Source                                          | Images  | Classes                          | Format     |
|-------------------------------------------------|---------|----------------------------------|------------|
| Roboflow — PPE Combined Model (v8)              | ~44,000 | helmet, vest, mask, gloves, etc. | YOLO TXT   |
| Roboflow — PPE Annotation (v7)                  | ~10,129 | PPE multi-class                  | YOLO TXT   |
| Roboflow — Construction Site Safety             | ~2,801  | hardhat, mask, vest, person      | YOLO TXT   |
| Kaggle — Safety Helmet & Reflective Jacket      | ~5,000  | helmet, no-helmet, vest, no-vest | YOLO TXT   |
| Custom captures (Phase 2)                       | TBD     | All 12 classes                   | YOLO TXT   |

### 4.2 Dataset Pipeline

```
Download (Roboflow API / Kaggle)
    │
    ▼
Class Remapping ──► Unify labels to our 12-class schema
    │
    ▼
Quality Filter ──► Remove blurry / mislabeled / duplicate images
    │
    ▼
Augmentation Config ──► albumentations pipeline (see Section 7)
    │
    ▼
Split ──► train (80%) / val (10%) / test (10%)
    │
    ▼
data.yaml ──► Single config for both YOLO26 and YOLO11 training
```

---

## 5. Project Structure

```
ppb-vision/
├── agents.md                    # ◄ THIS FILE — project blueprint
├── requirements.txt             # Python dependencies
├── pyproject.toml               # uv project config (auto-generated)
├── .env.example                 # Environment variable template
├── .gitignore
│
├── configs/                     # Training & inference configurations
│   ├── data.yaml                # Dataset paths + class names
│   ├── yolo26l_train.yaml       # YOLO26-L hyperparameters
│   ├── yolo26n_train.yaml       # YOLO26-N hyperparameters
│   ├── yolo11l_train.yaml       # YOLO11-L hyperparameters
│   ├── yolo11n_train.yaml       # YOLO11-N hyperparameters
│   └── augmentation.yaml        # Albumentations pipeline config
│
├── src/
│   ├── __init__.py
│   │
│   ├── data/                    # Dataset management
│   │   ├── __init__.py
│   │   ├── downloader.py        # Roboflow / Kaggle dataset download
│   │   ├── remapper.py          # Unify class labels across datasets
│   │   ├── splitter.py          # Train/val/test split
│   │   └── validator.py         # Annotation integrity checks
│   │
│   ├── training/                # Model training
│   │   ├── __init__.py
│   │   ├── trainer.py           # Unified trainer (accepts model_name arg)
│   │   ├── callbacks.py         # MLflow / custom logging callbacks
│   │   └── hyperparams.py       # Hyperparameter loading & validation
│   │
│   ├── evaluation/              # Model benchmarking
│   │   ├── __init__.py
│   │   ├── benchmark.py         # Compare YOLO26 vs YOLO11 metrics
│   │   ├── visualize.py         # Confusion matrix, PR curves, side-by-side
│   │   └── report.py            # Generate markdown benchmark report
│   │
│   ├── inference/               # Production inference
│   │   ├── __init__.py
│   │   ├── detector.py          # Core detection class (model-agnostic)
│   │   ├── compliance.py        # PPE checklist verification logic
│   │   ├── gemini_explainer.py  # Gemini 2.5 Flash integration (hybrid)
│   │   └── schemas.py           # Pydantic response models
│   │
│   ├── export/                  # Model export for deployment
│   │   ├── __init__.py
│   │   ├── onnx_export.py       # Export to ONNX format
│   │   └── quantize.py          # FP16 / INT8 quantization
│   │
│   └── api/                     # FastAPI inference server
│       ├── __init__.py
│       ├── main.py              # FastAPI app entry point
│       ├── routes.py            # /detect, /compliance, /health endpoints
│       ├── deps.py              # Model loading, dependency injection
│       └── middleware.py         # CORS, rate limiting, logging
│
├── cli.py                       # Typer CLI: train, eval, export, serve
│
├── notebooks/                   # Jupyter notebooks for EDA & prototyping
│   ├── 01_dataset_exploration.ipynb
│   ├── 02_training_yolo26.ipynb
│   ├── 03_training_yolo11.ipynb
│   ├── 04_benchmark_comparison.ipynb
│   └── 05_gemini_hybrid_test.ipynb
│
├── tests/
│   ├── test_detector.py
│   ├── test_compliance.py
│   ├── test_api.py
│   └── conftest.py              # Fixtures: sample images, mock models
│
├── data/                        # .gitignore'd — local dataset storage
│   ├── raw/                     # Downloaded datasets
│   ├── processed/               # Remapped + cleaned
│   └── splits/                  # train / val / test
│
├── models/                      # .gitignore'd — trained model weights
│   ├── yolo26l_ppe_best.pt
│   ├── yolo26n_ppe_best.pt
│   ├── yolo11l_ppe_best.pt
│   ├── yolo11n_ppe_best.pt
│   └── exports/                 # ONNX / TensorRT exports
│
└── runs/                        # .gitignore'd — training run artifacts
    └── ...                      # auto-generated by ultralytics
```

---

## 6. Architecture — AI Pipeline Flow

```
┌─────────────────────────────────────────────────────────────┐
│                     WORKER MOBILE APP                       │
│               (React Native + Expo Camera)                  │
└──────────────────────┬──────────────────────────────────────┘
                       │  POST /api/v1/compliance/check
                       │  (image + worker_id + job_category)
                       ▼
┌─────────────────────────────────────────────────────────────┐
│                     FASTAPI BACKEND                         │
│                                                             │
│  1. Validate request (Pydantic)                             │
│  2. Load image → preprocess (resize, normalize)             │
│  3. Run YOLO inference ─────────────────────┐               │
│     ┌───────────────┐  ┌───────────────┐    │               │
│     │   YOLO26-L    │  │   YOLO11-L    │◄───┘               │
│     │  (production) │  │  (fallback)   │  ← model selected  │
│     └───────┬───────┘  └───────┬───────┘    via config      │
│             │                  │                             │
│             ▼                  ▼                             │
│     4. Parse detections → PPEDetectionResult                │
│                                                             │
│     5. Compliance check against Job PPE Matrix              │
│        ┌─────────────────────────────────────┐              │
│        │  Job: "Welder"                      │              │
│        │  Required: hardhat, goggles, gloves,│              │
│        │           vest, face_shield, boots   │              │
│        │  Detected: hardhat ✓, goggles ✓,    │              │
│        │           gloves ✗, vest ✓           │              │
│        │  → FAIL (missing: gloves,           │              │
│        │    face_shield, boots)               │              │
│        └─────────────────────────────────────┘              │
│                                                             │
│     6. IF all_pass → return PASS (YOLO-only, no Gemini)     │
│        IF fail → call Gemini 2.5 Flash for explanation      │
│        ┌─────────────────────────────────────┐              │
│        │  Gemini Prompt:                     │              │
│        │  "Worker is missing: gloves,        │              │
│        │   face_shield, boots.               │              │
│        │   Job: Welder. Explain each risk."  │              │
│        └─────────────────────────────────────┘              │
│                                                             │
│     7. Return ComplianceResponse JSON                       │
└─────────────────────────────────────────────────────────────┘
                       │
                       ▼
              ┌────────────────┐
              │  PostgreSQL    │  ← store result + timestamp
              │  GCP Storage   │  ← store annotated image
              └────────────────┘
```

**Cost optimization**: When YOLO detects all required PPE → instant PASS. Gemini is **only** invoked on failures to generate human-readable explanations. This keeps the per-check cost near zero for compliant workers.

---

## 7. Training Plan

### 7.1 Phase 1 — Baseline Training (Current Sprint)

| Step | Task                                            | Tool / Command                                |
|------|-------------------------------------------------|-----------------------------------------------|
| 1    | Create uv environment                           | `uv init && uv sync`                          |
| 2    | Download PPE dataset from Roboflow              | `src/data/downloader.py`                       |
| 3    | Remap classes to 12-class schema                | `src/data/remapper.py`                         |
| 4    | Validate annotations                            | `src/data/validator.py`                        |
| 5    | Train YOLO26-L on PPE (100 epochs, imgsz=640)   | `cli.py train --model yolo26l --epochs 100`    |
| 6    | Train YOLO11-L on PPE (100 epochs, imgsz=640)   | `cli.py train --model yolo11l --epochs 100`    |
| 7    | Train YOLO26-N on PPE (100 epochs, imgsz=640)   | `cli.py train --model yolo26n --epochs 100`    |
| 8    | Train YOLO11-N on PPE (100 epochs, imgsz=640)   | `cli.py train --model yolo11n --epochs 100`    |
| 9    | Run benchmark comparison                        | `cli.py benchmark`                             |
| 10   | Select production model                         | Based on mAP, latency, cost analysis           |

### 7.2 Hyperparameter Defaults

```yaml
# Shared across both YOLO26 and YOLO11
epochs: 100
imgsz: 640
batch: 16               # adjust based on GPU VRAM
patience: 20             # early stopping
lr0: 0.01
lrf: 0.01
momentum: 0.937
weight_decay: 0.0005
warmup_epochs: 3
warmup_momentum: 0.8
box: 7.5
cls: 0.5
dfl: 1.5                # ignored by YOLO26 (DFL removed)
hsv_h: 0.015
hsv_s: 0.7
hsv_v: 0.4
degrees: 10.0
translate: 0.1
scale: 0.5
fliplr: 0.5
mosaic: 1.0
mixup: 0.15
```

### 7.3 Augmentation Pipeline (albumentations)

```yaml
train_augmentations:
  - RandomBrightnessContrast:
      brightness_limit: 0.2
      contrast_limit: 0.2
      p: 0.5
  - GaussianBlur:
      blur_limit: [3, 7]
      p: 0.3
  - MotionBlur:
      blur_limit: 7
      p: 0.2
  - RandomRain:
      p: 0.1
  - RandomFog:
      p: 0.1
  - RandomSunFlare:
      p: 0.05
  - CLAHE:
      clip_limit: 4.0
      p: 0.3
```

These simulate real construction-site conditions: varying lighting, weather, dust, and motion blur from handheld phone cameras.

---

## 8. Evaluation Metrics

After training all 4 primary model variants, we generate a benchmark report comparing:

| Metric                 | What It Tells Us                                  |
|------------------------|---------------------------------------------------|
| mAP@0.5               | Overall detection accuracy at IoU 0.5             |
| mAP@0.5:0.95          | Stricter accuracy across IoU thresholds           |
| Per-class AP           | Which PPE items are hardest to detect             |
| Inference latency (ms) | GPU (T4/A100) and CPU times                       |
| Model size (MB)        | .pt weight file size                              |
| ONNX export size       | Deployment artifact size                          |
| FPS (frames/sec)       | Throughput for potential video mode (P3)           |
| Confusion matrix       | False positives/negatives per class               |
| F1 confidence curve    | Optimal confidence threshold                      |

The benchmark report is auto-generated as a Markdown file in `runs/benchmark_report.md`.

---

## 9. Inference API Endpoints

| Method | Endpoint                    | Description                              |
|--------|-----------------------------|------------------------------------------|
| POST   | `/api/v1/detect`            | Raw detection — returns bounding boxes   |
| POST   | `/api/v1/compliance/check`  | Full compliance check with PASS/FAIL     |
| GET    | `/api/v1/health`            | Service health + loaded model info       |
| GET    | `/api/v1/model/info`        | Current model version, class list        |

### Sample Response — `/api/v1/compliance/check`

```json
{
  "worker_id": "W-0042",
  "job_category": "welder",
  "timestamp": "2026-03-19T14:30:00Z",
  "verdict": "FAIL",
  "confidence_avg": 0.87,
  "detections": [
    {"class": "hardhat", "confidence": 0.94, "status": "worn_correctly", "bbox": [120, 30, 220, 130]},
    {"class": "safety_vest", "confidence": 0.91, "status": "worn_correctly", "bbox": [100, 200, 300, 500]},
    {"class": "goggles", "confidence": 0.82, "status": "worn_correctly", "bbox": [140, 80, 210, 120]}
  ],
  "missing_equipment": ["gloves", "face_shield", "boots"],
  "explanation": "Worker is missing critical hand protection (gloves) required for welding due to spark and heat exposure. A face shield is mandatory to protect against UV radiation and flying debris. Steel-toe boots are required in all welding zones per OSHA 1910.136.",
  "model_used": "yolo26l_ppe_v1",
  "inference_time_ms": 23.4,
  "gemini_invoked": true
}
```

---

## 10. Environment Setup — Quick Start

```bash
# 1. Install uv (if not already installed)
curl -LsSf https://astral.sh/uv/install.sh | sh

# 2. Clone and enter project
cd ppb-vision

# 3. Create virtual environment + install all deps
uv init
uv add -r requirements.txt
uv sync

# 4. Copy env template and fill in keys
cp .env.example .env
# Edit .env → add ROBOFLOW_API_KEY, GOOGLE_API_KEY

# 5. Download dataset
uv run python -m src.data.downloader

# 6. Train YOLO26-L
uv run python cli.py train --model yolo26l --epochs 100

# 7. Train YOLO11-L
uv run python cli.py train --model yolo11l --epochs 100

# 8. Compare both models
uv run python cli.py benchmark

# 9. Launch inference server
uv run python cli.py serve --model yolo26l --port 8000
```

---

## 11. Environment Variables

```env
# .env.example
ROBOFLOW_API_KEY=rf_xxxxxxxxxxxxxxxx
GOOGLE_API_KEY=AIzaXxxxxxxxxxxxxxxxxxx        # Gemini 2.5 Flash
PPB_MODEL_NAME=yolo26l                         # active model for inference
PPB_CONFIDENCE_THRESHOLD=0.45                  # minimum detection confidence
PPB_IOU_THRESHOLD=0.50                         # NMS IoU (YOLO11 only)
PPB_DEVICE=0                                   # GPU index, or "cpu"
PPB_IMG_SIZE=640                               # inference image size
MLFLOW_TRACKING_URI=http://localhost:5000       # experiment tracking
LOG_LEVEL=INFO
```

---

## 12. Coding Conventions

| Rule                          | Standard                                           |
|-------------------------------|----------------------------------------------------|
| Formatter + Linter            | `ruff` (replaces black + isort + flake8)           |
| Type hints                    | Required on all public functions                   |
| Docstrings                    | Google style                                       |
| Config management             | Pydantic Settings + `.env`                         |
| Model I/O                     | Always through `src/inference/detector.py`          |
| No hardcoded paths            | Everything via config or env vars                  |
| Logging                       | `loguru` — never bare `print()`                    |
| Tests                         | `pytest` — minimum 80% coverage on `src/`          |

---

## 13. Risk Register

| Risk                                       | Impact  | Mitigation                                           |
|--------------------------------------------|---------|------------------------------------------------------|
| PPE classes not in public datasets          | High    | Custom labeling sprint on Roboflow for rare classes  |
| YOLO26 too new, potential bugs              | Medium  | YOLO11 as fallback, pin ultralytics version          |
| Gemini API costs spike on high failure rate | Medium  | Cache explanations per (missing_set, job_category)   |
| Model accuracy drops in low-light / rain   | High    | Augmentation pipeline simulates adverse conditions   |
| On-device inference too slow                | Medium  | Quantize to INT8, use YOLO Nano variant              |

---

## 14. Phased Roadmap

| Phase | Milestone                                 | Status      |
|-------|-------------------------------------------|-------------|
| P0    | Stack selection & architecture            | ✅ Complete  |
| P1    | Vision module — dual YOLO training        | 🔄 Current  |
| P2    | FastAPI inference server + compliance API | ⬜ Next      |
| P3    | React Native mobile app (camera + upload) | ⬜ Planned   |
| P4    | Gemini hybrid explanations                | ⬜ Planned   |
| P5    | Job PPE Matrix + admin panel (Next.js)    | ⬜ Planned   |
| P6    | PostgreSQL + GCP Storage integration      | ⬜ Planned   |
| P7    | Firebase Auth + worker identity           | ⬜ Planned   |
| P8    | Live video mode                           | ⬜ Future    |

---

*This document is the single source of truth for the PPB vision module. Update it as decisions are made.*
