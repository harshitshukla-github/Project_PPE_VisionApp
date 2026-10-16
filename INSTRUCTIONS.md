# INSTRUCTIONS.md — Building the PPB Mobile App (React Native)

> **Who this is for:** Claude Code, working with a student developer (Priyanka)
> on Phase 3 of the PPB PPE Compliance System. Read this file fully before
> writing any code. Treat it as the source of truth for the mobile app;
> `agents.md` is the source of truth for the overall system.

---

## 1. Context — what already exists (DO NOT rebuild or modify)

The backend is **complete, tested (93% coverage), and running**. It is a
FastAPI server in `src/api/` of this repo. The mobile app is a **client only**.

**Rules for the backend:**
- Never modify files under `src/`, `configs/`, or `tests/` for app work.
  If the API seems to be missing something, stop and ask Harshit first.
- To run the backend locally: `uv run python cli.py serve --model yolo26n`
  (nano model — works on low-RAM laptops; ~30 ms/image on CPU).
- Swagger UI at `http://localhost:8000/docs` — use it to verify any
  assumption about the API before coding against it.

**What the system does:** a worker takes a selfie/photo before starting a
job → the app sends it to the backend with their worker ID and job category
→ the backend runs YOLO detection → returns **PASS / FAIL / NEEDS_REVIEW** with detected PPE,
missing equipment, and (on FAIL) a friendly Gemini-written explanation.

## 2. The app to build

A simple, robust compliance-check app for workers. Four screens:

1. **Home / Identify** — worker enters (or the app remembers) their
   `worker_id`; picks a `job_category` from a dropdown **populated from
   `GET /api/v1/jobs`** (never hardcode the job list).
2. **Capture** — camera preview → take photo (or pick from gallery for
   testing). Front camera default, full-body framing hint overlay.
3. **Result** — three verdicts, three distinct treatments:
   - **PASS**: big green check, detected items list with confidence.
   - **FAIL**: big red cross, the `explanation` text prominently, list of
     `missing_equipment` and `violations_detected`, and
     `unverifiable_equipment` shown separately as "verify manually"
     (amber, NOT red — these are not failures).
   - **NEEDS_REVIEW**: amber, **not red**. Neither the detector nor Gemini
     could confirm one or more required items, and neither found evidence
     of a violation. Wording must not accuse: "Couldn't verify: safety vest
     — ask your supervisor to confirm", never "You are missing a safety
     vest". `explanation` is null here, so render `missing_equipment`
     directly. Offer "Retake photo" as the primary action, since a clearer
     shot often resolves it.
   - A "Check again" button looping back to Capture.
   - **Always show a loading state while the check runs.** When the detector
     is unsure, the backend asks Gemini to look at the photo before it
     answers, which adds roughly 3–8 s. The app cannot know in advance
     which checks will take that path.

### 3.1 How the verdict is reached (read before building the Result screen)

A worker *wearing* a vest was once shown a red FAIL, because the backend
treated "not detected" as "not worn". Two things changed:

1. `FAIL` requires positive evidence of a breach. `NEEDS_REVIEW` covers
   everything genuinely unknown.
2. When the detector cannot find a required item, the backend sends the
   photo to Gemini for a second opinion on just those items. Gemini's
   answer usually settles it as `PASS` or `FAIL`; `NEEDS_REVIEW` is what
   remains when Gemini could not see the item either (out of frame, too
   dark, photo too small) or Gemini was unreachable.

`verdict_source` tells you who decided: `"detector"` or
`"detector+gemini"`. `verification.items` carries Gemini's finding and a
one-sentence reason per item — worth showing on the Result screen ("Seen:
orange hi-vis vest on torso"), because it tells the worker what the system
actually looked at.

Rendering NEEDS_REVIEW as a failure would reintroduce the original bug.
Treat it as "ask a human", not "you broke the rules".

**Privacy:** the second opinion sends the worker's photo to Google
(Gemini). The first version of this API sent Gemini text only. If the app
shows a consent or privacy notice, it needs to say so.
4. **History** — past checks stored **locally on-device** (AsyncStorage):
   timestamp, job, verdict. No backend persistence exists yet — do not
   invent endpoints for it.

Nice-to-have (only after the above works end-to-end): draw `bbox` overlays
on the result photo using the `detections` array.

## 3. Tech choices (decided — don't relitigate)

| Choice | Decision | Why |
|---|---|---|
| Framework | **Expo (managed workflow), TypeScript** | Student-friendly, no Android Studio/Xcode needed to start; Expo Go on a real phone |
| Navigation | expo-router | File-based, simple |
| Camera | expo-camera + expo-image-picker | Picker enables emulator/desktop testing |
| State | React hooks + context; no Redux | App is small |
| Storage | @react-native-async-storage/async-storage | History + remembered worker_id |
| HTTP | `fetch` with `FormData` | No axios needed |
| Location | New top-level folder **`mobile/`** in this repo | Keep monorepo simple |

Create the project with:
```bash
npx create-expo-app@latest mobile --template blank-typescript
```

## 4. API contract (verified against the running server)

Base URL: `http://<SERVER_IP>:8000/api/v1`

### The critical endpoint — POST /compliance/check
`multipart/form-data`:
- `image` — the photo file (JPEG). In RN FormData:
  `{ uri, name: "photo.jpg", type: "image/jpeg" }`
- `worker_id` — string. Once the app has worker login, send
  `Authorization: Bearer <token>` instead: the server then takes the worker
  from the token and ignores this field. Until the server is switched to
  `PPB_WORKER_AUTH=required`, requests without a token keep working as now.
  Full login contract: [`docs/auth-api.md`](docs/auth-api.md).
- `job_category` — string key from `GET /jobs`

Response `200`:
```json
{
  "worker_id": "W-0042",
  "job_category": "welder",
  "timestamp": "2026-07-29T13:08:34Z",
  "verdict": "PASS",
  "confidence_avg": 0.85,
  "detections": [
    {"class": "hardhat", "confidence": 0.85, "bbox": [376.1, 110.5, 518.0, 223.5]}
  ],
  "missing_equipment": [],
  "violations_detected": [],
  "unverifiable_equipment": ["face_shield"],
  "explanation": null,
  "model_used": "yolo26n",
  "inference_time_ms": 30.1,
  "gemini_invoked": false
}
```

Semantics the UI must respect:
- `verdict` is `"PASS"`, `"FAIL"`, or `"NEEDS_REVIEW"` — nothing else.
  **⚠️ BREAKING CHANGE (2026-10-05):** `NEEDS_REVIEW` is new. Clients built
  against the old two-value contract will fall through both branches and
  render nothing. See §3.1.
- `explanation` is **null on PASS and on NEEDS_REVIEW**; only FAIL has text.
- `verdict_source` is `"detector"` or `"detector+gemini"`.
- `verification` is `null` unless Gemini looked at the photo. When present:
  `{model, elapsed_ms, items: [{item, status, confidence, reason}]}` with
  `status` one of `"worn"`, `"not_worn"`, `"uncertain"`.
- `missing_equipment` — on FAIL, items confirmed not worn; on NEEDS_REVIEW,
  items nobody could confirm.
- `unverifiable_equipment` = required by the job but could not be checked
  from this photo (no training data for the class, or — on a FAIL — Gemini
  could not see it clearly) → **amber "verify manually"**, never a failure.
- `gemini_invoked` is `true` when Gemini was called for the second opinion,
  the explanation, or both. Response time ranges from under 1 s (detector
  alone) to roughly 10 s (second opinion plus explanation).
- `bbox` = `[x1, y1, x2, y2]` in pixels of the uploaded image.
- Errors: `400` bad image, `422` invalid `job_category` (body's `detail`
  lists valid ones), `503` job requirements temporarily unavailable (the
  server's database is down and it has no copy yet; show "try again
  shortly", not a verdict), network failure = server unreachable (see §5).

### GET /jobs
Returns a **flat map** — job key → requirements (no wrapper object, no
display names):
```json
{
  "welder": { "required": ["hardhat", "safety_vest", "goggles", "gloves", "boots"], "pending": ["face_shield"] },
  "general_labor": { "required": ["hardhat", "safety_vest", "boots"], "pending": [] }
}
```
- Dropdown values = the object keys. For labels, title-case the key in the
  app (`general_labor` → "General Labor").
- `boots` is in `required` for every job except `painter`. The worker's
  feet must be in the photo: if they are cut off, the check comes back
  `NEEDS_REVIEW` with `boots` in `missing_equipment`, so keep the full-body
  framing hint on the Capture screen.
- `required` = what the check enforces; `pending` = equipment the model
  can't detect yet (these surface as `unverifiable_equipment` in check
  results). Cache in memory; refetch on pull-to-refresh.
- The list is now managed centrally (in the server's database, soon edited
  from the admin dashboard), so jobs can be added, renamed or removed
  between app launches. Never hardcode job keys, and refetch before a check
  if the cached list is old. Can also return `503` (see above).

### GET /health
`{ "status": "ok", "model": "...", ... }` — use for the connection
indicator on the Home screen.

## 5. The #1 gotcha: reaching the server from a phone

`localhost` on the phone is the phone itself. The student WILL hit this.
- Laptop and phone must be on the **same Wi-Fi**.
- Serve with `--host 0.0.0.0` (the CLI already does this by default).
- Find the laptop's LAN IP (`ipconfig` → IPv4, e.g. `192.168.1.5`) and use
  `http://192.168.1.5:8000`.
- Windows Firewall may need to allow inbound port 8000 the first time.
- Make the base URL **editable in a Settings screen / dev menu** and persist
  it in AsyncStorage — do not bury it in code.
- Android emulator special case: host machine is `10.0.2.2`, not localhost.
- Plain `http://` is fine for dev (Expo Go allows cleartext); note that a
  production build would need HTTPS or manifest tweaks — out of scope now.

## 6. Build order (work in these phases; finish and demo each before the next)

1. **Skeleton** — Expo app boots in Expo Go; navigation between 4 empty
   screens; Settings screen with server URL + `GET /health` "Connected ✓"
   check.
2. **Jobs + identity** — job dropdown from `/jobs`; worker_id input,
   remembered across launches.
3. **Capture → check → result** — camera/gallery photo → multipart POST →
   render PASS and FAIL states fully (use the semantics table in §4).
   *This is the demo milestone.*
4. **History** — save each result locally; list screen; tap for detail.
5. **Polish** — bbox overlays, framing hints, loading/error states, empty
   states, app icon.

Keep commits small and per-phase: `feat(mobile): phase 1 — skeleton + health check`.

## 7. Conventions for the mobile code

- TypeScript strict; define types for all API responses in
  `mobile/src/api/types.ts` (mirror §4 exactly).
- One API client module `mobile/src/api/client.ts` — all fetch calls live
  there; screens never call `fetch` directly.
- No secrets in the app: the app talks only to our backend; Gemini and
  Roboflow keys live server-side and must never appear in `mobile/`.
- Handle every await with try/catch → user-readable error UI ("Can't reach
  server — check the URL in Settings"), never a silent failure or raw stack.
- Test on a real phone via Expo Go early and often; don't develop blind.

## 8. When to stop and ask Harshit

- Anything requiring backend changes (new endpoints, response changes,
  auth, persistence).
- Anything requiring paid services, app-store accounts, or EAS builds.
- If model weights are missing (`models/` is gitignored) — the server won't
  start without them; get the 5 MB `yolo26n_ppe_best.pt` from Harshit.
