# Phase 6 — Checks, photos and review: API contract

**Status: agreed (draft 2, with the decisions in §10, approved in PR #34).
The backend, the dashboard and the app are built against it from here.** Any
change after that goes through a pull request that edits this file first.
No app release until the photo notice's wording (§5) passes the company's
privacy check. After approval, Priyanka prepares the release build; track
the approval, build and rollout in the [release checklist](release-checklist.md).
Wording-only changes are server changes and need no app update (§5).

This is the contract that `docs/admin-api.md` §10 promised: every check is
stored with the worker's photo, supervisors review the checks the system
couldn't settle, and the photos are kept for future model training
(AGENTS.md §15, roadmap P6). AGENTS.md §15 also sets three conditions on
keeping photos: a consent notice in the app, restricted access, and a way to
delete a worker's photos on request. All three are covered below.

It builds on [`docs/auth-api.md`](auth-api.md) (logins, error shape) and
[`docs/admin-api.md`](admin-api.md) (roles, sites, paging, audit log). Its
conventions apply unchanged: dashboard endpoints are under `/api/v1/admin`,
need the `ppb_session` cookie, and need `X-CSRF-Token` on writes. Timestamps
are UTC with a `Z`.

Numbering: this is Phase 6, as on the roadmap. `admin-api.md` §10, which
called it "Phase 5", now points here.

---

## 1. Who may do what

| Area | admin | supervisor | worker (app) |
|---|:-:|:-:|:-:|
| See checks and their photos | all sites | own site | |
| Review a check (any verdict) | all sites | own site | |
| Replace someone's review | ✓ | | |
| Delete a worker's photos | ✓ | | |
| Agree to, or withdraw from, photos being kept for training | | | own |
| Export photos for training | CLI on the server only | | |

- A supervisor asking for a check at another site gets `404 not_found`, the
  same rule as workers (`admin-api.md` §1).
- The site of a check is **the worker's site when the check was made**, stored
  with the check. Moving a worker later doesn't move their old checks, so a
  supervisor keeps seeing the checks made at their site.
- A check sent without a login, under a worker ID that doesn't exist, has no
  site. Only admins see it.
- Nothing in the mobile app shows other workers' checks. The app's history
  stays on the phone (INSTRUCTIONS.md), and a review's outcome isn't shown
  in the app in this phase (§10, 9).
- Who may see a photo is the control on photos: admins, and the supervisor
  of the check's site. Opening one isn't written to the audit log (§10, 4).

## 2. What is stored for each check

Every `POST /compliance/check` that returns a verdict is stored: the response
the app got, plus what's needed to review it and train on it later.

```json
{
  "id": 48213,
  "at": "2026-12-03T07:41:09Z",
  "worker_id": "W-0042",
  "worker_name": "Asha Verma",
  "site": "pune_tower_b",
  "signed_in": true,
  "job": "welder",
  "job_name": "Welder",
  "job_version": 4,
  "required": ["hardhat", "safety_vest", "goggles", "gloves", "boots"],
  "pending": ["face_shield"],
  "verdict": "NEEDS_REVIEW",
  "verdict_source": "detector+gemini",
  "missing_equipment": ["gloves"],
  "violations_detected": [],
  "unverifiable_equipment": ["face_shield"],
  "detections": [ { "class": "hardhat", "confidence": 0.94, "bbox": [120, 30, 220, 130] } ],
  "verification": { "model": "gemini-2.5-flash", "items": [ … ], "elapsed_ms": 2140 },
  "explanation": null,
  "model_used": "yolo11l",
  "confidence_threshold": 0.45,
  "gemini_verify": true,
  "gemini_min_confidence": 0.75,
  "photo": { "state": "kept", "width": 1080, "height": 1440, "bytes": 412377, "type": "image/jpeg" },
  "training_consent": true,
  "review": null
}
```

- `signed_in` is false for a check sent without a worker token, which happens
  while `PPB_WORKER_AUTH=optional`. Such a check's `worker_id` was typed into
  the app, and the dashboard says so next to it. `worker_name` and `site` are
  `null` when no worker has that ID.
- `required` and `pending` are the job's requirements **when the check was
  made**, as `GET /jobs` gives them, and `job_name` is its name then. The
  `jobs` table only holds today's requirements, so without these a reviewer
  couldn't see what the verdict was judged against. `job_version` stays, to
  find the matching `job.requirements` entry in the audit log.
- `confidence_threshold`, `gemini_verify` and `gemini_min_confidence` are
  the three settings that decide a verdict, as they were at the time.
  Gemini's findings below `gemini_min_confidence` count as "uncertain"
  (`apply_verification` in `src/inference/compliance.py`). An admin can
  change all three from the dashboard, so a reviewer looking at a borderline
  check later needs them.
- `model_used` is the variant name the server reports, such as `"yolo11l"`.
- `photo.state` is `kept`, `deleted` (on request or by the retention rule,
  §6) or `not_stored`: storing failed, photos are switched off (§7), the
  check was sent without a login, whose photo is never kept (§10, 8), or the
  worker hasn't answered the photo notice yet (§5).
- `training_consent` is the worker's choice **when the check was made** (§5),
  set to `false` on all their checks if they later withdraw.
- `review` is `null` until someone reviews the check (§4.3).

**Storing never costs the worker a verdict.** The check is stored after the
verdict is decided. If the database or the photo store fails, the app still
gets its answer, the server logs what wasn't stored, and nothing is retried
from the phone. The response gains one field:

```json
{ "check_id": 48213, "…": "the response as today" }
```

`check_id` is `null` when the check couldn't be stored. The app shows nothing
different either way.

The photo is the uploaded file, not a re-encoded copy, because training needs
the original. Only its **location** (the EXIF GPS tags) is removed before it
is stored (§10, 6); every other tag stays, the **orientation** tag above
all:

- The server decodes the upload with its EXIF orientation applied, and the
  detector's boxes are measured on that. (Harshit checked with the project's
  OpenCV 4.13: a 400×200 JPEG tagged "rotate 90°" decodes as 200×400.)
- Browsers apply the same tag when they show the image, so the dashboard
  draws the boxes on the photo as the browser shows it, and they line up.
- Dropping all of the EXIF would drop the orientation too: a photo taken
  sideways would then be stored sideways, and the boxes wouldn't match it.

`photo.width` and `photo.height` are the size after orientation, the space
the boxes are in.

## 3. Errors

The shape from `auth-api.md` §3. Codes added here:

| HTTP | `code` | When |
|---|---|---|
| 404 | `not_found` | No such check (or one at another site, for a supervisor), or a photo that was never stored (§4.2) |
| 409 | `already_reviewed` | Someone has reviewed it already, and the request isn't an admin's `"replace": true` (§4.3). The body carries the check as it is now in `current`, as with `version_conflict` |
| 409 | `notice_changed` | The app answered an older version of the consent notice. The body carries the current `notice` (§5) |
| 410 | `photo_deleted` | The check had a photo, and it has been deleted |
| 422 | `invalid` | A review without an outcome, a note over 500 characters, `replace` from a supervisor, or a bad filter |
| 422 | `out_of_range` | `photo_training_days` shorter than `photo_review_days` (§7), as for any setting outside its range (`admin-api.md` §6) |

## 4. Dashboard endpoints

### 4.1 `GET /checks?site=&worker=&job=&verdict=&reviewed=&signed_in=&from=&to=&limit=&cursor=`

Newest first, paged as in `admin-api.md` §2, with `total`.

- `verdict` is `PASS`, `FAIL` or `NEEDS_REVIEW`. `reviewed` is `true` or
  `false`, for any verdict, since any check can be reviewed (§4.3).
- `signed_in=false` lists checks sent without a login. With `from`, its
  `total` is the count the switch to `PPB_WORKER_AUTH=required` waits for:
  none for 3 working days (`auth-api.md` §5). The dashboard shows it on the
  Settings page next to `worker_auth`.
- `from` and `to` are timestamps (inclusive, exclusive), as in the audit log.
- A supervisor's results are their own site's, whatever `site` says.
- Each item is the check without `detections` and `verification`, to keep
  the list small. The detail has everything.

The dashboard's default view for a supervisor is
`verdict=NEEDS_REVIEW&reviewed=false`: the checks waiting for a person. The
others are a filter away, for overturning a wrong `FAIL` or flagging a wrong
`PASS`.

### 4.2 `GET /checks/{id}` and `GET /checks/{id}/photo`

`GET /checks/{id}` returns the whole check (§2).

`GET /checks/{id}/photo` returns the image itself, with its own content type
and:

- `Cache-Control: private, no-store`: a shared computer's cache shouldn't
  keep workers' photos.
- `Content-Disposition: inline`, and no file name containing the worker ID.
- `404 not_found` for a check whose photo was never stored
  (`photo.state: "not_stored"`).
- `410 photo_deleted` for a photo that was stored and has since been deleted
  (`photo.state: "deleted"`).

It's an ordinary `/api` URL, so the session cookie (`Path=/api`) goes with it
and the dashboard can use it in an `<img>`. The page's `img-src 'self'`
policy already allows it. Opening a photo isn't recorded in the audit log
(§10, 4).

### 4.3 `POST /checks/{id}/review`

```json
{ "outcome": "compliant", "note": "Gloves are in his left hand, worn once welding starts." }
```

- `outcome` is `compliant` or `not_compliant`. `note` is optional, up to 500
  characters. One decision for the whole check, not one per item (§10, 11).
- **Any verdict can be reviewed** (§10, 10): `NEEDS_REVIEW` above all, but
  also a `FAIL` that accused a compliant worker, or a `PASS` that missed
  something, which is the worst failure for a safety check.
- **Once, unless an admin replaces it** (§10, 3). A check someone has
  already reviewed answers `409 already_reviewed`, with the check as it is
  now in `current`, so two reviewers can't silently overwrite each other. An
  admin may send `"replace": true` to change any review; from a supervisor
  it is `422 invalid`. Every review, first or replacing, is its own
  `check.review` audit entry (§8).
- The stored verdict doesn't change: it is what the system decided. The
  review is recorded beside it:

```json
"review": { "outcome": "compliant", "note": "…", "by": "supervisor.b@example.com", "at": "2026-12-03T08:02:51Z" }
```

- The answer is `200` with `{ "check": … }`: the whole check as it is now
  (§2), like `{ "worker": … }` and `{ "site": … }` in `admin-api.md`. A note
  is trimmed, and one that is only spaces is stored as `null`.

### 4.4 `DELETE /workers/{worker_id}/photos`

Deletes every stored photo of that worker. It doesn't delete the checks:
their verdicts and reviews stay, with `photo.state: "deleted"`.

```json
{ "deleted": 37 }
```

- It deletes the files, not only the records pointing at them, and it can't
  be undone. The dashboard says so and asks for confirmation first.
- A check is marked `deleted` only once its file is gone. If a file can't be
  deleted (a disk error, say), the request stops there and fails with a
  server error. The photos deleted before it stay deleted and are counted in
  that request's audit entry. Asking again deletes the rest.
- Admins only (§10, 5). A worker asks their supervisor, who passes the
  request to an admin.
- Withdrawing training consent (§5) doesn't delete photos at once: it puts
  them under the shorter review period, and the next daily run deletes the
  ones older than that.
- The same is available on the server as
  `cli.py photos delete --worker W-0042`, for a request that arrives by
  other means. It says how many photos it will delete and asks first
  (`--yes` skips the question), and its audit entry has `actor: "cli"`.

## 5. Consent (mobile app)

AGENTS.md §15 asks for a consent notice. This contract separates two uses of
the photo, because they need different things from the worker (§10, 1):

1. **Checking and reviewing.** Every check needs the photo. When the
   detector can't settle a check, the photo is also sent to Google (Gemini)
   for a second opinion, as it is today; the explanation on a `FAIL` sends
   Gemini only text. Supervisors may look at the photo for the time in §6.
   This is how the system works, so the app **tells** the worker; there is
   nothing to agree to.
2. **Training future models.** Keeping a worker's photos longer, to train on
   them, is not needed for their check, so it is **their choice**, off until
   they say yes, and they can change their mind in Settings.

The notice's wording goes past whoever handles privacy for the company before
it ships in an app release (§10, 1). Nothing in this contract is legal
advice.

### `GET /auth/worker/consent`, `PUT /auth/worker/consent`

```json
{
  "notice": {
    "version": 2,
    "text": "Photos from your PPE checks are kept on the company's server. …",
    "training_question": "Can we also keep your photos for up to 730 days to improve the PPE detector?"
  },
  "seen_version": 1,
  "training": false,
  "updated_at": "2026-12-01T06:10:00Z"
}
```

- **The server writes the notice**, with the current retention periods (§6)
  filled in. The app shows `text` and `training_question` as they come, so
  an admin changing a period, or a change of wording, needs no app release.
  Every change to the text, including a changed period, makes a new
  `version`.
- The app shows the notice after sign-in when `seen_version` is `null` or
  older than `notice.version`, and sends
  `PUT { "version": 2, "training": true|false }` when the worker closes it.
  A `PUT` for an older version gets `409 notice_changed` with the current
  `notice`, so the worker always answers the wording they were shown.
- `PUT` answers `200` with the same body as `GET`. Both need a worker token
  whose worker has chosen their own PIN (`403 pin_change_required`
  otherwise, as for checks). A body without `version` or `training`, or with
  the wrong types (`"training": "yes"`), gets FastAPI's own `422`.
- Every version's words are kept, so the `version` in a `worker.consent`
  audit entry still says what that worker read. Answering the same version
  with the same choice again changes nothing and writes no entry.
- First wording, pending the privacy check (the periods come from §6):

  > Photos from your PPE checks are kept on the company's server for up to
  > {photo_review_days} days, so supervisors at your site can review checks.
  > When the app can't tell from the photo alone, it is also sent to Google's
  > Gemini service to check it. Ask your supervisor if you want your photos
  > deleted.
  >
  > Can we also keep your photos for up to {photo_training_days} days to
  > improve the PPE detector? You can change this in Settings at any time.
- A check is never refused for consent. Until the worker answers,
  `training` is false.
- **A check's photo is kept only once the worker has answered the notice**,
  any version of it (`seen_version` isn't `null`). Until then the check is
  stored without its photo (`photo.state: "not_stored"`, §2): the worker
  hasn't been told yet that photos are kept. A later version of the notice
  doesn't stop photos being kept while the app shows it again.
- The server stamps each check with the worker's `training` choice at that
  moment (§2).
- **Withdrawing** (`training: false` after a yes, §10, 2) sets
  `training_consent` to `false` on all of that worker's earlier checks, in
  the same transaction as the `worker.consent` audit entry. Their photos then
  fall under `photo_review_days`: the next daily run deletes the ones older
  than that, and none of them is exported for training again.
- With `PPB_WORKER_AUTH=optional`, a check sent without a login has no worker
  to ask and never showed the notice. It is stored, for the `signed_in`
  count (§4.1), but its photo never is (`photo.state: "not_stored"`, §10, 8).
- The app's Settings screen has the training choice, and says "Ask your
  supervisor" for deleting photos (§4.4).

## 6. How long photos are kept

Two periods, both admin settings (`admin-api.md` §6), with these defaults
(§10, 2):

| Setting | Applies to | Default |
|---|---|---|
| `photo_review_days` | Photos without training consent | 90 days |
| `photo_training_days` | Photos with training consent | 730 days |

`photo_training_days` is never shorter than `photo_review_days`: saying yes
can't mean a photo is kept for less time (§7).

A daily job on the server deletes photos past their period (`photo.state:
"deleted"`), and so does `cli.py photos purge`. Check records are kept, so
verdict history doesn't disappear with the photos.

- A photo is past its period when its check is older than
  `photo_review_days`, or `photo_training_days` if the check carries training
  consent (§5).
- The API server runs the job itself. Every hour it looks whether a day has
  passed since the last `photos.purge` entry, whoever wrote it, so restarts
  don't reset the clock. A Postgres lock keeps two servers from running it
  at once, and a run that fails is tried again at the next look rather than
  a day later.
- `cli.py photos purge` runs it now; `--dry-run` only says how many are due.
- A photo is deleted file first, as in §4.4: if a file can't be deleted, the
  run stops, the ones before it stay deleted and counted, and the next run
  deletes the rest.

Disk space, to size the periods: at about 300 KB a photo, 500 workers making
two checks a day fill roughly 110 GB a year. A full disk is a storing
failure like any other: the worker still gets their verdict (§2).

## 7. Settings added

| Name | Type | Default | Notes |
|---|---|---|---|
| `store_photos` | bool | `true` | Off: checks are stored without photos (`photo.state: "not_stored"`). Turning it off doesn't delete anything already stored |
| `photo_review_days` | int, 7–3650 | `90` | §6 |
| `photo_training_days` | int, 30–3650, and at least `photo_review_days` | `730` | §6 |

- A `PATCH /settings` that would leave `photo_training_days` shorter than
  `photo_review_days` (changing either one, or both) gets
  `422 out_of_range` with `field` naming the setting it sent, and nothing
  is saved, as for any patch (`admin-api.md` §6).
- Changing a retention period gets a warning, `retention_changed`, saying
  how many photos the next daily run will delete. One per period changed,
  with the number also in `count`:
  `{ "code": "retention_changed", "setting": "photo_review_days", "count": 3, "message": "…" }`.
- Both periods are whole days; `90.5` gets `422 invalid`.
- Their `.env` defaults are `PPB_PHOTO_REVIEW_DAYS` and
  `PPB_PHOTO_TRAINING_DAYS`. The server refuses to start if the training one
  is the shorter.

## 8. Audit log

New actions, written in the same transaction as the change:

| `action` | `target` | Keys in `before` / `after` |
|---|---|---|
| `check.review` | `check:<id>` | `outcome`, `note`. `before` is `null` for a first review; an admin's replacement has the review it replaced in `before` |
| `worker.photos_delete` | `worker:<id>` | after: `deleted` (a count) |
| `worker.consent` | `worker:<id>` | `version`, `training` (written by the worker's own `PUT`; `actor` is `worker:<id>`) |
| `photos.purge` | `photos` | after: `deleted` (a count) |
| `photos.export` | `photos` | after: `exported` (a count), `folder` |

- The daily retention purge writes one entry per run (`actor: "cli"`,
  `action: "photos.purge"`, after: `deleted`), not one per photo, even when
  nothing was due: the entries are also the job's clock (§6).
- `cli.py photos export --consented --to <folder>` writes one
  `photos.export` entry (`actor: "cli"`), also when it stops part way, so
  every photo that left the store is on record.
- Opening a photo isn't recorded (§10, 4). If the privacy check asks for it,
  it comes back as a change to this contract.

## 9. What each side builds

### Backend (Harshit)

- Migration `0004`: a `checks` table (§2, with `site`, `job_name`,
  `job_version`, `required`, `pending`, `confidence_threshold`,
  `gemini_verify` and `gemini_min_confidence` copied at check time),
  `consent` columns on `workers`, and the settings in §7, with
  `photo_training_days` never shorter than `photo_review_days`.
- Removing only the GPS tags from a photo's EXIF before it is stored,
  keeping orientation (§2).
- The notice text (§5), with the current periods filled in and a new version
  whenever the text changes.
- A photo store behind one interface: a folder on the server first
  (`PPB_PHOTO_DIR`, outside anything that is served), so GCP Storage can
  replace it later (§10, 7).
- Storing in `POST /compliance/check` after the verdict, never failing the
  check (§2), and returning `check_id`. Detection, thresholds and Gemini are
  unchanged.
- The endpoints in §4 and §5, with site scoping and the audit entries in §8.
  A review on any verdict; `"replace": true` for admins only.
- Withdrawing training consent clears `training_consent` on the worker's
  earlier checks in the same transaction (§5).
- The retention job and `cli.py photos delete | purge | export --consented`.
  The export writes `images/<check id>.<ext>` and `manifest.jsonl`, one
  line per photo with the job, its requirements, the verdict, the review's
  outcome and the detector's boxes. It has no worker ID, name or review
  note, and no photo past its period. Migration `0005` keeps each version
  of the notice (§5).
- Tests: a supervisor never sees another site's checks or photos, a deleted
  photo is gone from disk, a failed store still returns the verdict, a check
  without a login never keeps its photo, consent is stamped per check, and a
  withdrawal stops earlier photos being exported.

### Mobile app (Priyanka)

- The notice from §5, shown after sign-in in the server's words, with the
  training choice off until the worker says yes, and showing the new wording
  on `409 notice_changed`.
- In Settings: the training choice, and "Ask your supervisor" for deleting
  photos (§10, 5).
- Built against a stand-in until the backend's endpoints land, as in Phase 4.
  The notice doesn't go out in a release until its wording has passed the
  privacy check (§10, 1).
- Nothing else changes: the check, the result screen and the on-phone
  history stay as they are.

### Dashboard (Priyanka)

- **Checks**: the list with the filters in §4.1. For a supervisor it opens on
  "Needs review, not reviewed", and the navigation shows how many are waiting.
- **A check**: the photo with the detector's boxes drawn on it, what the job
  required then, what was missing or unverifiable, Gemini's findings, the
  review form (§4.3, on any verdict, with "Replace review" for an admin), and
  the three settings it was decided with (`confidence_threshold`,
  `gemini_verify`, `gemini_min_confidence`).
- Built against the in-browser stand-in until the backend's endpoints land,
  as in Phase 4.
- **Workers**: a link to each worker's checks, and "Delete photos…" for an
  admin (§4.4), with a confirmation that says how many photos and that it
  can't be undone.
- **Settings**: the three settings in §7, generated as today, and next to
  `worker_auth` the number of checks without a login in the last 3 working
  days (§4.1).
- **Audit log**: the new actions in §8.

## 10. Decisions on the review questions

Answered in Harshit's review of draft 2 (PR #34), and adopted above. Items
10–12 came from his own draft (#33).

1. **Consent: the split in §5.** Workers are told photos are kept for
   review, with nothing to agree to. Keeping them longer, for training, is
   opt-in and off until they say yes. Before the notice ships in the app,
   its wording goes past whoever handles privacy for the company.
2. **Retention: 90 days, and 730 with training consent** (§6, §7). When a
   worker withdraws training consent, their earlier photos fall back to the
   90-day rule: the next daily run deletes the ones older than that, and none
   of them is exported for training again. Withdrawing sets
   `training_consent` to `false` on their earlier checks, in the same
   transaction as the `worker.consent` entry (§5).
3. **Changing a review: yes, for an admin,** with `"replace": true` (§4.3).
   Each change is its own `check.review` audit entry, with `before` and
   `after` (§8). A supervisor still gets `409 already_reviewed` on a check
   someone has reviewed.
4. **Who looked at a photo: no audit entry per view.** Restricted access
   (admins, and the supervisor of the check's site) is the control (§1). If
   the privacy check in 1 asks for view logging, it comes back then.
5. **Deleting photos: admins only** (§4.4). A worker asks their supervisor,
   who passes the request to an admin. The app's Settings says "Ask your
   supervisor". The CLI covers a request that arrives some other way.
6. **Location: remove only the GPS tags,** keeping orientation (§2).
7. **Storage: a folder on the server first,** behind one interface so GCP
   Storage can replace it later (§9).
8. **Checks without a login: store the check, never its photo**
   (`photo.state: "not_stored"`, §2, §5). The check is needed for the
   `signed_in` count (§4.1), but an app without login never showed the
   notice.
9. **A review's outcome in the app: not in this phase** (§1).
10. **Which checks can be reviewed: any verdict** (§4.3). A supervisor can
    overturn a wrong `FAIL` (a compliant worker accused) or flag a wrong
    `PASS`, the worst failure for a safety check. So `409 not_reviewable` is
    gone, `reviewed` applies to every verdict (§4.1), and a supervisor's
    default view stays `verdict=NEEDS_REVIEW&reviewed=false`.
11. **One decision per check** for now (§4.3). Per-item decisions can come
    later, if training needs them as labels.
12. **Erasing a worker completely: not now.** Deleting their photos covers
    the request expected. Anonymising their check records can come later, as
    a CLI command, if a request needs it.
