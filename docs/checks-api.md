# Phase 5 — Checks and review contract

**Status: draft for review by Harshit and Priyanka. Nothing here is built yet.**

Today a check's result exists only in the response to the phone, and in the
phone's own history. This phase stores every check on the server, with the
photo the detector looked at, so that:

- supervisors can review the checks the system couldn't settle
  (`NEEDS_REVIEW`), and see their site's record;
- admins can see when no phone still sends checks without a login, the
  condition for switching to `PPB_WORKER_AUTH=required` (`auth-api.md` §5);
- photos are kept for training the next model (AGENTS.md §15).

Once agreed, the backend, the dashboard and the app are built against it in
parallel. Any change after that goes through a pull request that edits this
file first.

It builds on [`docs/auth-api.md`](auth-api.md) (sessions, error shape) and
[`docs/admin-api.md`](admin-api.md) (§1 roles and sites, §2 errors, warnings
and lists, §6 settings, §9 audit log). The new dashboard endpoints are under
`/api/v1/admin`. Requests and responses are JSON, and timestamps are UTC with
a `Z` suffix.

---

## 1. Who may do what

| Area | admin | supervisor | worker (app) |
|---|:-:|:-:|:-:|
| See checks and their photos | all sites | own site | |
| Review a check | all sites | own site | |
| Delete a worker's photos | ✓ | | |
| Change the photo settings | ✓ | | |
| Agree or decline to have their photos kept | | | own |

- A check belongs to the site its worker was at when the check was made. It
  stays there if the worker moves later, so a site's record doesn't change
  under its supervisor.
- A check sent without a login, under a worker ID that doesn't exist, has no
  site. Only admins see it.
- A supervisor asking for a check at another site gets `404 not_found`, as
  for workers (`admin-api.md` §1).
- Workers don't get their history from the server in this phase. The app
  keeps its own history on the phone, as now.

## 2. What is stored

Every check that reaches a verdict is stored: `PASS`, `FAIL` and
`NEEDS_REVIEW`. A request refused before a verdict (`401`, `400`,
`422 unknown_job`, `503`) is not. Check records are never deleted in this
phase; photos expire (§3).

### A check, in a list

```json
{
  "id": 1042,
  "at": "2026-11-24T06:58:12Z",
  "worker_id": "W-0042",
  "worker_name": "Asha Verma",
  "signed_in": true,
  "site": "pune_tower_b",
  "job": "welder",
  "job_name": "Welder",
  "verdict": "NEEDS_REVIEW",
  "verdict_source": "detector+gemini",
  "missing_equipment": ["gloves"],
  "has_photo": true,
  "review": null
}
```

- `signed_in` is `false` for a check sent without a worker token, which
  `PPB_WORKER_AUTH=optional` still allows. Its `worker_id` is then whatever
  the app sent, unverified, and `worker_name` and `site` are `null` when no
  worker has that ID.
- `job_name` is the job's name when the check was made.
- `review` is `null` until someone reviews the check (§4).

### A check, in full

`GET /checks/{id}` returns the fields above, everything the app was told
(`ComplianceResponse`), and what the job required at the time:

```json
{
  "…": "the list fields",
  "required": ["hardhat", "safety_vest", "goggles", "gloves", "boots"],
  "pending": ["face_shield"],
  "confidence_avg": 0.84,
  "detections": [
    { "class": "hardhat", "confidence": 0.93, "bbox": [376.1, 110.5, 518.0, 223.5] }
  ],
  "violations_detected": [],
  "unverifiable_equipment": ["face_shield"],
  "verification": {
    "model": "gemini-2.5-flash",
    "items": [
      { "item": "gloves", "status": "uncertain", "confidence": 0.55, "reason": "Both hands are out of frame." }
    ],
    "elapsed_ms": 2840
  },
  "explanation": null,
  "model_used": "yolo11l",
  "inference_time_ms": 52.3,
  "photo": { "width": 1280, "height": 960, "deleted_at": null }
}
```

- `required` and `pending` are the job's requirements when the check was
  made, with the same meaning as in `GET /jobs`. The matrix can change
  afterwards, and a review has to be judged by the rules the verdict used.
- `bbox` is in the stored photo's pixels (§3), so the dashboard can draw the
  boxes on it.
- `photo` is `null` when no photo was kept. `deleted_at` is set once the
  photo has been deleted, by expiry or on request; the check stays.

## 3. The photo

### What is kept

The image the detector looked at, saved as a JPEG: not the bytes the phone
sent. The boxes in `detections` then line up with it, and the phone's
metadata (EXIF, which can hold the GPS location and the phone's model) is not
kept.

### When it is kept

Only when **all three** are true:

1. the check was signed in (a worker token);
2. the worker has agreed to the current photo notice (§5);
3. the `store_photos` setting is on (§6).

Otherwise the check is stored without a photo and works exactly as before.
A check from an app without login never has its photo kept: that app never
showed the notice.

### Where

On the server's disk, under `PPB_PHOTO_DIR` (an `.env` setting, default
`data/photos`, ignored by git). The folder is outside anything served by URL,
and file names are random, never the worker's ID. The backend keeps storage
behind one small interface, so cloud storage (AGENTS.md §6) can replace the
disk later without changing this contract.

### For how long

A photo is deleted once it is older than `photo_retention_days` (§6). The
server checks once an hour, and the check record stays with
`photo.deleted_at` set.

### Who sees it

`GET /checks/{id}/photo` (§4): an admin, or the supervisor of the check's
site. Photos never appear in list responses or in the audit log.

## 4. Endpoints for the dashboard

### `GET /checks?site=&worker_id=&job=&verdict=&review=&signed_in=&from=&to=&limit=&cursor=`

Newest first, paged (`admin-api.md` §2), with `total`.

- `site` filters to one site. Admins only; a supervisor always gets their
  own site.
- `worker_id` matches without regard to case.
- `job` is a job key; `verdict` is `PASS`, `FAIL` or `NEEDS_REVIEW`.
- `review` is `awaiting` (can be reviewed and hasn't been), `done`, or left
  out for both.
- `signed_in=false` lists checks sent without a login. With `from`, its
  `total` is the count the switch to `required` waits for
  (`auth-api.md` §5).
- `from` and `to` are timestamps (inclusive, exclusive), as in the audit
  log.

### `GET /checks/{id}`

The check in full (§2). `404 not_found` for no such check, or one at another
site for a supervisor.

### `GET /checks/{id}/photo`

The photo, as `image/jpeg`, with `Cache-Control: no-store` so it isn't left
in the browser's cache. `404 not_found` when the check has no photo (never
kept, or deleted). The session cookie's `Path=/api` covers this address, so
an `<img>` on a dashboard page can load it directly.

### `POST /checks/{id}/review`

```json
{ "decision": "compliant", "note": "Gloves on: visible in the supervisor's own photo." }
```

`200 OK` with `{ "check": { …in full… } }`, whose `review` is now:

```json
{ "decision": "compliant", "note": "…", "by": "supervisor@example.com", "at": "2026-11-24T07:15:00Z" }
```

- `decision` is `compliant` or `not_compliant`. `note` is optional, up to
  500 characters.
- A `NEEDS_REVIEW` check can be reviewed (question 2 in §9 asks about the
  others).
- The verdict never changes. The review sits beside it, and whoever reports
  on checks reads the review where there is one.
- To change an existing review, send `"replace": true`. Without it, a check
  someone has already reviewed answers `409 already_reviewed`, with the check
  as it is now in `current`, like `version_conflict` (`admin-api.md` §2). Two
  supervisors can't silently overwrite each other.
- Errors: `404 not_found`, `409 already_reviewed`, `409 not_reviewable` (a
  verdict that can't be reviewed), `422 invalid` (an unknown `decision`, or a
  note that is too long).

### `DELETE /workers/{worker_id}/photos` — admin

Deletes every kept photo of the worker, at once, when they ask. `200 OK`
with `{ "deleted": 37 }`.

- The check records stay: they are the site's safety record. Each one's
  `photo.deleted_at` is set.
- The worker's agreement to the notice (§5) is withdrawn too, so no new photo
  is kept until they agree again.
- Errors: `404 not_found`.

## 5. The photo notice, in the app

AGENTS.md §15 asks for a consent notice before photos are kept.

- After signing in, before the first check, the app shows the notice, once
  per version of it. The worker chooses **Agree** or **Not now**. Either way
  they can check; with **Not now** their photos aren't kept.
- The app's settings screen lets the worker change their answer later.
- The server writes the notice, so its wording and the number of days stay
  right without an app release. Proposed first version:

  > Photos from your PPE checks are kept on the company's server for up to
  > {days} days. Supervisors at your site may look at them to review checks,
  > and they may be used to improve the PPE detector. Ask your supervisor if
  > you want them deleted.

### `GET /auth/worker/me` gains two fields

```json
{
  "worker": { "…": "…" },
  "must_change_pin": false,
  "expires_at": "…",
  "photo_notice": { "version": 1, "text": "Photos from your PPE checks are kept…" },
  "photo_consent": { "version": 1, "agreed": true, "at": "2026-11-24T06:50:00Z" }
}
```

- `photo_notice` is `null` while `store_photos` is off: there is nothing to
  agree to.
- `photo_consent` is `null` until the worker answers. The app shows the
  notice when `photo_notice` isn't `null` and `photo_consent` is `null` or
  has an older `version`.

### `POST /auth/worker/photo-consent`

```json
{ "version": 1, "agreed": true }
```

`200 OK` with `{ "photo_consent": { … } }`. Answering an old `version`
gets `409 notice_changed` with the current `photo_notice`, so the app shows
the new wording before asking again.

### `POST /compliance/check` gains one field

`check_id`: the stored check's ID, or `null` if it couldn't be stored. The
current app ignores it; a later one can show it ("Check #1042") so a worker
can quote it to their supervisor.

Storing never decides the verdict. If the database or the disk fails while
storing, the worker still gets their verdict, `check_id` is `null`, and the
server logs an error (question 7 in §9).

## 6. Settings (`admin-api.md` §6)

Two new runtime settings, both admin only and both with `confirm: true`:

| `name` | `type` | Range | Default (`.env`) | What it does |
|---|---|---|---|---|
| `store_photos` | boolean | | `true` (`PPB_STORE_PHOTOS`) | Keep photos of checks, for workers who agreed. Off: checks are still stored, without photos; photos already kept stay until they expire |
| `photo_retention_days` | number | 7–3650, step 1 | `365` (`PPB_PHOTO_RETENTION_DAYS`) | How long a photo is kept |

Shortening `photo_retention_days` deletes older photos within the hour, so
`PATCH /settings` warns first (`admin-api.md` §2):

| `code` | Extra key | Returned by |
|---|---|---|
| `photos_will_be_deleted` | `setting` | Settings: a shorter `photo_retention_days` that will delete photos already kept, with how many in `message` |

## 7. Audit log (`admin-api.md` §9)

New entries, written in the same transaction as the change:

| `action` | `target` | Keys in `before` / `after` |
|---|---|---|
| `check.review` | `check:<id>` | `decision`, `note` |
| `worker.photos_delete` | `worker:<id>` | after: `count` |
| `worker.photo_consent` | `worker:<id>` | `version`, `agreed` |
| `photos.expire` | `photos` | after: `count`, `older_than_days` (one entry per hourly run that deleted anything) |

`actor` gains two values: `worker:<id>` for a worker's own answer to the
notice, and `system` for the hourly expiry. `site` is the check's or the
worker's site. Looking at a photo isn't recorded (question 5 in §9).

## 8. What each side builds

### Backend (Harshit)

- Migration `0004`:
  - a `checks` table: the fields of §2, the photo's file, size and
    `deleted_at`, and the review; indexed for the filters in §4;
  - the worker's photo agreement (`version`, `agreed`, `at`) on `workers`.
- Storing in `POST /compliance/check`, with the JPEG written off the event
  loop, and `check_id` in the response.
- The endpoints in §4, scoped by site for supervisors.
- `photo_notice` and `photo_consent` in `GET /auth/worker/me`, and
  `POST /auth/worker/photo-consent`.
- The two settings, `PPB_PHOTO_DIR` in `.env.example`, the photo folder in
  `.gitignore`, and the hourly expiry.
- Tests against PostgreSQL for every rule here, and in particular:
  - a photo is never kept without a token, without agreement, or with
    `store_photos` off;
  - a supervisor never sees another site's check or photo;
  - a failure to store doesn't change the verdict the worker gets.

### Mobile app (Priyanka)

- The notice after signing in (§5), and the setting to change the answer.
- Nothing else has to change. Showing `check_id` on the result screen is
  optional.

### Dashboard (Priyanka)

| Page | Uses |
|---|---|
| Checks (admin, supervisor): filters as in §4. A supervisor starts on their site's checks awaiting review | §4 |
| A check: the photo with its boxes, the verdict, what was required, Gemini's finding for each item, and the review form | §2, §4 |
| Workers: a link to each worker's checks. Admin: "Delete this worker's photos", with confirmation | §4 |
| Settings: the two new settings appear by themselves, since the page is built from `GET /settings` | §6 |

## 9. Open questions for review

1. **How long to keep photos.** 365 days as written. Training wants photos
   kept long, and privacy wants them kept short. Should training photos be
   copied out by an admin command instead, so the server's copy can be
   short, say 90 days?
2. **Which checks can be reviewed.** `NEEDS_REVIEW` only, as written. Should
   a supervisor also be able to overturn a wrong `FAIL` (a compliant worker
   accused), or flag a wrong `PASS`?
3. **One decision, or one per item.** As written, the whole check is
   compliant or not. A decision per item (gloves: worn, not worn) would make
   better training labels, but takes longer to fill in.
4. **Declining the notice.** As written, a worker who declines can still
   check, without photos kept. Should keeping photos be a condition of using
   the app instead?
5. **Recording who looked at a photo.** Not recorded as written, since an
   entry per view would swamp the audit log. Should views be logged
   somewhere?
6. **Erasing a worker completely.** Their photos can be deleted on request
   (§4). Should there also be a way to remove their name and ID from their
   check records, leaving anonymous records?
7. **When a check can't be stored.** As written, the worker still gets their
   verdict. Should the check be refused instead, so no check goes
   unrecorded?
