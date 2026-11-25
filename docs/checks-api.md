# Phase 6 — Checks, photos and review: API contract

**Status: draft 2, for review. Nothing here is built. The open questions in
§10 need answers before anything is.** Draft 2 adds what Harshit's review
brought over from his draft (#33): the `signed_in` filter, the job's
requirements as they were, the notice text from the server, and keeping a
photo's orientation when its location is removed. It also corrects when a
photo goes to Gemini (§5).

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

Numbering: `admin-api.md` §10 calls this "Phase 5"; the roadmap numbers it
P6. This contract follows the roadmap, and the same pull request points §10
here.

---

## 1. Who may do what

| Area | admin | supervisor | worker (app) |
|---|:-:|:-:|:-:|
| See checks and their photos | all sites | own site | |
| Review a `NEEDS_REVIEW` check | all sites | own site | |
| Delete a worker's photos | ✓ | see §10, Q5 | |
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
  stays on the phone (INSTRUCTIONS.md); §10, Q9 asks whether a worker should
  see a review's outcome.

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
  "model_used": "yolo11l_ppe_v1",
  "confidence_threshold": 0.45,
  "gemini_verify": true,
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
- `confidence_threshold` and `gemini_verify` are the two settings that
  decide a verdict, as they were at the time. An admin can change both from
  the dashboard, so a reviewer looking at a borderline check later needs them.
- `photo.state` is `kept`, `deleted` (on request or by the retention rule,
  §6) or `not_stored` (storing failed, or photos are switched off, §7).
- `training_consent` is the worker's choice **when the check was made** (§5).
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
is stored (§10, Q6); every other tag stays, the **orientation** tag above
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
| 404 | `not_found` | No such check (or one at another site, for a supervisor) |
| 409 | `not_reviewable` | The check's verdict isn't `NEEDS_REVIEW` |
| 409 | `already_reviewed` | Someone has reviewed it already. The body carries the check as it is now in `current`, as with `version_conflict` |
| 409 | `notice_changed` | The app answered an older version of the consent notice. The body carries the current `notice` (§5) |
| 410 | `photo_deleted` | The check exists, but its photo has been deleted |
| 422 | `invalid` | A review without an outcome, a note over 500 characters, or a bad filter |

## 4. Dashboard endpoints

### 4.1 `GET /checks?site=&worker=&job=&verdict=&reviewed=&signed_in=&from=&to=&limit=&cursor=`

Newest first, paged as in `admin-api.md` §2, with `total`.

- `verdict` is `PASS`, `FAIL` or `NEEDS_REVIEW`. `reviewed` is `true` or
  `false`, and only means something with `NEEDS_REVIEW`.
- `signed_in=false` lists checks sent without a login. With `from`, its
  `total` is the count the switch to `PPB_WORKER_AUTH=required` waits for:
  none for 3 working days (`auth-api.md` §5). The dashboard can show it on
  the Settings page next to `worker_auth`. If §10, Q8 decides not to store
  such checks, this filter goes too.
- `from` and `to` are timestamps (inclusive, exclusive), as in the audit log.
- A supervisor's results are their own site's, whatever `site` says.
- Each item is the check without `detections` and `verification`, to keep
  the list small. The detail has everything.

The dashboard's default view for a supervisor is
`verdict=NEEDS_REVIEW&reviewed=false`: the checks waiting for a person.

### 4.2 `GET /checks/{id}` and `GET /checks/{id}/photo`

`GET /checks/{id}` returns the whole check (§2).

`GET /checks/{id}/photo` returns the image itself, with its own content type
and:

- `Cache-Control: private, no-store`: a shared computer's cache shouldn't
  keep workers' photos.
- `Content-Disposition: inline`, and no file name containing the worker ID.
- `410 photo_deleted` once the photo is gone.

It's an ordinary `/api` URL, so the session cookie (`Path=/api`) goes with it
and the dashboard can use it in an `<img>`. The page's `img-src 'self'`
policy already allows it. Who opened which photo is recorded; see §10, Q4.

### 4.3 `POST /checks/{id}/review`

```json
{ "outcome": "compliant", "note": "Gloves are in his left hand, worn once welding starts." }
```

- `outcome` is `compliant` or `not_compliant`. `note` is optional, up to 500
  characters.
- Only for a `NEEDS_REVIEW` check (`409 not_reviewable` otherwise), and only
  once (`409 already_reviewed`, with `current`). Changing a review is §10, Q3.
- The stored verdict doesn't change: it is what the system decided. The
  review is recorded beside it:

```json
"review": { "outcome": "compliant", "note": "…", "by": "supervisor.b@example.com", "at": "2026-12-03T08:02:51Z" }
```

### 4.4 `DELETE /workers/{worker_id}/photos`

Deletes every stored photo of that worker. It doesn't delete the checks:
their verdicts and reviews stay, with `photo.state: "deleted"`.

```json
{ "deleted": 37 }
```

- It deletes the files, not only the records pointing at them, and it can't
  be undone. The dashboard says so and asks for confirmation first.
- Withdrawing training consent (§5) doesn't delete photos by itself; see
  §10, Q2.
- The same is available on the server as
  `cli.py photos delete --worker W-0042`, for a request that arrives by
  other means.

## 5. Consent (mobile app)

AGENTS.md §15 asks for a consent notice. This draft separates two uses of the
photo, because they need different things from the worker:

1. **Checking and reviewing.** Every check needs the photo. When the
   detector can't settle a check, the photo is also sent to Google (Gemini)
   for a second opinion, as it is today; the explanation on a `FAIL` sends
   Gemini only text. Supervisors may look at the photo for the time in §6.
   This is how the system works, so the app **tells** the worker; there is
   nothing to agree to.
2. **Training future models.** Keeping a worker's photos longer, to train on
   them, is not needed for their check, so it is **their choice**, off until
   they say yes, and they can change their mind in Settings.

Whether this split is right, or a notice alone is enough, is §10, Q1. It is
also a question for whoever handles privacy and legal matters for the
company. Nothing in this contract is legal advice.

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
- Proposed first wording (the periods come from §6):

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
- The server stamps each check with the worker's `training` choice at that
  moment (§2), so a later "no" doesn't make earlier "yes" photos usable
  without deciding §10, Q2.
- With `PPB_WORKER_AUTH=optional`, a check sent without a login has no worker
  to ask. It is stored with `training_consent: false`.

## 6. How long photos are kept

Two periods, both admin settings (`admin-api.md` §6), with defaults to agree
in §10, Q2:

| Setting | Applies to | Suggested default |
|---|---|---|
| `photo_review_days` | Photos without training consent | 90 days |
| `photo_training_days` | Photos with training consent | 2 years |

A daily job on the server deletes photos past their period (`photo.state:
"deleted"`), and so does `cli.py photos purge`. Check records are kept, so
verdict history doesn't disappear with the photos.

Disk space, to size the periods: at about 300 KB a photo, 500 workers making
two checks a day fill roughly 110 GB a year. A full disk is a storing
failure like any other: the worker still gets their verdict (§2).

## 7. Settings added

| Name | Type | Default | Notes |
|---|---|---|---|
| `store_photos` | bool | `true` | Off: checks are stored without photos (`photo.state: "not_stored"`). Turning it off doesn't delete anything already stored |
| `photo_review_days` | int, 7–3650 | §10, Q2 | §6 |
| `photo_training_days` | int, 30–3650 | §10, Q2 | §6 |

Changing a retention period gets a warning, `retention_changed`, saying how
many photos the next daily run will delete.

## 8. Audit log

New actions, written in the same transaction as the change:

| `action` | `target` | Keys in `before` / `after` |
|---|---|---|
| `check.review` | `check:<id>` | after: `outcome`, `note` |
| `worker.photos_delete` | `worker:<id>` | after: `deleted` (a count) |
| `worker.consent` | `worker:<id>` | `version`, `training` (written by the worker's own `PUT`; `actor` is `worker:<id>`) |
| `check.photo_view` | `check:<id>` | none (only if §10, Q4 says so) |

The daily retention purge writes one entry per run (`actor: "cli"`,
`action: "photos.purge"`, after: `deleted`), not one per photo.

## 9. What each side builds

### Backend (Harshit)

- Migration `0004`: a `checks` table (§2, with `site`, `job_name`,
  `job_version`, `required`, `pending`, `confidence_threshold` and
  `gemini_verify` copied at check time), `consent` columns on `workers`, and
  the settings in §7.
- Removing only the GPS tags from a photo's EXIF before it is stored,
  keeping orientation (§2).
- The notice text (§5), with the current periods filled in and a new version
  whenever the text changes.
- A photo store behind one interface: a folder on the server first
  (`PPB_PHOTO_DIR`, outside anything that is served), cloud storage later
  (§10, Q7).
- Storing in `POST /compliance/check` after the verdict, never failing the
  check (§2), and returning `check_id`. Detection, thresholds and Gemini are
  unchanged.
- The endpoints in §4 and §5, with site scoping and the audit entries in §8.
- The retention job and `cli.py photos delete | purge | export --consented`.
- Tests: a supervisor never sees another site's checks or photos, a deleted
  photo is gone from disk, a failed store still returns the verdict, and
  consent is stamped per check.

### Mobile app (Priyanka)

- The notice from §5, shown after sign-in in the server's words, with the
  training choice off until the worker says yes, and showing the new wording
  on `409 notice_changed`.
- In Settings: the training choice, and how to ask for their photos to be
  deleted (§10, Q5 decides who they ask).
- Nothing else changes: the check, the result screen and the on-phone
  history stay as they are.

### Dashboard (Priyanka)

- **Checks**: the list with the filters in §4.1. For a supervisor it opens on
  "Needs review, not reviewed", and the navigation shows how many are waiting.
- **A check**: the photo with the detector's boxes drawn on it, what the job
  required then, what was missing or unverifiable, Gemini's findings, the
  review form (§4.3), and the threshold and Gemini setting it was decided
  with.
- **Workers**: a link to each worker's checks, and "Delete photos…" for an
  admin (§4.4), with a confirmation that says how many photos and that it
  can't be undone.
- **Settings**: the three settings in §7, generated as today, and next to
  `worker_auth` the number of checks without a login in the last 3 working
  days (§4.1).
- **Audit log**: the new actions in §8.

## 10. Open questions

1. **Consent.** Is the split in §5 right: a notice for checking and review,
   and an opt-in for training? Or is a notice enough for both? Or is an
   opt-in needed for any photo to be kept at all? Please also check with
   whoever handles privacy for the company before this is built.
2. **Retention.** How long are photos kept, with and without training
   consent (§6)? When a worker withdraws training consent, are their earlier
   training photos deleted, or kept under the shorter review period?
3. **Changing a review.** Once is the rule in §4.3. Should an admin be able
   to change a supervisor's review, recorded as a second audit entry?
4. **Recording who looked at a photo.** Every `GET /checks/{id}/photo` as a
   `check.photo_view` audit entry, or none? It's personal data, but it would
   make the audit log much longer.
5. **Who can delete a worker's photos.** Admins only, as drafted, or a
   supervisor for their own site too? Where does a worker send the request:
   to their supervisor, or somewhere the app shows?
6. **Location data.** Remove only the GPS tags from photos before storing
   them, as drafted (§2)? Location is personal data that a PPE check doesn't
   need. Removing just those tags leaves the image and its orientation as
   they were, so the boxes still line up; removing all of the EXIF wouldn't.
7. **Where photos live.** A folder on the server first, as drafted, or cloud
   storage from the start (AGENTS.md §6 mentions GCP Storage)? With cloud
   storage the photo endpoint would still be served through the API, so
   access stays checked per request.
8. **Checks sent without a login.** While `PPB_WORKER_AUTH=optional`, the
   `worker_id` of such a check is only what was typed. Store them as drafted
   (with `signed_in: false`), or don't store them, or don't keep their
   photos?
9. **Should a worker see a review's outcome?** For example, "Reviewed:
   compliant" in the app's history. It would need a worker-facing endpoint,
   which this draft leaves out.

From Harshit's draft (#33), still open:

10. **Which checks can be reviewed.** `NEEDS_REVIEW` only, as drafted.
    Should a supervisor also be able to overturn a wrong `FAIL` (a compliant
    worker accused), or flag a wrong `PASS`? If so, the same review action,
    allowed on those verdicts too.
11. **One decision, or one per item.** As drafted, the whole check is
    compliant or not. A decision per item (gloves: worn, not worn) would make
    better training labels, but takes longer to fill in.
12. **Erasing a worker completely.** Their photos can be deleted on request
    (§4.4). Should there also be a way to remove their name and ID from their
    check records, leaving anonymous records?
