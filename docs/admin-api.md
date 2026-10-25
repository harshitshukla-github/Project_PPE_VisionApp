# Phase 3 — Admin API contract

**Status: agreed (draft 2, plus the fixes in PR #18). The backend (§3–§9)
is built; the dashboard pages (§11) are in progress.**

This is what the admin dashboard (Phase 4) calls to run the system: sites,
the job PPE matrix, runtime settings, dashboard users, workers, and the
audit log. Once agreed, the backend and the dashboard are built against it
in parallel. Any change after that goes through a pull request that edits
this file first.

It builds on the login contract, [`docs/auth-api.md`](auth-api.md): every
endpoint here needs a dashboard session (the `ppb_session` cookie), every
write needs `X-CSRF-Token`, and errors use the same shape.

All paths are under `/api/v1/admin`. Requests and responses are JSON.
Timestamps are UTC with a `Z` suffix.

---

## 1. Who may do what

The system has **sites** (§3). An admin works across all of them. A
supervisor belongs to exactly one site and sees only that site's workers.

| Area | admin | supervisor |
|---|:-:|:-:|
| See sites | all | own site |
| Create, rename, deactivate sites | ✓ | |
| See the job matrix, items and jobs | ✓ | ✓ |
| Change the job matrix; add, rename, reorder or deactivate jobs | ✓ | |
| See and change runtime settings | ✓ | |
| Manage dashboard users | ✓ | |
| See, create, rename, deactivate and reactivate workers; reset PINs | all sites | own site |
| Move a worker to another site | ✓ | |
| Read the audit log | ✓ | |

- Anything else returns `403 forbidden`.
- A supervisor asking for a worker at another site gets `404 not_found`, so
  the API doesn't reveal anything about another site's workers. The one
  exception is creating a worker whose ID is already taken: `409
  already_exists` says the ID is taken, not where. Worker IDs are printed on
  badges, so this is accepted.
- A user who still has to change their password gets
  `403 password_change_required` on every endpoint here.
- The job matrix is shared by all sites in this phase. Per-site job lists can
  come later if sites need them.

## 2. Conventions

### Errors

The shape from `auth-api.md` §3: `detail` is a sentence for a person, `code`
is what the dashboard branches on. Validation errors also give the `field`
to highlight, as a path down to the exact entry:

```json
{ "detail": "\"flippers\" isn't a known item.", "code": "unknown_item", "field": "requirements[3].item" }
```

`field` is a top-level name (`display_name`, `confidence_threshold`) or a
path into a list (`requirements[3].item`, `keys[2]`).

A request whose body is malformed (wrong types, missing fields) still gets
FastAPI's own `422`, whose `detail` is a list of problems rather than a
sentence. The dashboard's error handler must accept both shapes.

Codes added by this contract:

| HTTP | `code` | When |
|---|---|---|
| 404 | `not_found` | No such site, job, user or worker (or one at another site, for a supervisor) |
| 409 | `already_exists` | The site key, job key, email or worker ID is taken |
| 409 | `version_conflict` | Someone else changed this job since you loaded it. The body carries the job as it is now in `current` |
| 409 | `last_admin` | The change would leave no active admin |
| 409 | `cannot_change_self` | You tried to deactivate yourself, or change your own role or site |
| 409 | `last_active_job` | The change would leave the app with no jobs to choose from |
| 409 | `site_in_use` | A site can't be deactivated while it has active workers or supervisors |
| 422 | `invalid_key` | A site or job key isn't a lowercase identifier (`^[a-z][a-z0-9_]*$`, 2–64 characters) |
| 422 | `unknown_item` | A requirement names an item that isn't in the catalogue |
| 422 | `no_required_item` | A job would have no required item. This applies to every job, active or not |
| 422 | `out_of_range` | A setting is outside its allowed range (§6) |
| 422 | `invalid` | Anything else malformed: an item listed twice, an incomplete job order, a bad email or name, a supervisor without a site, a deactivated site (`field: "site"`) |

For the mobile app, `POST /compliance/check` answers `422 unknown_job` when
the job key doesn't exist or the job has been deactivated (INSTRUCTIONS.md).
The app refreshes its job list and asks the worker to pick again.

### Warnings

Some valid changes deserve a second look. A successful response may carry
`warnings`, and they never block the change. Each warning names what it is
about, so the dashboard can put a badge on the right cell or field:

```json
{
  "job": { "…": "…" },
  "warnings": [
    { "code": "gemini_only_item", "item": "face_shield",
      "message": "Face shield can only be judged by Gemini, whose accuracy on it hasn't been measured." }
  ]
}
```

| `code` | Extra key | Returned by |
|---|---|---|
| `gemini_only_item` | `item` | Job create or requirements: a required item the detector can't see |
| `gemini_verify_off` | none | Job create or requirements, while `gemini_verify` is off: anything the detector misses stays "Needs review" |
| `worker_auth_required` | `setting` | Settings: checks from an app without login will now be refused |
| `gemini_verify_switched_off` | `setting` | Settings: turning `gemini_verify` off |
| `threshold_changed` | `setting` | Settings: every verdict from now on uses the new detector threshold |

`message` is a sentence for a person; branch on `code`. A code always has the
same extra key, so a typed client can rely on it.

### Lists

Small collections (sites, items, jobs, users, settings) come back whole.
Workers and the audit log are paged:

```
GET /api/v1/admin/workers?limit=50&cursor=…
```

```json
{ "items": [ … ], "next_cursor": "eyJpZCI6MTIzfQ", "total": 312 }
```

- `limit` defaults to 50, at most 200.
- `next_cursor` is `null` on the last page. Treat it as opaque.
- `total` counts everything that matches the filters, so the page can say
  "51–100 of 312".

### When changes take effect

The server that handled the change uses it on its next request. Any other
server process picks it up within `PPB_MATRIX_CACHE_S` (30 seconds); write
responses carry `"applies_within_s": 30` so the dashboard can say so.

The mobile app refetches the job list before each check, so a deactivated or
new job shows up by the next check. A check sent against a job deactivated
in the meantime gets `422 unknown_job` (above).

---

## 3. Sites — admin (supervisors see their own)

A site is a place where workers are checked: a construction site, a plant.
Every worker belongs to one site, and every supervisor belongs to one site.

### A site

```json
{ "key": "pune_tower_b", "display_name": "Pune Tower B", "is_active": true,
  "worker_count": 38, "created_at": "2026-10-19T09:00:00Z" }
```

`key` never changes after creation; rename with `display_name`.

### `GET /sites`

Every site, active or not, for an admin; only their own site for a
supervisor.

### `POST /sites` — admin

```json
{ "key": "pune_tower_b", "display_name": "Pune Tower B" }
```

`201 Created` with `{ "site": … }`. Errors: `409 already_exists`,
`422 invalid_key`.

### `PATCH /sites/{key}` — admin

```json
{ "display_name": "…", "is_active": false }
```

`200 OK` with `{ "site": … }`. Errors: `404 not_found`, `409 site_in_use`.
A site is deactivated only once its workers and supervisors have been moved
or deactivated; sites are never deleted, because checks will refer to them.

### Existing data

Migration `0003` creates a site `main` ("Main site") and puts every existing
worker and supervisor in it.

---

## 4. Items (read-only)

The equipment the system knows. Items come from the detector's classes, so
the dashboard can't add or remove them.

### `GET /items`

```json
[
  { "key": "hardhat", "display_name": "Hardhat", "detectable": true, "detector_ap50": 0.909 },
  { "key": "face_shield", "display_name": "Face shield", "detectable": false, "detector_ap50": null }
]
```

- `detectable` is whether the detector was trained on the item. A required
  item that isn't detectable is judged by Gemini alone.
- `detector_ap50` is the loaded model's accuracy on the item, from the
  metrics file saved with the model's weights at training time. It is `null`
  for items the model wasn't trained on, or when the model has no metrics
  file. It changes when the model does, so it is never stored per item.

---

## 5. Jobs and the matrix

### A job

```json
{
  "key": "welder",
  "display_name": "Welder",
  "position": 1,
  "is_active": true,
  "version": 4,
  "requirements": [
    { "item": "hardhat", "mode": "required" },
    { "item": "safety_vest", "mode": "required" },
    { "item": "goggles", "mode": "required" },
    { "item": "gloves", "mode": "required" },
    { "item": "boots", "mode": "required" },
    { "item": "face_shield", "mode": "pending" }
  ],
  "updated_at": "2026-10-16T10:12:00Z"
}
```

- `mode` is `required` (checked by the detector, then Gemini) or `pending`
  (listed for the worker as "verify manually", never scored). An item not in
  the list isn't asked for.
- The list's order is the order the app shows.
- Every job, active or not, has at least one required item.
- `key` never changes after creation: checks and the app refer to jobs by
  key. Rename with `display_name`.
- `version` goes up by one when the display name, the active flag or the
  requirements change. **Reordering doesn't change it** (`PUT /job-order`).
  Edits send the version they started from. If someone else saved in
  between, the server answers `409 version_conflict` with the current job in
  `current`, instead of overwriting their work:

```json
{ "detail": "Someone changed this job while you were editing it.", "code": "version_conflict",
  "current": { "key": "welder", "version": 5, "…": "…" } }
```

### `GET /jobs` — admin, supervisor

Every job, active or not, in display order.

### `POST /jobs` — admin

```json
{
  "key": "scaffolder",
  "display_name": "Scaffolder",
  "requirements": [
    { "item": "hardhat", "mode": "required" },
    { "item": "safety_vest", "mode": "required" },
    { "item": "boots", "mode": "required" },
    { "item": "harness", "mode": "pending" }
  ]
}
```

`201 Created` with `{ "job": …, "warnings": […], "applies_within_s": 30 }`.
The new job goes last and is active. Errors: `409 already_exists`,
`422 invalid_key`, `422 unknown_item`, `422 no_required_item`,
`422 invalid`.

### `PATCH /jobs/{key}` — admin

```json
{ "version": 4, "display_name": "Welder (hot work)", "is_active": false }
```

Either field may be left out. `200 OK` with `{ "job": …,
"applies_within_s": 30 }`. Errors: `404 not_found`, `409 version_conflict`,
`409 last_active_job`.

A deactivated job disappears from the app's job list, and checks against it
get `422 unknown_job`. Reactivating it brings it back unchanged. Jobs are
never deleted, because checks refer to them.

### `PUT /jobs/{key}/requirements` — admin

Replaces the job's whole equipment list in one step: the dashboard's matrix
row is saved as a unit.

```json
{
  "version": 4,
  "requirements": [
    { "item": "hardhat", "mode": "required" },
    { "item": "face_shield", "mode": "required" }
  ]
}
```

`200 OK` with `{ "job": …, "warnings": […], "applies_within_s": 30 }`.
Errors: `404 not_found`, `409 version_conflict`, `422 unknown_item`,
`422 no_required_item`, and `422 invalid` if an item is listed twice. Each
error names the entry in `field`, e.g. `requirements[1].item`.

### `PUT /job-order` — admin

```json
{ "keys": ["general_labor", "welder", "electrician", "height_work", "demolition", "painter"] }
```

Sets the display order. Every job, active or not, must appear exactly once
(`422 invalid` otherwise). `200 OK` with the full job list. Versions don't
change, so a matrix page that was already open keeps working without a
reload. (A separate path, so a job keyed `order` can't collide with it.)

---

## 6. Runtime settings — admin

The few settings worth changing without a restart. Each has a default from
`.env`; a value saved here overrides it until it is cleared. API keys, the
model, the database, session lengths and cookie settings stay in `.env` and
are never shown here.

### `GET /settings`

Each setting describes itself, so the dashboard builds its form from this
answer instead of hard-coding ranges:

```json
[
  { "name": "confidence_threshold", "type": "number", "min": 0.25, "max": 0.80, "step": 0.05,
    "choices": null, "confirm": true,
    "description": "Minimum detector confidence for a detection to count.",
    "value": 0.45, "default": 0.45, "overridden": false, "updated_by": null, "updated_at": null },
  { "name": "gemini_verify", "type": "boolean", "min": null, "max": null, "step": null,
    "choices": null, "confirm": false,
    "description": "Whether Gemini looks at the photo when the detector can't find a required item.",
    "value": true, "default": true, "overridden": false, "updated_by": null, "updated_at": null },
  { "name": "gemini_min_confidence", "type": "number", "min": 0.50, "max": 1.00, "step": 0.05,
    "choices": null, "confirm": false,
    "description": "How sure Gemini must be for its answer to change a verdict.",
    "value": 0.75, "default": 0.75, "overridden": false, "updated_by": null, "updated_at": null },
  { "name": "worker_auth", "type": "choice", "min": null, "max": null, "step": null,
    "choices": ["optional", "required"], "confirm": true,
    "description": "Whether a check needs a worker login (auth-api.md §5).",
    "value": "optional", "default": "optional", "overridden": false,
    "updated_by": null, "updated_at": null }
]
```

- `confirm: true` means the dashboard shows the old and new value and asks
  for confirmation before saving.
- `confidence_threshold` is limited to 0.25–0.80. At very low thresholds
  almost anything counts as a detection, which produces false passes, the
  worst failure for a safety check. Tighten the range further if the
  benchmark's F1-confidence curve says so.

### `PATCH /settings`

A JSON Merge Patch (RFC 7396) of setting values:

```json
{ "gemini_min_confidence": 0.8, "worker_auth": "required", "confidence_threshold": null }
```

- A setting left out is unchanged.
- A setting given a value is set to it.
- `null` clears the override, returning the setting to its `.env` default.
- The patch is all or nothing: if any one value is rejected, none is saved.

`200 OK` with `{ "settings": [ …full list as GET… ], "warnings": [ … ],
"applies_within_s": 30 }`. `warnings` is always present, empty when there is
nothing to say. Error: `422 out_of_range` (or `invalid` for a value of the
wrong type or not in `choices`), with `field` naming the setting.

---

## 7. Dashboard users — admin

### A user

```json
{
  "id": 2,
  "email": "supervisor@example.com",
  "display_name": "Site Supervisor",
  "role": "supervisor",
  "site": "pune_tower_b",
  "is_active": true,
  "must_change_password": true,
  "locked_until": null,
  "last_login_at": null,
  "created_at": "2026-10-16T10:00:00Z"
}
```

`site` is the supervisor's site key, and `null` for an admin.

### `GET /users`

Every user, active or not.

### `POST /users`

```json
{ "email": "supervisor@example.com", "display_name": "Site Supervisor", "role": "supervisor",
  "site": "pune_tower_b" }
```

`201 Created`:

```json
{ "user": { … }, "temporary_password": "glade-copper-79-rivet" }
```

- The temporary password is shown **once**, for the admin to hand over. The
  user must replace it at first login (`must_change_password: true`).
- A supervisor needs an active `site`; an admin must not have one
  (`422 invalid`).
- Errors: `409 already_exists`, `404 not_found` for an unknown site,
  `422 invalid` for an invalid email, name or site combination, or for a
  deactivated site (`field: "site"`).

### `PATCH /users/{id}`

```json
{ "display_name": "…", "role": "admin", "site": null, "is_active": false }
```

Any field may be left out. `200 OK` with `{ "user": … }`.

- Making someone an admin clears their site. Making someone a supervisor
  needs a site in the same request.
- Deactivating a user ends all their sessions at once.
- Errors: `404 not_found` (the user, or an unknown `site`),
  `409 cannot_change_self` (your own role, site or active flag),
  `409 last_admin` (no active admin would remain), `422 invalid` (including a
  deactivated `site`, `field: "site"`).

### `POST /users/{id}/reset-password`

`200 OK` with `{ "temporary_password": "…" }`. Also ends all the user's
sessions and unlocks them. Resetting your own password here is refused
(`409 cannot_change_self`); use `POST /auth/password`.

---

## 8. Workers — admin, supervisor (own site)

### A worker

```json
{
  "worker_id": "W-0042",
  "display_name": "Asha Verma",
  "site": "pune_tower_b",
  "is_active": true,
  "must_change_pin": false,
  "locked_until": null,
  "last_login_at": "2026-10-20T06:58:00Z",
  "created_at": "2026-10-16T10:05:00Z",
  "created_by": "supervisor@example.com"
}
```

`created_by` is the email of the user who made the worker, or `null` for a
worker made from the command line (`cli.py auth create-worker`). The audit log
records those as actor `cli`.

Worker IDs are unique across all sites.

### `GET /workers?search=&site=&active=&locked=&limit=&cursor=`

Paged (§2), with `total`.

- `search` matches the worker ID or name, ignoring case.
- `site` filters to one site. Admins only; a supervisor always gets their
  own site.
- `active` is `true`, `false`, or left out for both.
- `locked=true` lists the workers locked out right now.

### `GET /workers/{worker_id}`

One worker. `worker_id` matches without regard to case.

### `POST /workers`

```json
{ "worker_id": "W-0043", "display_name": "Ravi Kumar", "site": "pune_tower_b" }
```

`201 Created`:

```json
{ "worker": { … }, "one_time_pin": "482913" }
```

- The PIN is shown **once**; the worker replaces it at first login.
- An admin must give `site`. A supervisor's new workers go to their own
  site; any other `site` is `403 forbidden`.
- Errors: `409 already_exists`, `404 not_found` for an unknown site,
  `422 invalid` for an invalid worker ID (`auth-api.md` §4) or name, or for a
  deactivated site (`field: "site"`).

### `PATCH /workers/{worker_id}`

```json
{ "display_name": "…", "site": "pune_tower_c", "is_active": false }
```

`200 OK` with `{ "worker": … }`. Only an admin may change `site`.
Errors: `404 not_found` (the worker, or an unknown `site`),
`403 forbidden` (a supervisor sending `site`), `422 invalid` (a deactivated
`site`, `field: "site"`).
Deactivating a worker ends their sessions at once. Workers are never
deleted here; erasing a worker's data on request belongs with photo storage
in Phase 5.

### `POST /workers/{worker_id}/reset-pin`

`200 OK` with `{ "one_time_pin": "…" }`. Unlocks the worker and ends all
their sessions. This is what a supervisor does for a locked-out worker.

---

## 9. Audit log — admin

Every change made through this API, and every account or site created or
reset from the CLI, is recorded in the same database transaction as the
change itself: if the change is saved, so is its record.

### An entry

```json
{
  "id": 318,
  "at": "2026-10-16T10:12:00Z",
  "actor": "admin@example.com",
  "action": "job.requirements",
  "target": "job:welder",
  "site": null,
  "before": { "requirements": [ … ] },
  "after": { "requirements": [ … ] },
  "ip": "192.168.29.20"
}
```

- `actor` is the user's email, or `cli` for a command run on the server.
- `site` is the site the change concerns (a worker's or supervisor's), or
  `null`.
- `before` and `after` hold only the keys below, and **never** a password,
  PIN, hash or token.

| `action` | `target` | Keys in `before` / `after` |
|---|---|---|
| `site.create` | `site:<key>` | after: `key`, `display_name` |
| `site.update` | `site:<key>` | `display_name`, `is_active` (only those that changed) |
| `job.create` | `job:<key>` | after: `key`, `display_name`, `requirements` |
| `job.update` | `job:<key>` | `display_name`, `is_active` (only those that changed) |
| `job.requirements` | `job:<key>` | `requirements` |
| `job.order` | `jobs` | `keys` |
| `settings.update` | `setting:<name>` | `value` (one entry per setting changed; `null` = back to default) |
| `user.create` | `user:<email>` | after: `email`, `display_name`, `role`, `site` |
| `user.update` | `user:<email>` | `display_name`, `role`, `site`, `is_active` (only those that changed) |
| `user.reset_password` | `user:<email>` | none: the entry records only that it happened |
| `worker.create` | `worker:<id>` | after: `worker_id`, `display_name`, `site` |
| `worker.update` | `worker:<id>` | `display_name`, `site`, `is_active` (only those that changed) |
| `worker.reset_pin` | `worker:<id>` | none: the entry records only that it happened |

### `GET /audit?action=&target=&actor=&site=&from=&to=&limit=&cursor=`

Newest first, paged (§2), with `total`. `from` and `to` are timestamps
(inclusive, exclusive). The other filters match exactly. Leave any of them
out for everything.

---

## 10. Coming later: checks and their review (Phase 5)

Supervisors are meant to review checks, `NEEDS_REVIEW` above all
(AGENTS.md §15). Checks aren't stored yet, so there is nothing to review in
this phase. Phase 5 stores every check, with the worker's photo, and gets
its own contract:

- a checks list, filtered by site, worker, job, verdict and date
- a check's detail, with the photo and Gemini's findings
- a review action on `NEEDS_REVIEW` checks (compliant / not compliant, with
  a note), recorded in the audit log

Until then, the supervisor's part of the dashboard is the workers page and
the read-only matrix. Don't build a review view yet.

---

## 11. What each side builds

### Backend (Harshit)

- Migration `0003`:
  - a `sites` table, with `site_id` on `workers` (required) and on `users`
    (required for supervisors, empty for admins), backfilled to a `main` site
  - a `version` column on `jobs`
  - tables `app_settings` (name, value, updated_by, updated_at) and
    `audit_log` (columns as §9)
- The endpoints above, using `require_user(...)` from Phase 2 with the roles
  in §1, and site scoping for supervisors
- `site` in the `user` object of `POST /auth/login` and `GET /auth/me`
  (`auth-api.md` §6): the site key, `null` for an admin. The dashboard needs
  it on every page load, to hide the site filter from a supervisor and to
  default the site of a new worker
- Settings read through a cache like the job matrix, so the detector
  threshold, Gemini verification and worker login follow a change without a
  restart. Cache invalidation after every matrix or settings write
- A metrics file saved next to the weights at training time, holding AP50
  per class, read for `GET /items`. One is written by hand for the current
  model from its benchmark
- Audit entries written in the same transaction as each change, plus from
  the CLI. CLI: `auth create-worker` gains `--site` (optional when there is
  only one active site), and a `sites create` command
- Tests for every rule here, against PostgreSQL. The existing "every route
  needs a login" test covers the new routes automatically; new tests check
  that each route refuses a supervisor where §1 says so, and never shows a
  supervisor another site's workers

### Dashboard (Priyanka, Phase 4)

| Page | Uses |
|---|---|
| Matrix: jobs × items grid, each cell Required / Manual / Off, saved per row, with each item's AP50 and warning badges | §4, §5 |
| Jobs: add, rename, reorder, deactivate | §5 |
| Sites (admin only): add, rename, deactivate | §3 |
| Workers: search, filter by site and "locked", add (show the one-time PIN once), deactivate, reset PIN | §8 |
| Users (admin only): add (show the temporary password once), role, site, deactivate, reset | §7 |
| Settings (admin only), generated from `GET /settings`, confirming where `confirm` is true | §6 |
| Audit log (admin only), with date range | §9 |

- Hide what the signed-in role can't use, but treat `403 forbidden` as the
  real rule.
- On `409 version_conflict`, show the job from `current` and what changed;
  no second request is needed.

## 12. Decisions on the review questions

Answered in Priyanka's review of the first draft (PR #10), and adopted:

1. **Supervisors and the matrix: read-only.** The matrix decides every
   worker's verdict. Per-site job lists are a later option now that sites
   exist (§1).
2. **Bulk worker import: later, as a CLI command first.** A page showing a
   hundred one-time PINs at once defeats "shown once".
   `cli.py auth import-workers <file.csv>`, printing a sheet to hand out,
   comes when a site needs it.
3. **List of active sessions: not now.** Deactivation and resets already end
   every session, which covers a lost phone or laptop.
4. **Detector threshold: in the dashboard, admins only,** with a narrower
   range (0.25–0.80) and a confirmation showing the old and new value. Every
   change is audited, where editing `.env` left no trace.
5. **Accuracy per item: per model, not per item.** `GET /items` returns
   `detector_ap50` from the loaded model's metrics file (§4).
6. **Sites: in scope now** (decided by Harshit), so `site_id` is added to
   workers and users in this phase rather than retrofitted later.
