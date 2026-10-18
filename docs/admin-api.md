# Phase 3 — Admin API contract

**Status: draft for review by Harshit and Priyanka. Nothing here is built yet.**

This is what the admin dashboard (Phase 4) calls to run the system: the job
PPE matrix, runtime settings, dashboard users, workers, and the audit log.
Once agreed, the backend and the dashboard are built against it in parallel.
Any change after that goes through a pull request that edits this file first.

It builds on the login contract, [`docs/auth-api.md`](auth-api.md): every
endpoint here needs a dashboard session (the `ppb_session` cookie), every
write needs `X-CSRF-Token`, and errors use the same shape.

All paths are under `/api/v1/admin`. Requests and responses are JSON.

---

## 1. Who may do what

| Area | admin | supervisor |
|---|:-:|:-:|
| See the job matrix, items and jobs | ✓ | ✓ |
| Change the job matrix, add, rename, reorder or deactivate jobs | ✓ | |
| See and change runtime settings | ✓ | |
| Manage dashboard users | ✓ | |
| See, create, rename, deactivate and reactivate workers; reset PINs | ✓ | ✓ |
| Read the audit log | ✓ | |

Anything else returns `403 forbidden`. A user who still has to change their
password gets `403 password_change_required` on every endpoint here.

## 2. Conventions

### Errors

The shape from `auth-api.md` §3: `detail` is a sentence for a person, `code`
is what the dashboard branches on. Validation errors also name the `field`
the dashboard should highlight:

```json
{ "detail": "A job needs at least one required item.", "code": "no_required_item", "field": "requirements" }
```

Codes added by this contract:

| HTTP | `code` | When |
|---|---|---|
| 404 | `not_found` | No such job, user or worker |
| 409 | `already_exists` | The job key, email or worker ID is taken |
| 409 | `version_conflict` | Someone else changed this job since you loaded it; reload and reapply |
| 409 | `last_admin` | The change would leave no active admin |
| 409 | `cannot_change_self` | You tried to deactivate yourself or change your own role |
| 409 | `last_active_job` | The change would leave the app with no jobs to choose from |
| 422 | `invalid_key` | A job key isn't a lowercase identifier (`^[a-z][a-z0-9_]*$`, 2–64 characters) |
| 422 | `unknown_item` | A requirement names an item that isn't in the catalogue |
| 422 | `no_required_item` | An active job would have no required item |
| 422 | `out_of_range` | A setting is outside its allowed range (§5) |
| 422 | `invalid` | Anything else malformed: an item listed twice, an incomplete job order, a bad email or name |

### Warnings

Some valid changes deserve a second look. A successful response may carry
`warnings`: sentences the dashboard shows after saving. They never block the
change.

```json
{ "job": { "…": "…" }, "warnings": ["Face shield can only be judged by Gemini, whose accuracy on it hasn't been measured."] }
```

### Lists

Small collections (items, jobs, users, settings) come back whole. Workers and
the audit log are paged:

```
GET /api/v1/admin/workers?limit=50&cursor=…
```

```json
{ "items": [ … ], "next_cursor": "eyJpZCI6MTIzfQ" }
```

`limit` defaults to 50, at most 200. `next_cursor` is `null` on the last page.
Treat it as opaque.

### When changes take effect

The server that handled the change uses it on its next request. Any other
server process picks it up within `PPB_MATRIX_CACHE_S` (30 seconds). The
mobile app sees new job lists the next time it calls `GET /jobs`.

---

## 3. Items (read-only)

The equipment the system knows. Items come from the detector's classes, so
the dashboard can't add or remove them.

### `GET /items`

```json
[
  { "key": "hardhat", "display_name": "Hardhat", "detectable": true, "judged_by": "detector, then Gemini" },
  { "key": "face_shield", "display_name": "Face shield", "detectable": false, "judged_by": "Gemini only" }
]
```

`detectable` is whether the detector was trained on the item. A required item
that isn't detectable is judged by Gemini alone.

---

## 4. Jobs and the matrix

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
- `key` never changes after creation: checks and the app refer to jobs by
  key. Rename with `display_name`.
- `version` goes up by one with every change. Edits must send the version
  they started from; if someone else saved in between, the server answers
  `409 version_conflict` instead of overwriting their work.

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

`201 Created` with `{ "job": …, "warnings": […] }`. The new job goes last
and is active. Errors: `409 already_exists`, `422 invalid_key`,
`422 unknown_item`, `422 no_required_item`.

### `PATCH /jobs/{key}` — admin

```json
{ "version": 4, "display_name": "Welder (hot work)", "is_active": false }
```

Either field may be left out. `200 OK` with `{ "job": … }`. Errors:
`404 not_found`, `409 version_conflict`, `409 last_active_job`.

A deactivated job disappears from the app's job list and can no longer be
checked against. Reactivating it brings it back unchanged. Jobs are never
deleted, because past checks refer to them.

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

`200 OK` with `{ "job": …, "warnings": […] }`. Errors: `404 not_found`,
`409 version_conflict`, `422 unknown_item`, `422 no_required_item`, and
`422 invalid` if an item is listed twice.

Warnings returned:

- a required item that isn't detectable: it is judged by Gemini alone, and
  Gemini's accuracy on it hasn't been measured
- required items while Gemini verification is switched off (§5): anything the
  detector misses stays "Needs review"

### `PUT /job-order` — admin

```json
{ "keys": ["general_labor", "welder", "electrician", "height_work", "demolition", "painter"] }
```

Sets the display order. Every job, active or not, must appear exactly once
(`422 invalid` otherwise). `200 OK` with the full job list. (A separate path,
so a job keyed `order` can't collide with it.)

---

## 5. Runtime settings — admin

The few settings worth changing without a restart. Each has a default from
`.env`; a value saved here overrides it until it is cleared.

| Name | Type | Allowed | What it does |
|---|---|---|---|
| `confidence_threshold` | number | 0.05–0.95 | Minimum detector confidence for a detection to count |
| `gemini_verify` | boolean | | Whether Gemini looks at the photo when the detector can't find a required item |
| `gemini_min_confidence` | number | 0.50–1.00 | How sure Gemini must be for its answer to change a verdict |
| `worker_auth` | string | `optional`, `required` | Whether a check needs a worker login (`auth-api.md` §5) |

API keys, the model, the database, session lengths and cookie settings stay
in `.env` and are never shown here.

### `GET /settings`

```json
[
  { "name": "confidence_threshold", "value": 0.45, "default": 0.45, "overridden": false,
    "updated_by": null, "updated_at": null },
  { "name": "worker_auth", "value": "required", "default": "optional", "overridden": true,
    "updated_by": "admin@example.com", "updated_at": "2026-10-20T08:00:00Z" }
]
```

### `PATCH /settings`

```json
{ "gemini_min_confidence": 0.8, "worker_auth": "required" }
```

Only the settings named change. `null` clears an override and returns to the
`.env` default. `200 OK` with the full list and any `warnings`. Error:
`422 out_of_range`, with `field` naming the setting.

Warnings returned:

- `worker_auth` → `required`: checks from an app without login will be
  refused. The dashboard must ask for confirmation before sending this.
- `gemini_verify` → `false`: anything the detector misses stays "Needs
  review".
- `confidence_threshold` changed: every verdict from now on is affected.

---

## 6. Dashboard users — admin

### A user

```json
{
  "id": 2,
  "email": "supervisor@example.com",
  "display_name": "Site Supervisor",
  "role": "supervisor",
  "is_active": true,
  "must_change_password": true,
  "locked_until": null,
  "last_login_at": null,
  "created_at": "2026-10-16T10:00:00Z"
}
```

### `GET /users`

Every user, active or not.

### `POST /users`

```json
{ "email": "supervisor@example.com", "display_name": "Site Supervisor", "role": "supervisor" }
```

`201 Created`:

```json
{ "user": { … }, "temporary_password": "glade-copper-79-rivet" }
```

The temporary password is shown **once**, for the admin to hand over. The
user must replace it at first login (`must_change_password: true`). Errors:
`409 already_exists`, `422` for an invalid email or name.

### `PATCH /users/{id}`

```json
{ "display_name": "…", "role": "admin", "is_active": false }
```

Any field may be left out. `200 OK` with `{ "user": … }`.

- Deactivating a user ends all their sessions at once.
- Errors: `404 not_found`, `409 cannot_change_self` (your own role or active
  flag), `409 last_admin` (no active admin would remain).

### `POST /users/{id}/reset-password`

`200 OK` with `{ "temporary_password": "…" }`. Also ends all the user's
sessions and unlocks them. Resetting your own password here is refused
(`409 cannot_change_self`); use `POST /auth/password`.

---

## 7. Workers — admin, supervisor

### A worker

```json
{
  "worker_id": "W-0042",
  "display_name": "Asha Verma",
  "is_active": true,
  "must_change_pin": false,
  "locked_until": null,
  "last_login_at": "2026-10-20T06:58:00Z",
  "created_at": "2026-10-16T10:05:00Z",
  "created_by": "supervisor@example.com"
}
```

### `GET /workers?search=&active=&limit=&cursor=`

Paged (§2). `search` matches the worker ID or name, ignoring case. `active`
is `true`, `false`, or left out for both.

### `GET /workers/{worker_id}`

One worker. `worker_id` matches without regard to case.

### `POST /workers`

```json
{ "worker_id": "W-0043", "display_name": "Ravi Kumar" }
```

`201 Created`:

```json
{ "worker": { … }, "one_time_pin": "482913" }
```

Shown **once**; the worker replaces it at first login. Errors:
`409 already_exists`, `422` for an invalid worker ID (`auth-api.md` §4) or
name.

### `PATCH /workers/{worker_id}`

```json
{ "display_name": "…", "is_active": false }
```

`200 OK` with `{ "worker": … }`. Deactivating a worker ends their sessions at
once. Workers are never deleted here; erasing a worker's data on request
belongs with photo storage in Phase 5.

### `POST /workers/{worker_id}/reset-pin`

`200 OK` with `{ "one_time_pin": "…" }`. Unlocks the worker and ends all
their sessions. This is what a supervisor does for a locked-out worker.

---

## 8. Audit log — admin

Every change made through this API, and every account created or reset from
the CLI, is recorded in the same database transaction as the change itself:
if the change is saved, so is its record.

### An entry

```json
{
  "id": 318,
  "at": "2026-10-16T10:12:00Z",
  "actor": "admin@example.com",
  "action": "job.requirements",
  "target": "job:welder",
  "before": { "requirements": [ … ] },
  "after": { "requirements": [ … ] },
  "ip": "192.168.29.20"
}
```

- `actor` is the user's email, or `cli` for a command run on the server.
- `before` and `after` hold only what changed, and **never** a password,
  PIN, hash or token.

| `action` | Recorded when |
|---|---|
| `job.create`, `job.update`, `job.requirements`, `job.order` | §4 |
| `settings.update` | §5 |
| `user.create`, `user.update`, `user.reset_password` | §6, `cli.py auth create-admin` |
| `worker.create`, `worker.update`, `worker.reset_pin` | §7, `cli.py auth create-worker`, `reset-pin` |

### `GET /audit?action=&target=&actor=&limit=&cursor=`

Newest first, paged (§2). The filters match exactly; leave them out for
everything.

---

## 9. What each side builds

### Backend (Harshit)

- Migration `0003`: a `version` column on `jobs`, and tables `app_settings`
  (name, value, updated_by, updated_at) and `audit_log` (columns as §8)
- The endpoints above, using `require_user(...)` from Phase 2 with the roles
  in §1
- Settings read through a cache like the job matrix, so the detector
  threshold, Gemini verification and worker login follow a change without a
  restart
- Cache invalidation after every matrix or settings write
- Audit entries written in the same transaction as each change, plus from
  the CLI commands
- Tests for every rule here, against PostgreSQL. The existing "every route
  needs a login" test covers the new routes automatically; a new test checks
  that each one refuses a supervisor where §1 says so

### Dashboard (Priyanka, Phase 4)

| Page | Uses |
|---|---|
| Matrix: jobs × items grid, each cell Required / Manual / Off, saved per row | §3, §4 |
| Jobs: add, rename, reorder, deactivate | §4 |
| Workers: search, add (show the one-time PIN once), deactivate, reset PIN | §7 |
| Users (admin only): add (show the temporary password once), role, deactivate, reset | §6 |
| Settings (admin only), with a confirmation before `worker_auth: required` | §5 |
| Audit log (admin only) | §8 |

- Hide what the signed-in role can't use, but treat `403 forbidden` as the
  real rule.
- On `409 version_conflict`, reload the job and show what changed.

## 10. Open questions for review

1. **Supervisors and the matrix.** Read-only as written. Should supervisors
   be able to change it for their own site?
2. **Bulk worker import.** Onboarding a site one worker at a time is slow.
   Add a CSV import (worker ID, name → one-time PINs) in this phase, or later?
3. **Seeing and ending sessions.** As written, sessions end only through
   deactivation or a reset. Do admins need a list of active sessions with a
   "sign out" button?
4. **Which settings belong in the dashboard.** Is the detector threshold too
   risky to expose? It changes every verdict, and the audit log is the only
   trace.
5. **Accuracy per item.** The matrix page would be clearer with each item's
   detector accuracy (AP50) next to it. Store those figures in the database
   when a model is trained, or leave them to the dashboard?
