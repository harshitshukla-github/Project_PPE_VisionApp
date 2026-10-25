# Phase 2 — Login API contract

**Status: agreed (merged 2026-10-16). The backend (§1–§7) is built and on
`main`; the mobile app and dashboard parts (§8) are in progress.**

The backend, the mobile app and the admin dashboard are built against this
document. Any change from now on goes through a pull request that edits this
file first.

All paths are under `/api/v1`. Requests and responses are JSON unless noted.

---

## 1. Who signs in, and how

| Who | Signs in with | Where | Session carried by |
|---|---|---|---|
| Dashboard users: **admin**, **supervisor** | email + password | Admin dashboard (browser) | `ppb_session` cookie |
| **Workers** | worker ID + 6-digit PIN | Mobile app | `Authorization: Bearer <token>` header |

There is **no sign-up**. The first admin is created from the command line
(§7). After that, admins create dashboard users, and admins or supervisors
create workers and issue their first PIN. Workers must replace that PIN with
their own at first login, so the person who issued it doesn't know it.

## 2. Who may do what

| Action | admin | supervisor | worker |
|---|:-:|:-:|:-:|
| Run a compliance check (`POST /compliance/check`) | | | ✓ |
| Edit the job matrix and settings (Phase 3) | ✓ | | |
| Create, deactivate and reset dashboard users (Phase 3) | ✓ | | |
| Create and deactivate workers, reset PINs (Phase 3) | ✓ | ✓ | |
| View checks and work the review queue (Phase 5) | ✓ | ✓ | |

Account management endpoints belong to the Phase 3 admin API and get their
own document. In Phase 2, accounts are created from the CLI (§7).

## 3. Conventions

### Errors

Every error keeps `detail` as a **string written for a person**, so the
mobile app's current error display keeps working. A machine-readable `code`
is added next to it:

```json
{ "detail": "Too many wrong PINs. Try again in 14 minutes.", "code": "account_locked", "retry_after_s": 840 }
```

FastAPI's own validation errors (422 with a list in `detail`) are unchanged.

| HTTP | `code` | When | The app or dashboard should |
|---|---|---|---|
| 401 | `not_authenticated` | No token or cookie, or it expired or was revoked | Go to the login screen |
| 401 | `invalid_credentials` | Wrong worker ID or PIN, wrong email or password, or no such account | Show "Wrong worker ID or PIN" |
| 403 | `account_disabled` | Credentials were right, but the account is deactivated | "Account disabled — ask your supervisor" |
| 403 | `pin_change_required` | Worker hasn't replaced the issued PIN yet | Open the Change PIN screen |
| 403 | `password_change_required` | Dashboard user must change their password first | Open the change-password screen |
| 403 | `forbidden` | Signed in, but this role may not do this | "You don't have access to this" |
| 403 | `csrf_failed` | Dashboard write without the right `X-CSRF-Token` | Call `GET /auth/me`, retry once |
| 422 | `weak_pin`, `weak_password` | A new PIN or password breaks the rules (§4, §6) | Show `detail` |
| 423 | `account_locked` | Too many wrong attempts on this account | Show when to retry; a worker can ask a supervisor to reset the PIN |
| 429 | `too_many_attempts` | Too many attempts from this device or network | Wait for `Retry-After` |
| 503 | `auth_unavailable` | The server has no database configured or can't reach it, so it can't check a login | "Sign-in unavailable, try again shortly" |

401 responses to bearer-token requests carry `WWW-Authenticate: Bearer`.
423 and 429 carry a `Retry-After` header in seconds, matching `retry_after_s`.

### Timestamps

ISO 8601 in UTC, e.g. `"2026-10-16T03:12:00Z"`.

---

## 4. Worker login (mobile app)

### `POST /auth/worker/login`

```json
{ "worker_id": "W-0042", "pin": "482913" }
```

`200 OK`

```json
{
  "token": "Jx3v9…(43 characters)",
  "token_type": "bearer",
  "expires_at": "2026-10-16T03:12:00Z",
  "must_change_pin": true,
  "worker": { "worker_id": "W-0042", "display_name": "Asha Verma" }
}
```

- `worker_id` is matched without regard to case (`w-0042` works).
- The token is shown once. The server stores only a hash of it.
- A session lasts **12 hours** from login (one shift), then the worker signs
  in again. There is no idle timeout.
- Errors: `401 invalid_credentials`, `403 account_disabled`,
  `423 account_locked`, `429 too_many_attempts`.

### `POST /auth/worker/pin` — choose your own PIN

```json
{ "current_pin": "482913", "new_pin": "730154" }
```

`204 No Content`. Clears `must_change_pin`, and signs the worker out of every
other session.

**While `must_change_pin` is true**, the token works only for this endpoint,
`GET /auth/worker/me` and `POST /auth/worker/logout`. Everything else returns
`403 pin_change_required`.

PIN rules (`422 weak_pin` otherwise):

- exactly 6 digits
- not one digit repeated (`111111`)
- not a straight run up or down (`123456`, `654321`, `012345`)
- different from the current PIN

### `GET /auth/worker/me`

`200 OK`

```json
{
  "worker": { "worker_id": "W-0042", "display_name": "Asha Verma" },
  "must_change_pin": false,
  "expires_at": "2026-10-16T03:12:00Z"
}
```

### `POST /auth/worker/logout`

`204 No Content`. Revokes the token. The app deletes its stored copy whatever
the response.

### Lockout

- **5 wrong PINs** in a row lock the worker for **15 minutes** (`423`).
  The count resets after a successful login.
- A supervisor resetting the PIN (Phase 3) also unlocks the worker.
- Separately, **20 failed attempts per minute from one IP address**, across
  all accounts, gives `429`.

### Worker ID format

2–32 characters: letters, digits and hyphens, not starting with a hyphen,
stored in upper case. For example `W-0042`.

---

## 5. Changes to existing endpoints

### `POST /compliance/check`

- Send the worker's token: `Authorization: Bearer <token>`.
- The `worker_id` comes from the token. The `worker_id` form field is
  **ignored** when a token is present, and the response's `worker_id` is the
  logged-in worker's.
- An expired or revoked token returns `401 not_authenticated`. The server
  never falls back to the form field when a token was sent, so the app always
  learns that its session ended.

### Rollout without breaking the current app

Setting `PPB_WORKER_AUTH`:

| Value | No token sent | Token sent |
|---|---|---|
| `optional` (default during Phase 2) | Works as today, using the `worker_id` form field; logged as unauthenticated | `worker_id` from the token |
| `required` | `401 not_authenticated` | `worker_id` from the token |

Order of release:
1. The backend ships with `optional`. The current app keeps working. ✅
2. The app ships its login screen. ✅ (PR #11)
3. The server is switched to `required`, when the conditions below are met.

### When to switch to `required`

Switching too early locks out every phone still running an app without
login. Switching late leaves checks that aren't tied to a signed-in worker.
So the switch happens when **both** of these are true:

1. **Every phone in use runs the login build** (the app from PR #11 or
   later). The supervisor of each site confirms it for their site.
2. **The server has seen no check without a login for 3 working days in a
   row.** The backend logs every such check as `[<worker_id>, not signed
   in]`, so the log shows whether any old app is still in use. Once checks
   are stored (Phase 5), the dashboard can show this count instead.

Then:

- **Who:** Harshit (an admin).
- **How:** `PPB_WORKER_AUTH=required` in `.env` and a restart, or the
  `worker_auth` setting in the dashboard once Phase 3 is built (which also
  records it in the audit log). Once a value has been saved in the dashboard
  it wins over `.env`: `GET /settings` shows `overridden: true`, and a later
  edit of `.env` has no effect until the override is cleared (`null`).
- **Undo:** set it back to `optional`. Phones with the login build keep
  working either way.
- **Record it:** add the date of the switch to this section.

### Other endpoints

| Endpoint | Signed in? |
|---|---|
| `GET /health`, `GET /model/info` | No |
| `GET /jobs` | No (job lists aren't secret, and the dashboard needs them too) |
| `POST /detect` | With `required`: a worker token or a dashboard session |

---

## 6. Dashboard login (browser)

The dashboard is served by the same FastAPI server at `/admin` (see
`AGENTS.md` §15), so the cookie is first-party and no CORS setup is needed.

### `POST /auth/login`

```json
{ "email": "admin@example.com", "password": "correct horse battery staple" }
```

`200 OK`, with a `Set-Cookie: ppb_session=…` header:

```json
{
  "user": {
    "id": 1,
    "email": "admin@example.com",
    "display_name": "Harshit",
    "role": "admin",
    "site": null,
    "must_change_password": false
  },
  "csrf_token": "f8Q2…",
  "expires_at": "2026-10-16T03:12:00Z"
}
```

`site` is the site key a supervisor belongs to (`admin-api.md` §3), and `null`
for an admin, who works across all sites. It is added when sites arrive in
Phase 3 (migration `0003`).

Errors: `401 invalid_credentials`, `403 account_disabled`,
`423 account_locked`, `429 too_many_attempts`.

### The cookie

`ppb_session` is `HttpOnly; SameSite=Strict; Path=/api; Secure`. JavaScript
can't read it, and the browser sends it by itself.

`Secure` can be turned off with `PPB_COOKIE_SECURE=false`, for development
over plain `http://` on a LAN address; `http://localhost` works with it on.

A session ends **12 hours after login, or after 2 hours without a request**,
whichever comes first.

### CSRF

Every `POST`, `PUT`, `PATCH` and `DELETE` from the dashboard sends
`X-CSRF-Token: <csrf_token>`. Without it, or with the wrong value, the
server answers `403 csrf_failed`.

The dashboard keeps the token in memory, not in `localStorage`. After a page
reload it gets the token again from `GET /auth/me`.

### `GET /auth/me`

`200 OK` with the same body as login. Returns `401 not_authenticated` when
there is no valid session. The dashboard calls it on every page load to
decide between the login page and the app.

### `POST /auth/logout`

`204 No Content`. Revokes the session and clears the cookie.

### `POST /auth/password`

```json
{ "current_password": "…", "new_password": "…" }
```

`204 No Content`. Clears `must_change_password`, and signs the user out of
every other session.

Password rules (`422 weak_password` otherwise):

- 12–128 characters (passphrases welcome)
- not the same as the email
- different from the current password

Lockout: **5 wrong passwords lock the account for 15 minutes**, as for
workers.

---

## 7. Accounts from the command line (Phase 2)

Until the Phase 3 admin API and the Phase 4 dashboard exist, accounts come
from the CLI:

```bash
uv run python cli.py auth create-admin --email admin@example.com --name "Harshit"
uv run python cli.py auth create-worker --worker-id W-0042 --name "Asha Verma"
uv run python cli.py auth reset-pin --worker-id W-0042
```

- `create-admin` asks for the password twice at a hidden prompt.
- `create-worker` and `reset-pin` print a **one-time PIN** for the
  supervisor to hand over. The worker must replace it at first login.
  `reset-pin` also unlocks a locked worker.

---

## 8. What each side builds

### Backend (Harshit)

- Migration `0002`: tables `users`, `workers` and `sessions` (below)
- Password and PIN hashing with Argon2id (`argon2-cffi`)
- The endpoints above, the CSRF check, lockouts and the per-IP limit
- Reusable checks that routes declare: "requires a worker", "requires a dashboard user with role X"
- `PPB_WORKER_AUTH`, `PPB_COOKIE_SECURE`, and the CLI commands in §7
- A test that calls every route without credentials, so a forgotten check
  fails CI

| Table | Columns |
|---|---|
| `users` | id, email (unique, lower case), display_name, role, password_hash, must_change_password, is_active, failed_attempts, locked_until, last_login_at, created_at, updated_at |
| `workers` | id, worker_code (unique, upper case; the API calls it `worker_id`), display_name, pin_hash, must_change_pin, is_active, failed_attempts, locked_until, last_login_at, created_by (user id), created_at, updated_at |
| `sessions` | id, token_hash (unique), csrf_token (dashboard only), user_id or worker_id (exactly one), created_at, last_seen_at, expires_at, revoked_at, user_agent, ip |

### Mobile app (Priyanka)

- **Login screen** (worker ID + PIN on a number pad). It replaces the
  free-text worker ID on Home.
- **Change PIN screen**, shown when `must_change_pin` is true or a request
  returns `403 pin_change_required`.
- Store the token with `expo-secure-store`, not AsyncStorage
  (`npx expo install expo-secure-store`).
- Send `Authorization: Bearer <token>` on `/compliance/check` and stop sending
  the `worker_id` form field.
- Handle the error codes in §3:
  - `401` clears the token and opens Login
  - `423` shows the retry time
  - `429` waits
- **Log out** in Settings.
- History on shared phones: show only the logged-in worker's checks, or clear
  them at logout (open question 2).

### Admin dashboard (Priyanka, start of Phase 4)

- Create the Vite + React + TypeScript app in `admin/`. In development,
  Vite's dev server forwards `/api` to FastAPI, so the cookie stays
  first-party.
- Login page; call `GET /auth/me` on load; change-password screen.
- Send `X-CSRF-Token` on every write; on `401`, go to login.

---

## 9. Why it works this way

- **Tokens are stored hashed** (SHA-256), so a copy of the database contains
  no working sessions. Tokens and PINs are never written to logs.
- **Argon2id** hashes passwords and PINs. A 6-digit PIN has only a million
  combinations, so the lockout, not the hash, is what protects it. Access to
  the database must stay restricted.
- **Unknown account and wrong secret get the same answer**, and the server
  spends the same time on both, so the login form can't be used to discover
  accounts.
- **A locked account does say it is locked (423).** That reveals the account
  exists, which is a deliberate trade-off: a worker needs to know to go to a
  supervisor, and worker IDs are printed on badges anyway.
- **Revocation is immediate:** logging out, changing a PIN or password, and
  deactivating an account all end sessions in the database. No JWT that stays
  valid until it expires.
- **The issued PIN is temporary**, so a supervisor never knows the PIN a
  worker actually uses.

## 10. Decisions on the review questions

The review raised four questions. No objection was made when the contract
merged, so these are the answers in force. Changing one is a pull request
against this file.

1. **Worker session length: 12 hours,** one sign-in per shift
   (`PPB_WORKER_SESSION_HOURS`). Chosen for shared site phones; a longer
   session would be more convenient on personal phones but leaves a check
   open to whoever picks the phone up next.
2. **History on shared phones: each worker sees only their own checks.**
   Other workers' checks stay on the phone but are never shown. Decided and
   built in the app (Priyanka, `feat/mobile-login`).
3. **Checks on someone's behalf: not supported in Phase 2.** Every check is
   run by the worker who is signed in. Revisit if sites need supervisors to
   check workers who have no phone.
4. **Worker ID format: 2–32 letters, digits and hyphens, not starting with a
   hyphen,** stored in upper case (e.g. `W-0042`). The database enforces it.
   If a site's existing IDs don't fit, widen the rule here and in the
   database (with a new migration) before creating those workers.
