# PPB admin dashboard

React + Vite + TypeScript, served by FastAPI at `/admin` in production
(AGENTS.md §15). Built to [`docs/auth-api.md`](../docs/auth-api.md) §6 (sign-in),
[`docs/admin-api.md`](../docs/admin-api.md), and
[`docs/checks-api.md`](../docs/checks-api.md) (checks, photos and review, Phase 6).

## Pages

| Page | Who | What it does |
|---|---|---|
| Matrix | admin, supervisor | Jobs × items grid, each cell Required / Manual / Off, saved a row at a time with the job's version. Shows each item's AP50 or "Gemini only", the server's warnings as badges, and, on a `409 version_conflict`, what the other person changed (using `current`). A supervisor sees it read-only. |
| Checks | admin, supervisor | Every stored check, newest first, with filters (verdict, review, worker, job, login, site, dates) and paging. A supervisor starts on their site's checks waiting for review; the navigation shows how many. `#/checks?worker=W-0042` opens one worker's checks. |
| A check (`#/checks/1001`) | admin, supervisor | The photo with the detector's boxes drawn on it, what the job required then, Gemini's findings, the three settings the verdict was decided with, and the review: any verdict, once; an admin can replace one. If someone reviews it first, their review is shown, not overwritten. |
| Jobs | admin | Add, rename, reorder, deactivate. Never deletes: checks refer to jobs by key. |
| Workers | admin, supervisor | Search, site, status and "locked out now" filters, paging ("26–50 of 312"), add, edit, move (admin), deactivate, reset PIN. The one-time PIN is shown once. A supervisor sees only their own site. Each worker links to their checks; an admin can delete a worker's photos, after a confirmation that counts them. |
| Sites | admin | Add, rename, deactivate (refused while a site has active workers or supervisors). |
| Users | admin | Add (temporary password shown once), edit role and site, reset password, deactivate. You can't act on your own account. |
| Settings | admin | The form is generated from `GET /settings`; saves a merge patch of what changed, all or nothing, and asks first where a setting says `confirm`. Next to worker login, the number of checks without a login over the last 3 working days (`auth-api.md` §5). |
| Audit log | admin | Filters by action, target, who, site and date; paging; click a row for before/after. |

**Until the backend's Phase 6 lands**, the real server has no `/admin/checks`
endpoints. The Checks pages then say "This server doesn't have this part of the
dashboard yet", the navigation shows no count, and Settings leaves the count
out. With `VITE_ADMIN_API=mock` (below) they work against sample checks.

The photo is loaded from its own address (`/api/v1/admin/checks/{id}/photo`),
which the page's `img-src 'self'` policy allows, and the boxes are an SVG laid
over it in the photo's own pixels, drawn with attributes and classes only
(the page policy forbids inline styles).

## Run it

The pages need the Phase 3 admin API (`/api/v1/admin/*`, on `main`), which
needs the database at migration `0003`. Bring the backend up to date first:

```bash
git switch main && git pull
uv sync
uv run python cli.py db migrate        # schema at 0003
uv run python cli.py serve --model yolo11n --device cpu --port 8000
```

then:

```bash
cd admin
npm ci
npm run dev      # http://localhost:5173/admin/
```

Create a dashboard account once, if you don't have one (it asks for the
password; 12 characters at least):

```bash
uv run python cli.py auth create-admin --email you@example.com --name "Your Name"
```

The dev server forwards `/api` to `http://localhost:8000`, so the browser only
ever talks to `localhost:5173`. That keeps the session cookie
(`HttpOnly; SameSite=Strict; Path=/api`) first-party, with no CORS setup.

- **Safari** won't keep a `Secure` cookie from plain `http://localhost`. Set
  `PPB_COOKIE_SECURE=false` in `.env` and restart the backend, or use Chrome or
  Firefox.
- **Over a LAN address** (`http://192.168.x.x:5173` or `:8000/admin/`) the browser
  drops the `Secure` cookie, because the page isn't HTTPS. The dashboard says so
  right after you sign in ("You signed in, but your browser didn't keep the
  session…"), instead of letting you in and then bouncing you out. Use an
  `https://` address or, on the computer that runs the server, `http://localhost`
  (on any other device, `localhost` is that device). For development only, you can set
  `PPB_COOKIE_SECURE=false` in `.env` and restart the backend; the dashboard
  then signs in normally.

## Serve the build from FastAPI

This is how it runs in production: one address for the pages and the API, and no
dev server.

```bash
cd admin
npm ci
npm run build
```

`npm run build` writes to `admin/dist`, which isn't in git. Then, with the backend
running, open <http://localhost:8000/admin/>. Rebuilding needs no restart. Until
there is a build, `/admin/` says how to make one, and the API and the mobile app
work as usual.

Use HTTPS, or `localhost` on the computer that runs the server. Over plain `http://192.168.x.x:8000` the browser drops
the `Secure` session cookie, so sign-in won't stick; `PPB_COOKIE_SECURE=false` gets
round that for development only.

## Sample data, when you want it

By default the pages talk to the real server. To work without a backend, or to
see what a supervisor sees without having a supervisor account, answer the
`/admin/*` calls from a stand-in in the browser instead. Put one line in
`admin/.env.development.local` (any `.env` file Vite reads):

```
VITE_ADMIN_API=mock
```

The stand-in (`src/mock/backend.ts`) follows `docs/admin-api.md`: the same
shapes, error codes, roles, site scoping, versions, warnings and audit entries,
and it knows who is signed in, so a supervisor really gets a supervisor's view.
Sign-in (`/auth/*`) is still the real server. A banner on every page says the
data is sample data, and a reload starts from the seed again.

The stand-in is loaded on demand and left out of production builds (checked:
none of its data is in the built files). It was written against the contract
before the backend existed, then compared with the backend's code: the same
routes, field names, warning codes, settings descriptors and error fields.
`src/testing/harness.ts` shows how a test starts it as any user.

## Check it

```bash
npm run typecheck
npm test
npm run build
```

## How it works

- `src/api/client.ts` is the only code that calls `fetch`, and the one place
  to change when the mock goes. The CSRF token is kept in memory (never
  `localStorage`), sent as `X-CSRF-Token` on every write, refreshed from
  `GET /auth/me` and the write retried once on `403 csrf_failed`. A
  `401 not_authenticated` anywhere returns the user to the login page. A
  request gives up after 20 seconds (`REQUEST_TIMEOUT_MS`) as an outage, so a
  hung server ends in a message, not a spinner; a write that timed out says it
  may or may not have been saved.
- Each page is drawn inside an error boundary (`src/components/ErrorBoundary.tsx`):
  a page that breaks says so in place, and the navigation still works.
- `src/api/types.ts` is the contract's shapes written down once. The pages and
  the stand-in are both typed against it.
- `src/api/problems.ts` turns an error `code` into words. An outage (`isOutage`:
  no answer at all, or a `502`/`503`/`504` from the server or a proxy in front of
  it) is an amber notice, never a red error: it isn't the person's doing. A
  `502`/`504` on a write says it may or may not have been saved (the proxy gave
  up, the server may not have), and a `503 auth_unavailable` shows the server's
  own reason, so "no database configured" isn't hidden behind "try again".
- `src/auth/AuthContext.tsx` holds who is signed in. `GET /auth/me` runs on
  every page load; until it answers, the app shows a spinner, not the login page.
  On an outage the app says so and offers "Try again" instead of showing a
  sign-in form: nobody knows yet whether the session is still good. Any other
  error there (a `500`, say) means the server is up, so the sign-in form opens
  with a note saying the check failed.
- The check right after a sign-in (did the browser keep the cookie?) is
  `getSession({ quiet: true })`: its `401` is explained on the sign-in page, so
  it doesn't also count as "Your session ended".
- While a password change is forced, the address is moved off `#/password`, so
  finishing it lands on the dashboard rather than on the voluntary form.
- "Change password" in the header opens the same page that is forced on a first
  sign-in, inside the normal frame, with its own wording and a way back.
- A picked day in the audit filter means that day where the person is
  (`src/dates.ts`), because the table shows local times; the API gets the local
  midnights converted to UTC.
- Pages are chosen by a tiny hash router (`#/workers`): nothing to configure on
  the server for deep links. `src/pages/registry.tsx` lists them, with who gets
  a link. A page a role can't use isn't opened at all; the server's
  `403 forbidden` is still the real rule.
- Forms use `Field` (`src/components/widgets.tsx`), which keeps a field's error
  outside its label and ties it to the input with `aria-describedby`, so a
  screen reader says "Email" and then the problem, not one long name.
