# PPB admin dashboard

React + Vite + TypeScript, served by FastAPI at `/admin` in production
(AGENTS.md §15). Built to [`docs/auth-api.md`](../docs/auth-api.md) §6 (sign-in)
and [`docs/admin-api.md`](../docs/admin-api.md) (everything else).

## Pages

| Page | Who | What it does |
|---|---|---|
| Matrix | admin, supervisor | Jobs × items grid, each cell Required / Manual / Off, saved a row at a time with the job's version. Shows each item's AP50 or "Gemini only", the server's warnings as badges, and, on a `409 version_conflict`, what the other person changed (using `current`). A supervisor sees it read-only. |
| Jobs | admin | Add, rename, reorder, deactivate. Never deletes: checks refer to jobs by key. |
| Workers | admin, supervisor | Search, site, status and "locked out now" filters, paging ("26–50 of 312"), add, edit, move (admin), deactivate, reset PIN. The one-time PIN is shown once. A supervisor sees only their own site. |
| Sites | admin | Add, rename, deactivate (refused while a site has active workers or supervisors). |
| Users | admin | Add (temporary password shown once), edit role and site, reset password, deactivate. You can't act on your own account. |
| Settings | admin | The form is generated from `GET /settings`; saves a merge patch of what changed, all or nothing, and asks first where a setting says `confirm`. |
| Audit log | admin | Filters by action, target, who, site and date; paging; click a row for before/after. |

Not built, on purpose: a view for supervisors to review checks. Checks aren't
stored until Phase 5 (`docs/admin-api.md` §10).

## Run it

Start the backend first (`uv run python cli.py serve --model yolo11n --device cpu --port 8000`),
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
- **Over a LAN address** (`http://192.168.x.x:5173`) the cookie also needs
  `PPB_COOKIE_SECURE=false`.

## Sample data until the backend exists

The `/admin/*` endpoints are Phase 3 and aren't built yet. Until they are,
`npm run dev` answers them from a stand-in in the browser
(`src/mock/backend.ts`) that follows `docs/admin-api.md` exactly: the same
shapes, error codes, roles, site scoping, versions, warnings and audit entries.
Sign-in (`/auth/*`) is always the real server. A banner on every page says when
the data is sample data, and a reload starts from the seed again.

**To switch to the real backend, set one line** in `admin/.env.development.local`
(or any `.env` file Vite reads):

```
VITE_ADMIN_API=real
```

`VITE_ADMIN_API=mock` forces the stand-in. With it unset, `npm run dev` uses
the stand-in and a production build uses the real server. The stand-in is
loaded on demand and left out of production builds (checked: none of its data
is in the built files).

The stand-in knows who is signed in, so a supervisor really gets a supervisor's
view. The real login can't give you one until a supervisor account exists, but
the tests cover it, and `src/testing/harness.ts` shows how to start the
stand-in as any user.

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
  `401 not_authenticated` anywhere returns the user to the login page.
- `src/api/types.ts` is the contract's shapes written down once. The pages and
  the stand-in are both typed against it.
- `src/api/problems.ts` turns an error `code` into words. Server trouble (any
  `503`) is an amber "try again shortly" notice, never a red error.
- `src/auth/AuthContext.tsx` holds who is signed in. `GET /auth/me` runs on
  every page load; until it answers, the app shows a spinner, not the login page.
- Pages are chosen by a tiny hash router (`#/workers`): nothing to configure on
  the server for deep links. `src/pages/registry.tsx` lists them, with who gets
  a link. A page a role can't use isn't opened at all; the server's
  `403 forbidden` is still the real rule.
- Forms use `Field` (`src/components/widgets.tsx`), which keeps a field's error
  outside its label and ties it to the input with `aria-describedby`, so a
  screen reader says "Email" and then the problem, not one long name.
