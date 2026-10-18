# PPB admin dashboard

React + Vite + TypeScript, served by FastAPI at `/admin` in production
(AGENTS.md §15). Built to [`docs/auth-api.md`](../docs/auth-api.md) §6.

Today it has the login page, the change-password screen (shown while a user
still has the temporary password) and a placeholder home. The matrix, jobs,
workers, users, settings and audit pages come after
[`docs/admin-api.md`](../docs/admin-api.md) is agreed.

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

## Check it

```bash
npm run typecheck
npm test
npm run build
```

## How it works

- `src/api/client.ts` is the only code that calls `fetch`. The CSRF token is
  kept in memory (never `localStorage`), sent as `X-CSRF-Token` on every write,
  refreshed from `GET /auth/me` and the write retried once on `403 csrf_failed`.
  A `401 not_authenticated` anywhere returns the user to the login page.
- `src/api/problems.ts` turns an error `code` into words. Server trouble (any
  `503`) is an amber "try again shortly" notice, never a red error.
- `src/auth/AuthContext.tsx` holds who is signed in. `GET /auth/me` runs on
  every page load; until it answers, the app shows a spinner, not the login page.
