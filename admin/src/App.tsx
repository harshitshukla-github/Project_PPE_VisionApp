import type { DashboardUser } from "./api/client";
import { useAuth } from "./auth/AuthContext";
import { AppShell } from "./components/AppShell";
import { ChangePasswordPage } from "./pages/ChangePasswordPage";
import { LoginPage } from "./pages/LoginPage";
import { DEFAULT_PATH, PAGES } from "./pages/registry";
import { useHashPath } from "./router";

/**
 * Which screen to show is decided by who is signed in (auth-api §6):
 * checking → spinner; signed out → login; still on the temporary password →
 * change password; otherwise the dashboard. When the state changes (login,
 * logout, a 401 anywhere) the screen changes with it.
 */
export function App() {
  const { auth } = useAuth();

  if (auth.kind === "loading") {
    return (
      <main className="card" aria-busy="true">
        <p className="hint">Loading…</p>
      </main>
    );
  }
  if (auth.kind === "signedOut") return <LoginPage notice={auth.notice} />;
  if (auth.user.must_change_password) return <ChangePasswordPage />;
  return <Dashboard user={auth.user} />;
}

/** The signed-in app: the page for the address, if this role may have it. */
function Dashboard({ user }: { user: DashboardUser }) {
  const hash = useHashPath();
  const path = hash === "/" ? DEFAULT_PATH : hash;
  const page = PAGES.find((p) => p.path === path);

  return (
    <AppShell user={user} path={path}>
      {!page ? (
        <section className="page">
          <h1>Page not found</h1>
          <p>
            <a href={`#${DEFAULT_PATH}`}>Go to the matrix</a>
          </p>
        </section>
      ) : !page.roles.includes(user.role) ? (
        <section className="page">
          <h1>You don&apos;t have access to this page</h1>
          <p>{page.label} is for admins. If you need it, ask an admin.</p>
          <p>
            <a href={`#${DEFAULT_PATH}`}>Go to the matrix</a>
          </p>
        </section>
      ) : (
        // A key per page, so moving between pages starts each one fresh.
        <div key={page.path}>{page.render(user)}</div>
      )}
    </AppShell>
  );
}
