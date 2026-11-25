import { useEffect } from "react";

import type { DashboardUser } from "./api/client";
import { useAuth } from "./auth/AuthContext";
import { AppShell } from "./components/AppShell";
import { ErrorBoundary } from "./components/ErrorBoundary";
import { ChangePasswordPage } from "./pages/ChangePasswordPage";
import { LoginPage } from "./pages/LoginPage";
import { UnavailablePage } from "./pages/UnavailablePage";
import { DEFAULT_PATH, PAGES, PASSWORD_PATH } from "./pages/registry";
import { currentPath, replacePath, useHashPath } from "./router";

/**
 * Which screen to show is decided by who is signed in (auth-api §6):
 * checking → spinner; server didn't answer → "try again"; signed out → login; still on the temporary password →
 * change password; otherwise the dashboard. When the state changes (login,
 * logout, a 401 anywhere) the screen changes with it.
 */
export function App() {
  const { auth } = useAuth();
  const forced = auth.kind === "signedIn" && auth.user.must_change_password;

  // While the change is forced, the address is never the voluntary Change password page. Otherwise
  // finishing the forced change at `#/password` would open that page: a second, empty form with no
  // confirmation, as if the save had failed. Done while the forced page is up, not on success,
  // because the dashboard reads the address the moment the change is accepted.
  useEffect(() => {
    if (!forced) return;
    const leave = () => {
      if (currentPath() === PASSWORD_PATH) replacePath(DEFAULT_PATH);
    };
    leave();
    window.addEventListener("hashchange", leave);
    return () => window.removeEventListener("hashchange", leave);
  }, [forced]);

  if (auth.kind === "loading") {
    return (
      <main className="card" aria-busy="true">
        <p className="hint">Loading…</p>
      </main>
    );
  }
  if (auth.kind === "unavailable") return <UnavailablePage problem={auth.problem} />;
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
      {/* A key per address: a page that broke is tried afresh when it is opened again. */}
      <ErrorBoundary key={path}>
      {path === PASSWORD_PATH ? (
        <ChangePasswordPage voluntary />
      ) : !page ? (
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
      </ErrorBoundary>
    </AppShell>
  );
}
