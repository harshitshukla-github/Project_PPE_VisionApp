import { useAuth } from "./auth/AuthContext";
import { ChangePasswordPage } from "./pages/ChangePasswordPage";
import { HomePage } from "./pages/HomePage";
import { LoginPage } from "./pages/LoginPage";

/**
 * Which page to show is decided by who is signed in (auth-api §6):
 * checking → spinner; signed out → login; still on the temporary password →
 * change password; otherwise the dashboard. When the state changes (login,
 * logout, a 401 anywhere) the page changes with it.
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
  return <HomePage />;
}
