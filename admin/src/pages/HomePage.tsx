import { useAuth } from "../auth/AuthContext";

/**
 * Placeholder for the signed-in dashboard. The matrix, jobs, workers, users,
 * settings and audit pages come after the Phase 3 admin API contract
 * (docs/admin-api.md, PR #10) is agreed; building them first would mean
 * guessing at shapes that are still being reviewed.
 */
export function HomePage() {
  const { auth, signOut } = useAuth();
  if (auth.kind !== "signedIn") return null;
  const { user } = auth;

  return (
    <main className="card">
      <header className="bar">
        <div>
          <h1>PPB Admin</h1>
          <p className="hint" data-testid="whoami">
            {user.display_name} · {user.role}
          </p>
        </div>
        <button type="button" className="link" onClick={() => void signOut()}>
          Sign out
        </button>
      </header>
      <p>You are signed in. The admin pages arrive once the admin API contract is agreed.</p>
    </main>
  );
}
