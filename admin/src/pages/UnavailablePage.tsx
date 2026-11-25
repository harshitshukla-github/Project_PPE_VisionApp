import type { Problem } from "../api/problems";
import { useAuth } from "../auth/AuthContext";
import { ProblemBanner } from "../components/ProblemBanner";

/**
 * Shown when the server didn't answer "who is signed in?" on page load. A
 * sign-in form would be no use (it couldn't reach the server either) and
 * would suggest the session was lost, which nobody knows yet.
 */
export function UnavailablePage({ problem }: { problem: Problem }) {
  const { retry } = useAuth();
  return (
    <main className="card">
      <h1>PPB Admin</h1>
      <ProblemBanner problem={problem} />
      {/* The banner says what went wrong; this doesn't promise it will pass, since a server with no
          database won't come right by itself. */}
      <p className="hint">Until the server can answer, there&apos;s no way to tell yet whether you&apos;re still signed in.</p>
      <button type="button" onClick={retry}>
        Try again
      </button>
    </main>
  );
}
