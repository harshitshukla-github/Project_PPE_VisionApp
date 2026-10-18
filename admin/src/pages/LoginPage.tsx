import { useState, type FormEvent } from "react";

import { problemFrom, type Problem } from "../api/problems";
import { useAuth } from "../auth/AuthContext";
import { ProblemBanner } from "../components/ProblemBanner";

/**
 * Login page — auth-api §6. Signing in flips the auth state and App swaps
 * the page, so there is nothing to navigate to here.
 */
export function LoginPage({ notice }: { notice?: Problem }) {
  const { signIn } = useAuth();
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [submitting, setSubmitting] = useState(false);
  const [problem, setProblem] = useState<Problem | null>(null);

  // A wrong try's message replaces the "session ended" note that came before it.
  const shown = problem ?? notice;
  const canSubmit = email.trim() !== "" && password !== "" && !submitting;

  async function handleSubmit(event: FormEvent) {
    event.preventDefault();
    if (!canSubmit) return;
    setSubmitting(true);
    setProblem(null);
    try {
      await signIn(email, password);
    } catch (err) {
      setPassword("");
      setProblem(problemFrom(err, "Couldn't sign in. Try again."));
      setSubmitting(false);
    }
  }

  return (
    <main className="card">
      <h1>PPB Admin</h1>
      <p className="hint">Sign in with your dashboard account.</p>

      {shown && <ProblemBanner problem={shown} />}

      <form onSubmit={handleSubmit}>
        <label>
          Email
          <input
            type="email"
            value={email}
            onChange={(e) => setEmail(e.target.value)}
            autoComplete="username"
            autoFocus
            disabled={submitting}
          />
        </label>
        <label>
          Password
          <input
            type="password"
            value={password}
            onChange={(e) => setPassword(e.target.value)}
            autoComplete="current-password"
            disabled={submitting}
          />
        </label>
        <button type="submit" disabled={!canSubmit}>
          {submitting ? "Signing in…" : "Sign in"}
        </button>
      </form>
    </main>
  );
}
