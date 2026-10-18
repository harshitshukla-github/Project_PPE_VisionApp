import { useState, type FormEvent } from "react";

import { ApiError } from "../api/client";
import { problemFrom, type Problem } from "../api/problems";
import { useAuth } from "../auth/AuthContext";
import { ProblemBanner } from "../components/ProblemBanner";

// auth-api §6: 12–128 characters. The server also refuses a password equal to
// the email or to the current one, and says why (422 weak_password).
const MIN_LENGTH = 12;
const MAX_LENGTH = 128;

/**
 * Shown while the signed-in user still has the temporary password an admin
 * gave them (`must_change_password`). Nothing else works until it is
 * replaced, so the only other way out is signing out.
 */
export function ChangePasswordPage() {
  const { auth, changePassword, signOut } = useAuth();
  const [current, setCurrent] = useState("");
  const [next, setNext] = useState("");
  const [confirm, setConfirm] = useState("");
  const [submitting, setSubmitting] = useState(false);
  const [problem, setProblem] = useState<Problem | null>(null);

  const mismatch = confirm !== "" && confirm !== next;
  const tooShort = next !== "" && next.length < MIN_LENGTH;
  const canSubmit =
    current !== "" &&
    next.length >= MIN_LENGTH &&
    next.length <= MAX_LENGTH &&
    confirm === next &&
    !submitting;

  async function handleSubmit(event: FormEvent) {
    event.preventDefault();
    if (!canSubmit) return;
    setSubmitting(true);
    setProblem(null);
    try {
      await changePassword(current, next);
      // App now sees must_change_password = false and shows the dashboard.
    } catch (err) {
      // A wrong current password is reported as invalid_credentials by the server.
      if (err instanceof ApiError && err.code === "invalid_credentials") {
        setProblem({ message: "That isn't your current password.", tone: "error" });
      } else {
        setProblem(problemFrom(err, "Couldn't change your password. Try again."));
      }
      setSubmitting(false);
    }
  }

  return (
    <main className="card">
      <h1>Choose a new password</h1>
      <p className="hint">
        {auth.kind === "signedIn" ? `${auth.user.display_name}, the` : "The"} password you were
        given is only for your first sign-in. Pick one with at least {MIN_LENGTH} characters; a
        few ordinary words work well.
      </p>

      {problem && <ProblemBanner problem={problem} />}

      <form onSubmit={handleSubmit}>
        <label>
          Current password
          <input
            type="password"
            value={current}
            onChange={(e) => setCurrent(e.target.value)}
            autoComplete="current-password"
            autoFocus
            disabled={submitting}
          />
        </label>
        <label>
          New password
          <input
            type="password"
            value={next}
            onChange={(e) => setNext(e.target.value)}
            autoComplete="new-password"
            maxLength={MAX_LENGTH}
            disabled={submitting}
          />
        </label>
        {tooShort && <p className="field-error">Use at least {MIN_LENGTH} characters.</p>}
        <label>
          Type the new password again
          <input
            type="password"
            value={confirm}
            onChange={(e) => setConfirm(e.target.value)}
            autoComplete="new-password"
            maxLength={MAX_LENGTH}
            disabled={submitting}
          />
        </label>
        {mismatch && <p className="field-error">The two passwords don&apos;t match.</p>}
        <button type="submit" disabled={!canSubmit}>
          {submitting ? "Saving…" : "Save new password"}
        </button>
      </form>

      <button type="button" className="link" onClick={() => void signOut()}>
        Sign out
      </button>
    </main>
  );
}
