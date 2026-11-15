import { useState, type ChangeEvent, type FormEvent } from "react";

import { ApiError } from "../api/client";
import { problemFrom, type Problem } from "../api/problems";
import { useAuth } from "../auth/AuthContext";
import { ProblemBanner } from "../components/ProblemBanner";
import { DEFAULT_PATH } from "./registry";

// auth-api §6: 12–128 characters. The server also refuses a password equal to
// the email or to the current one, and says why (422 weak_password).
const MIN_LENGTH = 12;
const MAX_LENGTH = 128;

/**
 * Changing the signed-in user's password (`POST /auth/password`), in two ways.
 *
 * - **Forced** (the default): shown while the user still has the temporary
 *   password an admin gave them (`must_change_password`). Nothing else works
 *   until it is replaced, so the only other way out is signing out. Once it
 *   is changed the app opens the dashboard by itself.
 * - **Voluntary** (`voluntary`): opened from "Change password" in the header,
 *   inside the usual frame, so the navigation is the way back. It says when
 *   the change is done instead of moving on, and it can be cancelled.
 *
 * An error from the server stays only until the person edits a field or tries
 * again, so it never sits next to a message about the new input.
 */
export function ChangePasswordPage({ voluntary = false }: { voluntary?: boolean }) {
  const { auth, changePassword, signOut } = useAuth();
  const [current, setCurrent] = useState("");
  const [next, setNext] = useState("");
  const [confirm, setConfirm] = useState("");
  const [submitting, setSubmitting] = useState(false);
  const [problem, setProblem] = useState<Problem | null>(null);
  const [done, setDone] = useState(false);

  const mismatch = confirm !== "" && confirm !== next;
  const tooShort = next !== "" && next.length < MIN_LENGTH;
  const canSubmit =
    current !== "" &&
    next.length >= MIN_LENGTH &&
    next.length <= MAX_LENGTH &&
    confirm === next &&
    !submitting;

  /** A field's onChange: store the text, and drop the last server error, which was about the old text. */
  const edit = (setter: (text: string) => void) => (event: ChangeEvent<HTMLInputElement>) => {
    setProblem(null);
    setter(event.target.value);
  };

  async function handleSubmit(event: FormEvent) {
    event.preventDefault();
    if (!canSubmit) return;
    setSubmitting(true);
    setProblem(null);
    try {
      await changePassword(current, next);
      // Forced: the app now sees must_change_password = false and shows the dashboard.
      if (voluntary) {
        setCurrent("");
        setNext("");
        setConfirm("");
        setDone(true);
      }
    } catch (err) {
      // A wrong current password is reported as invalid_credentials by the server.
      if (err instanceof ApiError && err.code === "invalid_credentials") {
        setProblem({ message: "That isn't your current password.", tone: "error" });
      } else {
        setProblem(problemFrom(err, "Couldn't change your password. Try again."));
      }
    } finally {
      setSubmitting(false);
    }
  }

  if (voluntary && done) {
    return (
      <main className="card" data-testid="password-changed">
        <h1>Password changed</h1>
        <p>You stay signed in here. Anywhere else you were signed in has been signed out.</p>
        <p>
          <a href={`#${DEFAULT_PATH}`}>Back to the dashboard</a>
        </p>
      </main>
    );
  }

  return (
    <main className="card">
      <h1>{voluntary ? "Change your password" : "Choose a new password"}</h1>
      <p className="hint">
        {voluntary ? (
          <>
            Pick a new password with at least {MIN_LENGTH} characters; a few ordinary words work well. You stay
            signed in here, and anywhere else you are signed in is signed out.
          </>
        ) : (
          <>
            {auth.kind === "signedIn" ? `${auth.user.display_name}, the` : "The"} password you were given is
            only for your first sign-in. Pick one with at least {MIN_LENGTH} characters; a few ordinary words
            work well.
          </>
        )}
      </p>

      {problem && <ProblemBanner problem={problem} />}

      <form onSubmit={handleSubmit}>
        <label>
          Current password
          <input
            type="password"
            value={current}
            onChange={edit(setCurrent)}
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
            onChange={edit(setNext)}
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
            onChange={edit(setConfirm)}
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

      {voluntary ? (
        <p>
          <a href={`#${DEFAULT_PATH}`}>Cancel</a>
        </p>
      ) : (
        <button type="button" className="link" onClick={() => void signOut()}>
          Sign out
        </button>
      )}
    </main>
  );
}
