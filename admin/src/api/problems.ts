/**
 * Turns an API failure into words for a person — docs/auth-api.md §3 says to
 * branch on `code`, never on the `detail` sentence.
 *
 * `error` is red: something the person should do differently. `notice` is
 * amber: nothing is wrong with what they did, the server just can't answer
 * right now.
 */
import { ApiError } from "./client";

export interface Problem {
  message: string;
  tone: "error" | "notice";
}

/** "14 minutes", "1 minute", "40 seconds" — rounded up so it never says "0 minutes". */
export function formatWait(seconds: number): string {
  if (seconds < 60) {
    const s = Math.max(1, Math.ceil(seconds));
    return `${s} second${s === 1 ? "" : "s"}`;
  }
  const m = Math.ceil(seconds / 60);
  return `${m} minute${m === 1 ? "" : "s"}`;
}

export function problemFrom(err: unknown, fallback: string): Problem {
  if (!(err instanceof ApiError)) return { message: fallback, tone: "error" };

  if (err.status === 503) {
    return { message: "The server can't answer right now. Try again shortly.", tone: "notice" };
  }

  switch (err.code) {
    case "invalid_credentials":
      return { message: "Wrong email or password.", tone: "error" };
    case "account_disabled":
      return { message: "This account is turned off. Ask an admin.", tone: "error" };
    case "account_locked":
      return {
        message:
          err.retryAfterS != null
            ? `Too many wrong passwords. Try again in ${formatWait(err.retryAfterS)}, or ask an admin to reset your password.`
            : "Too many wrong passwords. Ask an admin to reset your password.",
        tone: "error",
      };
    case "too_many_attempts":
      return {
        message:
          err.retryAfterS != null
            ? `Too many attempts. Wait ${formatWait(err.retryAfterS)} and try again.`
            : "Too many attempts. Wait a moment and try again.",
        tone: "error",
      };
    case "not_authenticated":
      return { message: "Your session ended. Sign in again.", tone: "error" };
    case "csrf_failed":
      return { message: "This page was out of date. Reload it and try again.", tone: "error" };
    case "forbidden":
      return { message: "You don't have access to this.", tone: "error" };
    default:
      return { message: err.message, tone: "error" };
  }
}
