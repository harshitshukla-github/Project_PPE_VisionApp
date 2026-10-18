/**
 * Turns an API failure into words for a worker — docs/auth-api.md §3 says to
 * branch on the `code` field, never on the `detail` sentence.
 */
import { ApiError } from "./client";

/**
 * `error` is red: the worker should do something differently.
 * `notice` is amber: nothing is wrong with what they did, the server just
 * can't answer right now ("try again shortly", never red, never a verdict).
 */
export interface Problem {
  message: string;
  tone: "error" | "notice";
}

/** "14 minutes", "1 minute", "40 seconds" — rounded up so we never say "0 minutes". */
export function formatWait(seconds: number): string {
  if (seconds < 60) {
    const s = Math.max(1, Math.ceil(seconds));
    return `${s} second${s === 1 ? "" : "s"}`;
  }
  const m = Math.ceil(seconds / 60);
  return `${m} minute${m === 1 ? "" : "s"}`;
}

/**
 * @param fallback Shown for errors that aren't an ApiError at all.
 */
export function problemFrom(err: unknown, fallback: string): Problem {
  if (!(err instanceof ApiError)) return { message: fallback, tone: "error" };

  if (err.status === 503) {
    return {
      message:
        err.code === "auth_unavailable"
          ? "Sign-in is unavailable right now. Try again shortly."
          : "The server can't answer right now. Try again shortly.",
      tone: "notice",
    };
  }

  switch (err.code) {
    case "invalid_credentials":
      return { message: "Wrong worker ID or PIN.", tone: "error" };
    case "account_disabled":
      return {
        message: "This account is turned off. Ask your supervisor.",
        tone: "error",
      };
    case "account_locked":
      return {
        message:
          err.retryAfterS != null
            ? `Too many wrong PINs. Try again in ${formatWait(err.retryAfterS)}, or ask your supervisor to reset your PIN.`
            : "Too many wrong PINs. Ask your supervisor to reset your PIN.",
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
    default:
      return { message: err.message, tone: "error" };
  }
}
