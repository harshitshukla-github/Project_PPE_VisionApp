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

/**
 * The server accepted the sign-in, but the very next request wasn't signed in:
 * the browser didn't keep the session cookie. On a page that isn't a secure
 * context (plain http at an address other than localhost) that is the
 * browser refusing the cookie's `Secure` flag, which is by design; elsewhere
 * it usually means cookies are blocked. It is not an expired session, and the
 * person should hear the difference.
 */
export class SessionNotKeptError extends Error {
  /** True when the page is not a secure context: the usual cause on a LAN address. */
  readonly insecure: boolean;
  /** The page's own address, such as `http://192.168.29.210:8000`. */
  readonly address: string;

  constructor(insecure: boolean, address: string) {
    super("The browser didn't keep the session cookie.");
    this.name = "SessionNotKeptError";
    this.insecure = insecure;
    this.address = address;
  }
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
  if (err instanceof SessionNotKeptError) {
    return {
      message: err.insecure
        ? `You signed in, but your browser didn't keep the session. ${err.address} isn't a secure (HTTPS) address, ` +
          "and browsers drop the sign-in cookie on such pages. Open the dashboard at an https:// address, " +
          "or at http://localhost."
        : "You signed in, but your browser didn't keep the session. Check that it allows cookies for this site, " +
          "then sign in again.",
      tone: "error",
    };
  }
  if (!(err instanceof ApiError)) return { message: fallback, tone: "error" };

  // 502 and 504 come from whatever sits in front of the server (the dev proxy, a load balancer)
  // when the server is down or slow, and mean the same as 503: not the person's doing.
  if (err.status === 502 || err.status === 503 || err.status === 504) {
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
