import {
  createContext,
  useCallback,
  useContext,
  useEffect,
  useRef,
  useState,
  type ReactNode,
} from "react";

import {
  ApiError,
  changePassword as apiChangePassword,
  getSession,
  login,
  logout,
  setSessionEndedHandler,
  type DashboardUser,
} from "../api/client";
import { problemFrom, SessionNotKeptError, type Problem } from "../api/problems";

/**
 * Who is using the dashboard. `loading` lasts only until `GET /auth/me` has
 * answered on page load (auth-api §6); the app shows nothing but a spinner
 * until then, so a signed-in user never sees a flash of the login page.
 *
 * `unavailable` is for when the server didn't answer that question at all
 * (it is down, or something in front of it is). That says nothing about
 * whether the person is signed in, so the app doesn't show a sign-in form
 * they couldn't use; it says so and offers to try again.
 */
export type AuthState =
  | { kind: "loading" }
  | { kind: "unavailable"; problem: Problem }
  | { kind: "signedOut"; notice?: Problem }
  | { kind: "signedIn"; user: DashboardUser };

interface AuthApi {
  auth: AuthState;
  /** Rejects with the ApiError so the login page can say why. */
  signIn: (email: string, password: string) => Promise<void>;
  signOut: () => Promise<void>;
  /** Rejects with the ApiError (e.g. `weak_password`); on success the user may continue. */
  changePassword: (currentPassword: string, newPassword: string) => Promise<void>;
  /** Ask the server who is signed in again, after it didn't answer. */
  retry: () => void;
}

const AuthContext = createContext<AuthApi | undefined>(undefined);

const SESSION_ENDED: Problem = { message: "Your session ended. Sign in again.", tone: "error" };

export function AuthProvider({ children }: { children: ReactNode }) {
  const [auth, setAuth] = useState<AuthState>({ kind: "loading" });
  // Only the latest check may change the state; an older one still in flight is ignored.
  const attempt = useRef(0);

  const check = useCallback(() => {
    const mine = ++attempt.current;
    getSession().then(
      (session) => {
        if (attempt.current === mine) setAuth({ kind: "signedIn", user: session.user });
      },
      (err: unknown) => {
        if (attempt.current !== mine) return;
        if (err instanceof ApiError && err.status === 401) {
          setAuth({ kind: "signedOut" }); // just "not signed in yet": nothing to explain
        } else {
          setAuth({ kind: "unavailable", problem: problemFrom(err, "Couldn't reach the server.") });
        }
      },
    );
  }, []);

  useEffect(() => {
    // Any request answering 401 not_authenticated (expired, revoked, or idle
    // for 2 hours) sends the user back to the login page.
    setSessionEndedHandler(() => setAuth({ kind: "signedOut", notice: SESSION_ENDED }));
    check();
    return () => {
      attempt.current++; // whatever is still in flight no longer counts
      setSessionEndedHandler(null);
    };
  }, [check]);

  const retry = useCallback(() => {
    setAuth({ kind: "loading" });
    check();
  }, [check]);

  async function signIn(email: string, password: string) {
    const session = await login(email.trim(), password);
    // The login answer sets the session cookie, but the browser decides whether to keep it: on plain
    // http at a LAN address it refuses a `Secure` cookie. Ask who we are once, so that case is told
    // here, on the sign-in page, instead of as "Your session ended" after the dashboard has opened.
    // Only a 401 means the cookie didn't come back; a server hiccup doesn't undo a sign-in the server accepted.
    try {
      await getSession();
    } catch (err) {
      if (err instanceof ApiError && err.status === 401) {
        throw new SessionNotKeptError(window.isSecureContext === false, window.location.origin);
      }
    }
    setAuth({ kind: "signedIn", user: session.user });
  }

  async function signOut() {
    try {
      await logout();
    } catch {
      // Offline or already expired: signing out locally is what matters.
    }
    setAuth({ kind: "signedOut" });
  }

  async function changePassword(currentPassword: string, newPassword: string) {
    await apiChangePassword(currentPassword, newPassword);
    setAuth((current) =>
      current.kind === "signedIn"
        ? { kind: "signedIn", user: { ...current.user, must_change_password: false } }
        : current,
    );
  }

  return (
    <AuthContext.Provider value={{ auth, signIn, signOut, changePassword, retry }}>
      {children}
    </AuthContext.Provider>
  );
}

export function useAuth(): AuthApi {
  const ctx = useContext(AuthContext);
  if (!ctx) throw new Error("useAuth must be used within an AuthProvider");
  return ctx;
}
