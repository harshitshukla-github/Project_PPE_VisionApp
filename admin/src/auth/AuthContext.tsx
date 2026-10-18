import {
  createContext,
  useContext,
  useEffect,
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
import { problemFrom, type Problem } from "../api/problems";

/**
 * Who is using the dashboard. `loading` lasts only until `GET /auth/me` has
 * answered on page load (auth-api §6); the app shows nothing but a spinner
 * until then, so a signed-in user never sees a flash of the login page.
 */
export type AuthState =
  | { kind: "loading" }
  | { kind: "signedOut"; notice?: Problem }
  | { kind: "signedIn"; user: DashboardUser };

interface AuthApi {
  auth: AuthState;
  /** Rejects with the ApiError so the login page can say why. */
  signIn: (email: string, password: string) => Promise<void>;
  signOut: () => Promise<void>;
  /** Rejects with the ApiError (e.g. `weak_password`); on success the user may continue. */
  changePassword: (currentPassword: string, newPassword: string) => Promise<void>;
}

const AuthContext = createContext<AuthApi | undefined>(undefined);

const SESSION_ENDED: Problem = { message: "Your session ended. Sign in again.", tone: "error" };

export function AuthProvider({ children }: { children: ReactNode }) {
  const [auth, setAuth] = useState<AuthState>({ kind: "loading" });

  useEffect(() => {
    let cancelled = false;

    // Any request answering 401 not_authenticated (expired, revoked, or idle
    // for 2 hours) sends the user back to the login page.
    setSessionEndedHandler(() => setAuth({ kind: "signedOut", notice: SESSION_ENDED }));

    getSession().then(
      (session) => {
        if (!cancelled) setAuth({ kind: "signedIn", user: session.user });
      },
      (err: unknown) => {
        if (cancelled) return;
        // 401 just means "not signed in yet"; anything else (server down) is worth saying.
        const quiet = err instanceof ApiError && err.status === 401;
        setAuth({
          kind: "signedOut",
          notice: quiet ? undefined : problemFrom(err, "Couldn't reach the server."),
        });
      },
    );

    return () => {
      cancelled = true;
      setSessionEndedHandler(null);
    };
  }, []);

  async function signIn(email: string, password: string) {
    const session = await login(email.trim(), password);
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
    <AuthContext.Provider value={{ auth, signIn, signOut, changePassword }}>
      {children}
    </AuthContext.Provider>
  );
}

export function useAuth(): AuthApi {
  const ctx = useContext(AuthContext);
  if (!ctx) throw new Error("useAuth must be used within an AuthProvider");
  return ctx;
}
