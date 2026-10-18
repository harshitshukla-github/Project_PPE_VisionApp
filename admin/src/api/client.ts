/**
 * The dashboard's one door to the backend (docs/auth-api.md §3, §6): pages
 * never call fetch themselves.
 *
 * - Same-origin: the paths are relative, so the browser sends the
 *   `ppb_session` cookie by itself. JavaScript never sees it.
 * - The CSRF token lives in memory only, never in localStorage; after a
 *   reload it comes back from `GET /auth/me`. It goes in `X-CSRF-Token` on
 *   every write.
 */

const API = "/api/v1";

export type Role = "admin" | "supervisor";

export interface DashboardUser {
  id: number;
  email: string;
  display_name: string;
  role: Role;
  must_change_password: boolean;
}

/** Body of both `POST /auth/login` and `GET /auth/me`. */
export interface SessionResponse {
  user: DashboardUser;
  csrf_token: string;
  expires_at: string;
}

/**
 * A failed request. `status` is 0 when the server couldn't be reached at all.
 * `code` is what to branch on; `message` is the server's sentence for a person.
 */
export class ApiError extends Error {
  status: number;
  code?: string;
  retryAfterS?: number;
  /** Which form field a validation error is about (docs/admin-api.md §2). */
  field?: string;

  constructor(
    status: number,
    message: string,
    extra: { code?: string; retryAfterS?: number; field?: string } = {},
  ) {
    super(message);
    this.name = "ApiError";
    this.status = status;
    this.code = extra.code;
    this.retryAfterS = extra.retryAfterS;
    this.field = extra.field;
  }
}

let csrfToken: string | null = null;
let onSessionEnded: (() => void) | null = null;

/** Called when the server says the session is gone (401 `not_authenticated`). */
export function setSessionEndedHandler(handler: (() => void) | null): void {
  onSessionEnded = handler;
}

async function toApiError(response: Response): Promise<ApiError> {
  try {
    const body = await response.json();
    let message: string;
    if (typeof body?.detail === "string") {
      message = body.detail;
    } else if (Array.isArray(body?.detail)) {
      // FastAPI's own validation errors keep their list shape (auth-api §3).
      message = body.detail.map((d: { msg?: string }) => d.msg ?? JSON.stringify(d)).join(", ");
    } else {
      message = response.statusText || `HTTP ${response.status}`;
    }
    return new ApiError(response.status, message, {
      code: typeof body?.code === "string" ? body.code : undefined,
      retryAfterS: typeof body?.retry_after_s === "number" ? body.retry_after_s : undefined,
      field: typeof body?.field === "string" ? body.field : undefined,
    });
  } catch {
    return new ApiError(response.status, response.statusText || `HTTP ${response.status}`);
  }
}

async function request<T>(
  method: "GET" | "POST" | "PUT" | "PATCH" | "DELETE",
  path: string,
  body?: unknown,
  retriedAfterCsrf = false,
): Promise<T> {
  const headers: Record<string, string> = {};
  if (body !== undefined) headers["Content-Type"] = "application/json";
  if (method !== "GET" && csrfToken) headers["X-CSRF-Token"] = csrfToken;

  let response: Response;
  try {
    response = await fetch(`${API}${path}`, {
      method,
      headers,
      body: body === undefined ? undefined : JSON.stringify(body),
      credentials: "same-origin",
    });
  } catch {
    throw new ApiError(0, "Can't reach the server.");
  }

  if (!response.ok) {
    const error = await toApiError(response);
    // A write with a stale token: fetch a fresh one and try again, once
    // (auth-api §6, admin-api §2).
    if (error.code === "csrf_failed" && !retriedAfterCsrf) {
      await getSession();
      return request<T>(method, path, body, true);
    }
    if (error.status === 401 && error.code === "not_authenticated") {
      csrfToken = null;
      onSessionEnded?.();
    }
    throw error;
  }
  if (response.status === 204) return undefined as T;
  return (await response.json()) as T;
}

/** `GET /auth/me`: who is signed in, and a fresh CSRF token. Called on every page load. */
export async function getSession(): Promise<SessionResponse> {
  const session = await request<SessionResponse>("GET", "/auth/me");
  csrfToken = session.csrf_token;
  return session;
}

/** `POST /auth/login`. The session cookie is set by the server. */
export async function login(email: string, password: string): Promise<SessionResponse> {
  const session = await request<SessionResponse>("POST", "/auth/login", { email, password });
  csrfToken = session.csrf_token;
  return session;
}

/** `POST /auth/logout`. The token is forgotten whatever the server answers. */
export async function logout(): Promise<void> {
  try {
    await request<void>("POST", "/auth/logout");
  } finally {
    csrfToken = null;
  }
}

/** `POST /auth/password`: also signs the user out of every other session. */
export function changePassword(currentPassword: string, newPassword: string): Promise<void> {
  return request<void>("POST", "/auth/password", {
    current_password: currentPassword,
    new_password: newPassword,
  });
}

/** For tests: forget everything this module remembers. */
export function resetClientState(): void {
  csrfToken = null;
  onSessionEnded = null;
}
