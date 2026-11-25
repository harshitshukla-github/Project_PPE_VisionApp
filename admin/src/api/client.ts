/**
 * The dashboard's one door to the backend (docs/auth-api.md §3, §6;
 * docs/admin-api.md): pages never call fetch themselves.
 *
 * - Same-origin: the paths are relative, so the browser sends the
 *   `ppb_session` cookie by itself. JavaScript never sees it.
 * - The CSRF token lives in memory only, never in localStorage; after a
 *   reload it comes back from `GET /auth/me`. It goes in `X-CSRF-Token` on
 *   every write.
 * - Login and sign-in checks (`/auth/*`) always go to the real server. The
 *   admin endpoints (`/admin/*`) go to the real server too, unless the mock
 *   backend is switched on (see `mockAdminApi` below), which is the one place
 *   to change when the Phase 3 backend exists.
 */
import type {
  AdminUser,
  AuditEntry,
  AuditQuery,
  Item,
  Job,
  JobResult,
  Page,
  Requirement,
  Role,
  Setting,
  SettingsPatch,
  SettingsResult,
  Site,
  Worker,
  WorkerQuery,
} from "./types";

export type { Role } from "./types";

const API = "/api/v1";

export interface DashboardUser {
  id: number;
  email: string;
  display_name: string;
  role: Role;
  /** The supervisor's site key; null for an admin (auth-api §6). */
  site: string | null;
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
  /** Which form field a validation error is about, as a path (docs/admin-api.md §2). */
  field?: string;
  /** On `409 version_conflict`: the job as it is now. */
  current?: Job;
  /**
   * A write that may have been carried out although no proper answer came back
   * (it timed out, or a proxy answered `502`/`504`). The message says so, and
   * the person should look before doing it again.
   */
  unsure: boolean;

  constructor(
    status: number,
    message: string,
    extra: { code?: string; retryAfterS?: number; field?: string; current?: Job; unsure?: boolean } = {},
  ) {
    super(message);
    this.name = "ApiError";
    this.status = status;
    this.code = extra.code;
    this.retryAfterS = extra.retryAfterS;
    this.field = extra.field;
    this.current = extra.current;
    this.unsure = extra.unsure ?? false;
  }
}

// ---- the mock switch -------------------------------------------------------

type FetchLike = (url: string, init?: RequestInit) => Promise<Response>;

/**
 * Is the `/admin/*` API faked in the browser? Only when `VITE_ADMIN_API=mock`
 * says so: by default the dashboard talks to the real server. The stand-in is
 * for working without a backend, and for seeing a supervisor's view without a
 * supervisor account.
 */
const MOCK_ADMIN_API = import.meta.env.VITE_ADMIN_API === "mock";

let adminTransport: FetchLike | undefined;
let mockTransport: Promise<FetchLike> | undefined;

/** Is the admin API being faked (the stand-in, or a test's transport)? The shell shows a banner while it is. */
export function adminApiIsMocked(): boolean {
  return MOCK_ADMIN_API || (adminTransport !== undefined && adminTransport !== realFetch);
}

/** For tests: send `/admin/*` calls here instead (null = the real fetch). */
export function setAdminTransport(transport: FetchLike | null | undefined): void {
  adminTransport = transport === null ? realFetch : transport;
}

const realFetch: FetchLike = (url, init) => fetch(url, init);

async function transportFor(path: string): Promise<FetchLike> {
  if (!path.startsWith("/admin/")) return realFetch;
  if (adminTransport) return adminTransport;
  if (MOCK_ADMIN_API) {
    // Loaded on demand, so a production build never ships the mock.
    mockTransport ??= import("../mock/backend").then(
      (m) =>
        m.createMockBackend({
          currentUser: () => currentUser,
          delayMs: import.meta.env.MODE === "development" ? 150 : 0,
        }).fetch,
    );
    return mockTransport;
  }
  return realFetch;
}

// ---- requests --------------------------------------------------------------

let csrfToken: string | null = null;
let currentUser: DashboardUser | null = null;
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
      current: body?.current && typeof body.current === "object" ? body.current : undefined,
    });
  } catch {
    return new ApiError(response.status, response.statusText || `HTTP ${response.status}`);
  }
}

interface RequestOptions {
  /** Set on the one retry after a `csrf_failed`, so it can't loop. */
  retriedAfterCsrf?: boolean;
  /**
   * Report a `401 not_authenticated` to the caller only, not to the
   * session-ended handler. For a caller that is asking whether there is a
   * session at all, where "no" is an answer and not a session ending.
   */
  quiet?: boolean;
  /**
   * A non-GET that changes nothing worth checking on, so it is safe to send
   * again after a timeout or a `502`/`504`: signing in. Its failures say "try
   * again", not "may or may not have been saved".
   */
  repeatable?: boolean;
}

async function request<T>(
  method: "GET" | "POST" | "PUT" | "PATCH" | "DELETE",
  path: string,
  body?: unknown,
  options: RequestOptions = {},
): Promise<T> {
  const headers: Record<string, string> = {};
  if (body !== undefined) headers["Content-Type"] = "application/json";
  if (method !== "GET" && csrfToken) headers["X-CSRF-Token"] = csrfToken;

  const send = await transportFor(path);
  // Whether a failure without a proper answer could have left a change behind.
  const mayHaveChanged = method !== "GET" && !options.repeatable;
  // A server (or proxy) that takes the request and never answers would otherwise leave the page
  // on "Loading…" or "Saving…" for ever. The timer covers reading the answer too.
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), REQUEST_TIMEOUT_MS);
  let error: ApiError | undefined;
  let result: T | undefined;
  try {
    let response: Response;
    try {
      response = await send(`${API}${path}`, {
        method,
        headers,
        body: body === undefined ? undefined : JSON.stringify(body),
        credentials: "same-origin",
        signal: controller.signal,
      });
    } catch {
      throw new ApiError(0, "Can't reach the server.");
    }
    if (!response.ok) error = maybeDone(mayHaveChanged, await toApiError(response));
    else if (response.status !== 204) result = (await response.json()) as T;
  } catch (err) {
    // Anything that went wrong because the timer ran out is the timeout, whatever was thrown.
    // Otherwise it is passed on as it is: an answer that isn't JSON stays a SyntaxError.
    if (controller.signal.aborted) throw timedOut(mayHaveChanged);
    throw err;
  } finally {
    clearTimeout(timer);
  }

  if (error) {
    // A write with a stale token: fetch a fresh one and try again, once
    // (auth-api §6, admin-api §2).
    if (error.code === "csrf_failed" && !options.retriedAfterCsrf) {
      await getSession();
      return request<T>(method, path, body, { ...options, retriedAfterCsrf: true });
    }
    if (error.status === 401 && error.code === "not_authenticated") {
      csrfToken = null;
      currentUser = null;
      if (!options.quiet) onSessionEnded?.();
    }
    throw error;
  }
  return result as T;
}

/**
 * How long a request may take before the dashboard stops waiting. Every admin
 * call is a small database read or write, so this is generous; it only exists
 * so a hung server ends in a message and a Try again, not a spinner.
 */
export const REQUEST_TIMEOUT_MS = 20_000;

/**
 * Status 0, like no answer at all (`isOutage`). A read, or a sign-in, can
 * simply be tried again. A write may have been carried out before the server
 * went quiet, so the person is told to look before doing it again.
 */
function timedOut(mayHaveChanged: boolean): ApiError {
  return !mayHaveChanged
    ? new ApiError(0, "The server didn't answer in time.")
    : new ApiError(
        0,
        "The server didn't answer in time, so this may or may not have been saved. Reload the page to check.",
        { unsure: true },
      );
}

/**
 * A `502` or `504` is the proxy in front giving up on the server, which may
 * still have done the work. On a write, say so: "try again" could repeat a
 * change that went through, and a repeated password change fails on the old
 * password. A `503` is our own server refusing, so nothing was done, and it is
 * left as it is; so are reads and sign-in, which change nothing to check on.
 */
function maybeDone(mayHaveChanged: boolean, error: ApiError): ApiError {
  if (!mayHaveChanged || (error.status !== 502 && error.status !== 504)) return error;
  return new ApiError(
    error.status,
    "The server didn't answer properly, so this may or may not have been saved. Reload the page to check.",
    { code: error.code, unsure: true },
  );
}

/** `?a=1&b=x`, leaving out anything undefined, null or empty. */
function query(params: Record<string, string | number | boolean | null | undefined>): string {
  const search = new URLSearchParams();
  for (const [key, value] of Object.entries(params)) {
    if (value === undefined || value === null || value === "") continue;
    search.set(key, String(value));
  }
  const text = search.toString();
  return text ? `?${text}` : "";
}

// ---- sign-in (auth-api §6) -------------------------------------------------

/** An older backend answers without `site`; treat that as "no site". */
function withSite(session: SessionResponse): SessionResponse {
  return { ...session, user: { ...session.user, site: session.user.site ?? null } };
}

/**
 * `GET /auth/me`: who is signed in, and a fresh CSRF token. Called on every page load.
 *
 * With `quiet`, a `401` is only thrown to the caller; the session-ended
 * handler isn't told. The check right after a sign-in uses it: a `401` there
 * means the browser didn't keep the cookie, which the sign-in page explains
 * itself, and "Your session ended" would be the wrong thing to say.
 */
export async function getSession(options: { quiet?: boolean } = {}): Promise<SessionResponse> {
  const session = withSite(
    await request<SessionResponse>("GET", "/auth/me", undefined, { quiet: options.quiet }),
  );
  csrfToken = session.csrf_token;
  currentUser = session.user;
  return session;
}

/** `POST /auth/login`. The session cookie is set by the server. */
export async function login(email: string, password: string): Promise<SessionResponse> {
  // Signing in saves nothing, so a timeout or a 502/504 here is a plain "try again".
  const session = withSite(
    await request<SessionResponse>("POST", "/auth/login", { email, password }, { repeatable: true }),
  );
  csrfToken = session.csrf_token;
  currentUser = session.user;
  return session;
}

/** `POST /auth/logout`. The token is forgotten whatever the server answers. */
export async function logout(): Promise<void> {
  try {
    await request<void>("POST", "/auth/logout");
  } finally {
    csrfToken = null;
    currentUser = null;
  }
}

/** `POST /auth/password`: also signs the user out of every other session. */
export function changePassword(currentPassword: string, newPassword: string): Promise<void> {
  return request<void>("POST", "/auth/password", {
    current_password: currentPassword,
    new_password: newPassword,
  });
}

// ---- admin API (docs/admin-api.md) -----------------------------------------

/** §4 */
export const getItems = () => request<Item[]>("GET", "/admin/items");

/** §5 */
export const getJobs = () => request<Job[]>("GET", "/admin/jobs");

export const createJob = (job: { key: string; display_name: string; requirements: Requirement[] }) =>
  request<JobResult>("POST", "/admin/jobs", job);

export const updateJob = (
  key: string,
  patch: { version: number; display_name?: string; is_active?: boolean },
) => request<{ job: Job; applies_within_s: number }>("PATCH", `/admin/jobs/${key}`, patch);

/** Replaces the job's whole equipment list: a matrix row is saved as a unit. */
export const putRequirements = (key: string, version: number, requirements: Requirement[]) =>
  request<JobResult>("PUT", `/admin/jobs/${key}/requirements`, { version, requirements });

/** Sets the display order; every job, active or not, exactly once. Versions don't change. */
export const putJobOrder = (keys: string[]) => request<Job[]>("PUT", "/admin/job-order", { keys });

/** §3 */
export const getSites = () => request<Site[]>("GET", "/admin/sites");

export const createSite = (site: { key: string; display_name: string }) =>
  request<{ site: Site }>("POST", "/admin/sites", site);

export const updateSite = (key: string, patch: { display_name?: string; is_active?: boolean }) =>
  request<{ site: Site }>("PATCH", `/admin/sites/${key}`, patch);

/** §6 */
export const getSettings = () => request<Setting[]>("GET", "/admin/settings");

export const patchSettings = (patch: SettingsPatch) =>
  request<SettingsResult>("PATCH", "/admin/settings", patch);

/** §7 */
export const getUsers = () => request<AdminUser[]>("GET", "/admin/users");

export const createUser = (user: { email: string; display_name: string; role: Role; site?: string | null }) =>
  request<{ user: AdminUser; temporary_password: string }>("POST", "/admin/users", user);

export const updateUser = (
  id: number,
  patch: { display_name?: string; role?: Role; site?: string | null; is_active?: boolean },
) => request<{ user: AdminUser }>("PATCH", `/admin/users/${id}`, patch);

export const resetUserPassword = (id: number) =>
  request<{ temporary_password: string }>("POST", `/admin/users/${id}/reset-password`);

/** §8 */
export const getWorkers = (q: WorkerQuery = {}) =>
  request<Page<Worker>>(
    "GET",
    `/admin/workers${query({
      search: q.search,
      site: q.site,
      active: q.active,
      locked: q.locked,
      limit: q.limit,
      cursor: q.cursor,
    })}`,
  );

export const createWorker = (worker: { worker_id: string; display_name: string; site?: string }) =>
  request<{ worker: Worker; one_time_pin: string }>("POST", "/admin/workers", worker);

export const updateWorker = (
  workerId: string,
  patch: { display_name?: string; site?: string; is_active?: boolean },
) => request<{ worker: Worker }>("PATCH", `/admin/workers/${workerId}`, patch);

export const resetWorkerPin = (workerId: string) =>
  request<{ one_time_pin: string }>("POST", `/admin/workers/${workerId}/reset-pin`);

/** §9 */
export const getAudit = (q: AuditQuery = {}) =>
  request<Page<AuditEntry>>(
    "GET",
    `/admin/audit${query({
      action: q.action,
      target: q.target,
      actor: q.actor,
      site: q.site,
      from: q.from,
      to: q.to,
      limit: q.limit,
      cursor: q.cursor,
    })}`,
  );

/** For tests: forget everything this module remembers. */
export function resetClientState(): void {
  csrfToken = null;
  currentUser = null;
  onSessionEnded = null;
  adminTransport = undefined;
}
