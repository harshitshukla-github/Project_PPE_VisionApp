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

  constructor(
    status: number,
    message: string,
    extra: { code?: string; retryAfterS?: number; field?: string; current?: Job } = {},
  ) {
    super(message);
    this.name = "ApiError";
    this.status = status;
    this.code = extra.code;
    this.retryAfterS = extra.retryAfterS;
    this.field = extra.field;
    this.current = extra.current;
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

async function request<T>(
  method: "GET" | "POST" | "PUT" | "PATCH" | "DELETE",
  path: string,
  body?: unknown,
  retriedAfterCsrf = false,
): Promise<T> {
  const headers: Record<string, string> = {};
  if (body !== undefined) headers["Content-Type"] = "application/json";
  if (method !== "GET" && csrfToken) headers["X-CSRF-Token"] = csrfToken;

  const send = await transportFor(path);
  let response: Response;
  try {
    response = await send(`${API}${path}`, {
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
      currentUser = null;
      onSessionEnded?.();
    }
    throw error;
  }
  if (response.status === 204) return undefined as T;
  return (await response.json()) as T;
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

/** `GET /auth/me`: who is signed in, and a fresh CSRF token. Called on every page load. */
export async function getSession(): Promise<SessionResponse> {
  const session = withSite(await request<SessionResponse>("GET", "/auth/me"));
  csrfToken = session.csrf_token;
  currentUser = session.user;
  return session;
}

/** `POST /auth/login`. The session cookie is set by the server. */
export async function login(email: string, password: string): Promise<SessionResponse> {
  const session = withSite(await request<SessionResponse>("POST", "/auth/login", { email, password }));
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
