/**
 * Single API client module — per INSTRUCTIONS.md §7, screens never call
 * fetch directly; every backend call goes through here.
 */
import AsyncStorage from "@react-native-async-storage/async-storage";
import * as SecureStore from "expo-secure-store";

import type {
  ComplianceCheckResponse,
  HealthResponse,
  JobsResponse,
  WorkerLoginResponse,
  WorkerMeResponse,
} from "./types";

const SERVER_URL_KEY = "@ppb/server_url";

/**
 * No sensible default — every dev network is different (INSTRUCTIONS.md §5:
 * localhost on the phone is the phone itself). Returns null until the
 * Settings screen (Phase 1) sets one.
 */
export async function getServerUrl(): Promise<string | null> {
  return AsyncStorage.getItem(SERVER_URL_KEY);
}

/** Persists the server base URL (e.g. "http://192.168.1.5:8000"), trimmed and without a trailing slash. */
export async function setServerUrl(url: string): Promise<void> {
  await AsyncStorage.setItem(SERVER_URL_KEY, url.trim().replace(/\/+$/, ""));
}

const WORKER_ID_KEY = "@ppb/worker_id";

/** The last worker ID that signed in on this phone — only pre-fills the login form. */
export async function getWorkerId(): Promise<string | null> {
  return AsyncStorage.getItem(WORKER_ID_KEY);
}

/** Remembers the worker ID, trimmed, so the login form can pre-fill it. */
export async function setWorkerId(id: string): Promise<void> {
  await AsyncStorage.setItem(WORKER_ID_KEY, id.trim());
}

// docs/auth-api.md §8: the session token lives in expo-secure-store, never AsyncStorage.
const TOKEN_KEY = "ppb_worker_token";

/** undefined = not read from secure storage yet; null = known to be absent. */
let cachedToken: string | null | undefined;

/** The worker's session token, or null when signed out. */
export async function getToken(): Promise<string | null> {
  if (cachedToken !== undefined) return cachedToken;
  try {
    cachedToken = await SecureStore.getItemAsync(TOKEN_KEY);
  } catch {
    cachedToken = null;
  }
  return cachedToken;
}

/**
 * Keeps the token in secure storage. If storage fails the token still lives
 * in memory, so this session works but won't survive an app restart.
 */
export async function setToken(token: string): Promise<void> {
  cachedToken = token;
  try {
    await SecureStore.setItemAsync(TOKEN_KEY, token);
  } catch {
    // Memory copy above is enough for this session.
  }
}

/** Forgets the token. Always clears the in-memory copy, whatever storage does. */
export async function clearToken(): Promise<void> {
  cachedToken = null;
  try {
    await SecureStore.deleteItemAsync(TOKEN_KEY);
  } catch {
    // Nothing more to do — the in-memory copy is already gone.
  }
}

/**
 * Thrown by every client function on a failed request. `status` is 0 for
 * client-side/network failures. `code` and `retryAfterS` come from the
 * backend's error body (docs/auth-api.md §3) when it sent them.
 */
export class ApiError extends Error {
  status: number;
  code?: string;
  retryAfterS?: number;

  constructor(status: number, message: string, code?: string, retryAfterS?: number) {
    super(message);
    this.name = "ApiError";
    this.status = status;
    this.code = code;
    this.retryAfterS = retryAfterS;
  }
}

async function apiUrl(path: string): Promise<string> {
  const base = await getServerUrl();
  if (!base) {
    throw new ApiError(0, "No server URL configured. Set it in Settings.");
  }
  return `${base}/api/v1${path}`;
}

async function toApiError(response: Response): Promise<ApiError> {
  try {
    const body = await response.json();
    const code = typeof body?.code === "string" ? body.code : undefined;
    const retryAfterS =
      typeof body?.retry_after_s === "number" ? body.retry_after_s : undefined;
    let message: string;
    if (typeof body?.detail === "string") {
      message = body.detail;
    } else if (Array.isArray(body?.detail)) {
      message = body.detail
        .map((d: { msg?: string }) => d.msg ?? JSON.stringify(d))
        .join(", ");
    } else {
      message = JSON.stringify(body);
    }
    return new ApiError(response.status, message, code, retryAfterS);
  } catch {
    return new ApiError(
      response.status,
      response.statusText || `HTTP ${response.status}`,
    );
  }
}

async function requestJson<T>(path: string, init?: RequestInit): Promise<T> {
  // Let a missing-server-URL ApiError propagate as-is (distinct message).
  const url = await apiUrl(path);

  // Send the session token on every call once signed in. Endpoints that don't
  // need it ignore it, and the server never falls back to a form field when a
  // token was sent (docs/auth-api.md §5).
  const token = await getToken();
  const finalInit: RequestInit | undefined = token
    ? { ...init, headers: { ...(init?.headers as Record<string, string>), Authorization: `Bearer ${token}` } }
    : init;

  let response: Response;
  try {
    response = await fetch(url, finalInit);
  } catch {
    throw new ApiError(0, "Can't reach server — check the URL in Settings.");
  }

  if (!response.ok) {
    throw await toApiError(response);
  }
  if (response.status === 204) {
    return undefined as T;
  }
  return response.json() as Promise<T>;
}

function postJson<T>(path: string, body: unknown): Promise<T> {
  return requestJson<T>(path, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
}

/** GET /api/v1/health — used for the Settings screen connection indicator. */
export function getHealth(): Promise<HealthResponse> {
  return requestJson<HealthResponse>("/health");
}

/** GET /api/v1/jobs — populates the job dropdown. Never hardcode job categories. */
export function getJobs(): Promise<JobsResponse> {
  return requestJson<JobsResponse>("/jobs");
}

/** POST /api/v1/auth/worker/login. The caller stores the returned token with setToken(). */
export function workerLogin(workerId: string, pin: string): Promise<WorkerLoginResponse> {
  return postJson<WorkerLoginResponse>("/auth/worker/login", {
    worker_id: workerId.trim(),
    pin,
  });
}

/** GET /api/v1/auth/worker/me — is the stored token still good, and does the PIN need changing? */
export function getWorkerMe(): Promise<WorkerMeResponse> {
  return requestJson<WorkerMeResponse>("/auth/worker/me");
}

/** POST /api/v1/auth/worker/pin — swaps the issued PIN for one the worker chose. */
export function changeWorkerPin(currentPin: string, newPin: string): Promise<void> {
  return postJson<void>("/auth/worker/pin", {
    current_pin: currentPin,
    new_pin: newPin,
  });
}

/**
 * POST /api/v1/auth/worker/logout, then forget the token — whatever the
 * server said, so a failed request can never leave the worker signed in.
 */
export async function workerLogout(): Promise<void> {
  try {
    await requestJson<void>("/auth/worker/logout", { method: "POST" });
  } catch {
    // Offline or already expired: the local sign-out below still happens.
  } finally {
    await clearToken();
  }
}

export interface ComplianceCheckParams {
  /** Local file URI from expo-camera / expo-image-picker. */
  imageUri: string;
  /** Must be a key from getJobs(). */
  jobCategory: string;
}

/**
 * POST /api/v1/compliance/check — the core request. Resolves to PASS, FAIL,
 * or NEEDS_REVIEW; the backend may spend several seconds on a Gemini second
 * opinion before answering, and there is deliberately no client-side timeout.
 *
 * The worker is identified by the session token, not a form field.
 */
export async function postComplianceCheck(
  params: ComplianceCheckParams,
): Promise<ComplianceCheckResponse> {
  const formData = new FormData();
  // React Native FormData file shape — not a real Blob, hence the cast.
  formData.append("image", {
    uri: params.imageUri,
    name: "photo.jpg",
    type: "image/jpeg",
  } as unknown as Blob);
  formData.append("job_category", params.jobCategory);

  return requestJson<ComplianceCheckResponse>("/compliance/check", {
    method: "POST",
    body: formData,
  });
}
