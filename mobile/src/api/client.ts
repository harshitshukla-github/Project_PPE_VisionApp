/**
 * Single API client module — per INSTRUCTIONS.md §7, screens never call
 * fetch directly; every backend call goes through here.
 */
import AsyncStorage from "@react-native-async-storage/async-storage";

import type {
  ComplianceCheckResponse,
  HealthResponse,
  JobsResponse,
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

/** Returns the worker's remembered ID, or null if none has been saved yet (Home screen, Phase 2). */
export async function getWorkerId(): Promise<string | null> {
  return AsyncStorage.getItem(WORKER_ID_KEY);
}

/** Persists the worker ID, trimmed, so it's remembered across launches. */
export async function setWorkerId(id: string): Promise<void> {
  await AsyncStorage.setItem(WORKER_ID_KEY, id.trim());
}

/** Thrown by every client function on a failed request. `status` is 0 for client-side/network failures. */
export class ApiError extends Error {
  status: number;

  constructor(status: number, message: string) {
    super(message);
    this.name = "ApiError";
    this.status = status;
  }
}

async function apiUrl(path: string): Promise<string> {
  const base = await getServerUrl();
  if (!base) {
    throw new ApiError(0, "No server URL configured. Set it in Settings.");
  }
  return `${base}/api/v1${path}`;
}

async function extractErrorMessage(response: Response): Promise<string> {
  try {
    const body = await response.json();
    if (typeof body?.detail === "string") return body.detail;
    if (Array.isArray(body?.detail)) {
      return body.detail
        .map((d: { msg?: string }) => d.msg ?? JSON.stringify(d))
        .join(", ");
    }
    return JSON.stringify(body);
  } catch {
    return response.statusText || `HTTP ${response.status}`;
  }
}

async function requestJson<T>(path: string, init?: RequestInit): Promise<T> {
  // Let a missing-server-URL ApiError propagate as-is (distinct message).
  const url = await apiUrl(path);

  let response: Response;
  try {
    response = await fetch(url, init);
  } catch {
    throw new ApiError(0, "Can't reach server — check the URL in Settings.");
  }

  if (!response.ok) {
    throw new ApiError(response.status, await extractErrorMessage(response));
  }
  return response.json() as Promise<T>;
}

/** GET /api/v1/health — used for the Settings screen connection indicator. */
export function getHealth(): Promise<HealthResponse> {
  return requestJson<HealthResponse>("/health");
}

/** GET /api/v1/jobs — populates the job dropdown. Never hardcode job categories. */
export function getJobs(): Promise<JobsResponse> {
  return requestJson<JobsResponse>("/jobs");
}

export interface ComplianceCheckParams {
  /** Local file URI from expo-camera / expo-image-picker. */
  imageUri: string;
  workerId: string;
  /** Must be a key from getJobs(). */
  jobCategory: string;
}

/**
 * POST /api/v1/compliance/check — the core request. Resolves to PASS, FAIL,
 * or NEEDS_REVIEW; the backend may spend several seconds on a Gemini second
 * opinion before answering, and there is deliberately no client-side timeout.
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
  formData.append("worker_id", params.workerId);
  formData.append("job_category", params.jobCategory);

  return requestJson<ComplianceCheckResponse>("/compliance/check", {
    method: "POST",
    body: formData,
  });
}
