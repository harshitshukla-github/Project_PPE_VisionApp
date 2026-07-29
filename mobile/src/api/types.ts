/**
 * API response/request types for the PPB compliance backend.
 *
 * Mirrors src/inference/schemas.py and src/api/routes.py in the repo root
 * (NOT the simplified example in INSTRUCTIONS.md §4 — verified directly
 * against source on 2026-07-29, since two fields there are stale/wrong,
 * see the GET /jobs note below).
 */

export type Verdict = "PASS" | "FAIL";

export interface Detection {
  /** PPE class name, e.g. "hardhat", "no_hardhat". */
  class: string;
  /** 0..1 */
  confidence: number;
  /** [x1, y1, x2, y2] in pixel coordinates of the uploaded image. */
  bbox: [number, number, number, number];
}

/** Response body for POST /api/v1/compliance/check. */
export interface ComplianceCheckResponse {
  worker_id: string;
  job_category: string;
  /** ISO 8601 UTC timestamp. */
  timestamp: string;
  verdict: Verdict;
  /** Mean confidence across all detections (0 if none). */
  confidence_avg: number;
  detections: Detection[];
  /** Required PPE classes not detected on the worker. */
  missing_equipment: string[];
  /** Explicit violation classes detected (e.g. "no_hardhat"). */
  violations_detected: string[];
  /**
   * PPE required by the job but not yet detectable by the model.
   * Render as amber "verify manually" — never as a failure.
   */
  unverifiable_equipment: string[];
  /** Gemini-generated explanation. Present ONLY on FAIL, else null. */
  explanation: string | null;
  model_used: string;
  inference_time_ms: number;
  gemini_invoked: boolean;
}

/** Response body for GET /api/v1/health. */
export interface HealthResponse {
  status: string;
  model_loaded: boolean;
  model_name: string;
  device: string;
}

/** One job category's PPE requirements, as returned by GET /api/v1/jobs. */
export interface JobRequirements {
  /** Classes that must be detected for a PASS. */
  required: string[];
  /**
   * Classes the job needs but the model can't verify yet — surfaces as
   * `unverifiable_equipment` in a compliance check response.
   */
  pending: string[];
}

/**
 * Response body for GET /api/v1/jobs.
 *
 * IMPORTANT: this is a flat map keyed by job_category — e.g.
 * `{ "welder": { "required": [...], "pending": [...] } }`.
 *
 * INSTRUCTIONS.md §4 describes a `{ job_categories: { <key>: { display_name,
 * required_equipment, optional_equipment } } }` shape. That does NOT match
 * the live route (src/api/routes.py:138-144) — there is no `job_categories`
 * wrapper and no `display_name` field. Verified against source, not Swagger,
 * on 2026-07-29. Flagged for Harshit; until confirmed, the job dropdown
 * (Phase 2) will use the object key itself as the display label.
 */
export type JobsResponse = Record<string, JobRequirements>;

/** Shape of a FastAPI HTTPException body (400 / 422 responses). */
export interface ApiErrorBody {
  detail: string | { msg?: string; [key: string]: unknown }[] | unknown;
}
