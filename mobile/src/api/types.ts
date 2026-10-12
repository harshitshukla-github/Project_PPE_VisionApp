/**
 * API response/request types for the PPB compliance backend.
 *
 * Mirrors src/inference/schemas.py and src/api/routes.py in the repo root
 * (NOT the simplified example in INSTRUCTIONS.md §4 — verified directly
 * against source on 2026-07-29, since two fields there are stale/wrong,
 * see the GET /jobs note below).
 */

/**
 * "FAIL" requires positive evidence of a breach. "NEEDS_REVIEW" means neither
 * the detector nor Gemini could confirm a required item and nothing was seen
 * to be wrong — it must never be rendered as a failure (INSTRUCTIONS.md §3.1).
 */
export type Verdict = "PASS" | "FAIL" | "NEEDS_REVIEW";

/** Who settled the verdict: YOLO alone, or YOLO plus Gemini's look at the photo. */
export type VerdictSource = "detector" | "detector+gemini";

/** Gemini's finding for one required item the detector did not find. */
export type ItemStatus = "worn" | "not_worn" | "uncertain";

export interface ItemVerification {
  /** PPE class name, e.g. "safety_vest". */
  item: string;
  status: ItemStatus;
  /** Verifier's own confidence, 0..1. */
  confidence: number;
  /** One short sentence of visual evidence, e.g. "orange hi-vis vest on torso". */
  reason: string;
}

/** Second-opinion check of the photo — present only when Gemini actually looked. */
export interface GeminiVerification {
  model: string;
  items: ItemVerification[];
  elapsed_ms: number;
}

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
  /**
   * On FAIL: required items confirmed not worn. On NEEDS_REVIEW: required
   * items neither the detector nor Gemini could confirm — "couldn't verify",
   * NOT "missing".
   */
  missing_equipment: string[];
  /** Explicit violation classes detected (e.g. "no_hardhat"). */
  violations_detected: string[];
  /**
   * Required items that could not be checked from this photo (no training
   * data for the class, or Gemini couldn't see it clearly).
   * Render as amber "verify manually" — never as a failure.
   */
  unverifiable_equipment: string[];
  /** Gemini-generated explanation. Present ONLY on FAIL, else null. */
  explanation: string | null;
  /**
   * Who decided. Optional because History entries saved before this field
   * existed, and responses from an older backend, don't carry it.
   */
  verdict_source?: VerdictSource;
  /** Gemini's per-item findings; null unless it looked at the photo. Absent on legacy entries. */
  verification?: GeminiVerification | null;
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
