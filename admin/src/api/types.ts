/**
 * The shapes of docs/admin-api.md, written down once. Keep this file in step
 * with the contract: the mock backend and the pages are both typed against it,
 * so a change to the contract shows up as a compile error here first.
 */

export type Role = "admin" | "supervisor";

// ---- items (§4) -----------------------------------------------------------

export interface Item {
  key: string;
  display_name: string;
  /** Whether the detector was trained on it. A required item that isn't is judged by Gemini alone. */
  detectable: boolean;
  /** The loaded model's accuracy on the item; null if untrained or there is no metrics file. */
  detector_ap50: number | null;
}

// ---- jobs and the matrix (§5) ----------------------------------------------

/** `required`: checked by the detector, then Gemini. `pending`: listed for the worker as "verify manually", never scored. */
export type RequirementMode = "required" | "pending";

export interface Requirement {
  item: string;
  mode: RequirementMode;
}

export interface Job {
  key: string;
  display_name: string;
  position: number;
  is_active: boolean;
  version: number;
  requirements: Requirement[];
  updated_at: string;
}

// ---- warnings (§2) ---------------------------------------------------------

export type WarningCode =
  | "gemini_only_item"
  | "gemini_verify_off"
  | "worker_auth_required"
  | "gemini_verify_switched_off"
  | "threshold_changed"
  /** docs/checks-api.md §7: a retention period changed; says how many photos the next daily run deletes. */
  | "retention_changed";

export interface Warning {
  code: WarningCode;
  message: string;
  /** Set on `gemini_only_item`: the matrix cell to badge. */
  item?: string;
  /** Set on the settings warnings: the setting it is about. */
  setting?: string;
  /** Set on `retention_changed`: how many photos the next daily run will delete (checks-api §7). */
  count?: number;
}

// ---- sites (§3) ------------------------------------------------------------

export interface Site {
  key: string;
  display_name: string;
  is_active: boolean;
  worker_count: number;
  created_at: string;
}

// ---- settings (§6) ---------------------------------------------------------

export type SettingValue = number | boolean | string;

export interface Setting {
  name: string;
  type: "number" | "boolean" | "choice";
  min: number | null;
  max: number | null;
  step: number | null;
  choices: string[] | null;
  /** Show the old and new value and ask before saving. */
  confirm: boolean;
  description: string;
  value: SettingValue;
  default: SettingValue;
  overridden: boolean;
  updated_by: string | null;
  updated_at: string | null;
}

/** An RFC 7396 merge patch: left out = unchanged, a value = set, null = back to the default. */
export type SettingsPatch = Record<string, SettingValue | null>;

// ---- dashboard users (§7) ----------------------------------------------------

export interface AdminUser {
  id: number;
  email: string;
  display_name: string;
  role: Role;
  /** The supervisor's site key; null for an admin. */
  site: string | null;
  is_active: boolean;
  must_change_password: boolean;
  locked_until: string | null;
  last_login_at: string | null;
  created_at: string;
}

// ---- workers (§8) ----------------------------------------------------------

export interface Worker {
  worker_id: string;
  display_name: string;
  site: string;
  is_active: boolean;
  must_change_pin: boolean;
  locked_until: string | null;
  last_login_at: string | null;
  created_at: string;
  /** Email of the creating user; null for a worker made from the command line. */
  created_by: string | null;
}

export interface WorkerQuery {
  search?: string;
  site?: string;
  /** true, false, or omitted for both. */
  active?: boolean;
  /** true lists the workers locked out right now. */
  locked?: boolean;
  limit?: number;
  cursor?: string | null;
}

// ---- audit log (§9) --------------------------------------------------------

export interface AuditEntry {
  id: number;
  at: string;
  /** The user's email, or `cli` for a command run on the server. */
  actor: string;
  action: string;
  target: string;
  site: string | null;
  before: Record<string, unknown> | null;
  after: Record<string, unknown> | null;
  ip: string | null;
}

export interface AuditQuery {
  action?: string;
  target?: string;
  actor?: string;
  site?: string;
  /** Inclusive. */
  from?: string;
  /** Exclusive. */
  to?: string;
  limit?: number;
  cursor?: string | null;
}

// ---- checks (docs/checks-api.md) -------------------------------------------

export type Verdict = "PASS" | "FAIL" | "NEEDS_REVIEW";

/** One thing the detector found, in the stored photo's pixels (after EXIF orientation). */
export interface Detection {
  class: string;
  confidence: number;
  /** [x1, y1, x2, y2] */
  bbox: [number, number, number, number];
}

/** Gemini's second opinion on the required items the detector didn't find. */
export interface Verification {
  model: string;
  items: Array<{ item: string; status: "worn" | "not_worn" | "uncertain"; confidence: number; reason: string }>;
  elapsed_ms: number;
}

/** `kept`, `deleted` (on request or by retention), or `not_stored` (failed, switched off, or no login). */
export type PhotoState = "kept" | "deleted" | "not_stored";

export interface CheckPhoto {
  state: PhotoState;
  width: number | null;
  height: number | null;
  bytes: number | null;
  type: string | null;
}

export type ReviewOutcome = "compliant" | "not_compliant";

export interface Review {
  outcome: ReviewOutcome;
  note: string | null;
  /** The reviewer's email. */
  by: string;
  at: string;
}

/** A check as the list gives it: everything but `detections` and `verification` (§4.1). */
export interface CheckSummary {
  id: number;
  at: string;
  worker_id: string;
  /** null when the check was sent without a login under an ID no worker has. */
  worker_name: string | null;
  /** The worker's site when the check was made; null as for `worker_name`. */
  site: string | null;
  signed_in: boolean;
  job: string;
  job_name: string;
  job_version: number;
  /** The job's requirements when the check was made. */
  required: string[];
  pending: string[];
  verdict: Verdict;
  verdict_source: string;
  missing_equipment: string[];
  violations_detected: string[];
  unverifiable_equipment: string[];
  explanation: string | null;
  model_used: string;
  /** The three settings that decided the verdict, as they were then (§2). */
  confidence_threshold: number;
  gemini_verify: boolean;
  gemini_min_confidence: number;
  photo: CheckPhoto;
  training_consent: boolean;
  review: Review | null;
}

/** A check in full (§2, §4.2). */
export interface Check extends CheckSummary {
  detections: Detection[];
  verification: Verification | null;
}

export interface CheckQuery {
  site?: string;
  worker?: string;
  job?: string;
  verdict?: Verdict;
  reviewed?: boolean;
  signed_in?: boolean;
  from?: string;
  to?: string;
  limit?: number;
  cursor?: string | null;
}

// ---- paged lists (§2) ------------------------------------------------------

export interface Page<T> {
  items: T[];
  /** Opaque; null on the last page. */
  next_cursor: string | null;
  /** Everything that matches the filters, so a page can say "51–100 of 312". */
  total: number;
}

// ---- write responses -------------------------------------------------------

export interface JobResult {
  job: Job;
  warnings: Warning[];
  applies_within_s: number;
}

export interface SettingsResult {
  settings: Setting[];
  warnings: Warning[];
  applies_within_s: number;
}
