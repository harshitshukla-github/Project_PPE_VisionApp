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
  | "threshold_changed";

export interface Warning {
  code: WarningCode;
  message: string;
  /** Set on `gemini_only_item`: the matrix cell to badge. */
  item?: string;
  /** Set on the settings warnings: the setting it is about. */
  setting?: string;
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
