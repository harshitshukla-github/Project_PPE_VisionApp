/**
 * A stand-in for the Phase 3 backend, answering `/api/v1/admin/*` in the
 * browser with exactly the shapes and rules of docs/admin-api.md, so the
 * dashboard pages can be built and tried before the real endpoints exist.
 * `api/client.ts` loads it only when the mock is switched on; switching to
 * the real server changes nothing else.
 *
 * It keeps its state in memory (a page reload starts from the seed again) and
 * enforces what the contract promises: roles (§1), a supervisor seeing only
 * their own site (§1, §8), versions and `409 version_conflict` with the
 * current job (§5), the at-least-one-required rule for every job, warnings
 * (§2), the all-or-nothing settings patch (§6), paging with a total (§2), and
 * an audit entry for every write (§9).
 *
 * Assumption the contract doesn't state: a site's `worker_count` counts its
 * active workers.
 */
import type { DashboardUser } from "../api/client";
import type {
  AdminUser,
  AuditEntry,
  Job,
  Page,
  Requirement,
  Setting,
  SettingValue,
  Warning,
  Worker,
} from "../api/types";
import { seedState, type MockState } from "./seed";

export interface MockOptions {
  /** Who is signed in; the mock enforces what that user may do. */
  currentUser: () => DashboardUser | null;
  /** Simulated network time, so spinners are visible in the browser. */
  delayMs?: number;
  now?: () => Date;
}

export interface MockBackend {
  fetch: (url: string, init?: RequestInit) => Promise<Response>;
  /** The live state, so a test can play "someone else changed this". */
  state: MockState;
}

const APPLIES_WITHIN_S = 30;
const KEY_PATTERN = /^[a-z][a-z0-9_]{1,63}$/;
const WORKER_ID_PATTERN = /^[A-Za-z0-9][A-Za-z0-9-]{1,31}$/;
const EMAIL_PATTERN = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;

interface SettingDefinition extends Omit<Setting, "value" | "overridden" | "updated_by" | "updated_at"> {}

const SETTING_DEFINITIONS: SettingDefinition[] = [
  {
    name: "confidence_threshold", type: "number", min: 0.25, max: 0.8, step: 0.05, choices: null, confirm: true,
    description: "Minimum detector confidence for a detection to count.", default: 0.45,
  },
  {
    name: "gemini_verify", type: "boolean", min: null, max: null, step: null, choices: null, confirm: false,
    description: "Whether Gemini looks at the photo when the detector can't find a required item.", default: true,
  },
  {
    name: "gemini_min_confidence", type: "number", min: 0.5, max: 1, step: 0.05, choices: null, confirm: false,
    description: "How sure Gemini must be for its answer to change a verdict.", default: 0.75,
  },
  {
    name: "worker_auth", type: "choice", min: null, max: null, step: null, choices: ["optional", "required"], confirm: true,
    description: "Whether a check needs a worker login (auth-api.md §5).", default: "optional",
  },
];

const PASSWORD_WORDS = ["glade", "copper", "rivet", "maple", "harbor", "linen", "ember", "quartz", "willow", "anchor"];

interface Context {
  user: DashboardUser;
  params: string[];
  query: URLSearchParams;
  body: Record<string, unknown>;
}

type Handler = (ctx: Context) => Response;

export function createMockBackend(options: MockOptions): MockBackend {
  const now = options.now ?? (() => new Date());
  const state = seedState(now());

  // ---- responses ----------------------------------------------------------

  const send = (status: number, body?: unknown): Response =>
    body === undefined
      ? new Response(null, { status })
      : new Response(JSON.stringify(body), { status, headers: { "Content-Type": "application/json" } });

  const fail = (
    status: number,
    code: string,
    detail: string,
    extra: { field?: string; current?: Job } = {},
  ): Response => send(status, { detail, code, ...extra });

  const forbidden = () => fail(403, "forbidden", "You don't have access to this.");
  const notFound = (what: string) => fail(404, "not_found", `No such ${what}.`);

  // ---- who is asking ------------------------------------------------------

  /** The signed-in user must exist as a dashboard user, so "yourself" means something. */
  function ensureSelf(user: DashboardUser): void {
    if (state.users.some((u) => u.email === user.email)) return;
    state.users.unshift({
      id: user.id, email: user.email, display_name: user.display_name, role: user.role, site: user.site,
      is_active: true, must_change_password: false, locked_until: null, last_login_at: now().toISOString(),
      created_at: now().toISOString(),
    });
  }

  const isAdmin = (user: DashboardUser) => user.role === "admin";

  // ---- helpers ------------------------------------------------------------

  const stamp = () => now().toISOString();

  function record(
    user: DashboardUser,
    action: string,
    target: string,
    site: string | null,
    before: Record<string, unknown> | null,
    after: Record<string, unknown> | null,
  ): void {
    const entry: AuditEntry = {
      id: state.nextAuditId++, at: stamp(), actor: user.email, action, target, site, before, after, ip: "127.0.0.1",
    };
    state.audit.push(entry);
  }

  function readJob(key: string): Job | undefined {
    return state.jobs.find((j) => j.key === key);
  }

  const sortedJobs = () => [...state.jobs].sort((a, b) => a.position - b.position);

  function readSetting(name: string): Setting {
    const def = SETTING_DEFINITIONS.find((d) => d.name === name)!;
    const saved = state.settings[name];
    return {
      ...def,
      value: saved ? saved.value : def.default,
      overridden: !!saved,
      updated_by: saved?.updated_by ?? null,
      updated_at: saved?.updated_at ?? null,
    };
  }

  function geminiVerifyOn(): boolean {
    return readSetting("gemini_verify").value === true;
  }

  /** The first problem in a requirements list, as the response to send, or null if it is fine. */
  function checkRequirements(list: unknown): Response | null {
    if (!Array.isArray(list)) return fail(422, "invalid", "requirements must be a list.", { field: "requirements" });
    const seen = new Set<string>();
    for (let i = 0; i < list.length; i++) {
      const entry = list[i] as Partial<Requirement> | null;
      const field = `requirements[${i}].item`;
      if (!entry || typeof entry.item !== "string") return fail(422, "invalid", "Each requirement needs an item.", { field });
      if (!state.items.some((it) => it.key === entry.item)) {
        return fail(422, "unknown_item", `"${entry.item}" isn't a known item.`, { field });
      }
      if (entry.mode !== "required" && entry.mode !== "pending") {
        return fail(422, "invalid", "A requirement's mode is required or pending.", { field: `requirements[${i}].mode` });
      }
      if (seen.has(entry.item)) return fail(422, "invalid", `"${entry.item}" is listed twice.`, { field });
      seen.add(entry.item);
    }
    if (!(list as Requirement[]).some((r) => r.mode === "required")) {
      return fail(422, "no_required_item", "A job needs at least one required item.", { field: "requirements" });
    }
    return null;
  }

  function requirementWarnings(requirements: Requirement[]): Warning[] {
    const warnings: Warning[] = [];
    for (const r of requirements) {
      const item = state.items.find((it) => it.key === r.item);
      if (r.mode === "required" && item && !item.detectable) {
        warnings.push({
          code: "gemini_only_item",
          item: item.key,
          message: `${item.display_name} can only be judged by Gemini, whose accuracy on it hasn't been measured.`,
        });
      }
    }
    if (!geminiVerifyOn() && requirements.some((r) => r.mode === "required")) {
      warnings.push({
        code: "gemini_verify_off",
        message: "Gemini verification is off, so anything the detector misses stays \"Needs review\".",
      });
    }
    return warnings;
  }

  function versionConflict(job: Job): Response {
    return fail(409, "version_conflict", "Someone changed this job while you were editing it.", {
      current: structuredClone(job),
    });
  }

  function activeSite(key: unknown): { site?: MockState["sites"][number]; error?: Response } {
    const site = state.sites.find((s) => s.key === key);
    if (!site) return { error: notFound("site") };
    if (!site.is_active) return { error: fail(422, "invalid", `${site.display_name} is deactivated.`, { field: "site" }) };
    return { site };
  }

  function pageOf<T>(rows: T[], q: URLSearchParams): Page<T> | Response {
    const rawLimit = q.get("limit");
    const limit = rawLimit ? Number(rawLimit) : 50;
    if (!Number.isInteger(limit) || limit < 1 || limit > 200) {
      return fail(422, "invalid", "limit must be between 1 and 200.", { field: "limit" });
    }
    let offset = 0;
    const cursor = q.get("cursor");
    if (cursor) {
      try {
        offset = JSON.parse(atob(cursor)).o;
      } catch {
        return fail(422, "invalid", "That cursor isn't valid.", { field: "cursor" });
      }
    }
    const end = offset + limit;
    return {
      items: rows.slice(offset, end),
      next_cursor: end < rows.length ? btoa(JSON.stringify({ o: end })) : null,
      total: rows.length,
    };
  }

  function oneTimeSecret(): string {
    // Six digits that aren't one digit repeated or a straight run (auth-api §4).
    for (;;) {
      const pin = String(Math.floor(Math.random() * 1_000_000)).padStart(6, "0");
      const digits = [...pin].map(Number);
      const repeated = digits.every((d) => d === digits[0]);
      const up = digits.every((d, i) => i === 0 || d === digits[i - 1] + 1);
      const down = digits.every((d, i) => i === 0 || d === digits[i - 1] - 1);
      if (!repeated && !up && !down) return pin;
    }
  }

  function temporaryPassword(): string {
    const pick = () => PASSWORD_WORDS[Math.floor(Math.random() * PASSWORD_WORDS.length)];
    return `${pick()}-${pick()}-${10 + Math.floor(Math.random() * 90)}-${pick()}`;
  }

  // ---- sites (§3) ---------------------------------------------------------

  const siteView = (s: MockState["sites"][number]) => ({
    key: s.key,
    display_name: s.display_name,
    is_active: s.is_active,
    worker_count: state.workers.filter((w) => w.site === s.key && w.is_active).length,
    created_at: s.created_at,
  });

  const listSites: Handler = ({ user }) =>
    send(200, state.sites.filter((s) => isAdmin(user) || s.key === user.site).map(siteView));

  const createSite: Handler = ({ user, body }) => {
    if (!isAdmin(user)) return forbidden();
    const key = body.key;
    if (typeof key !== "string" || !KEY_PATTERN.test(key)) {
      return fail(422, "invalid_key", "A site key is lowercase letters, digits and underscores, starting with a letter.", { field: "key" });
    }
    if (typeof body.display_name !== "string" || !body.display_name.trim()) {
      return fail(422, "invalid", "A site needs a name.", { field: "display_name" });
    }
    if (state.sites.some((s) => s.key === key)) return fail(409, "already_exists", `The site key "${key}" is taken.`, { field: "key" });
    const site = { key, display_name: body.display_name.trim(), is_active: true, created_at: stamp() };
    state.sites.push(site);
    record(user, "site.create", `site:${key}`, key, null, { key, display_name: site.display_name });
    return send(201, { site: siteView(site) });
  };

  const patchSite: Handler = ({ user, params, body }) => {
    if (!isAdmin(user)) return forbidden();
    const site = state.sites.find((s) => s.key === params[0]);
    if (!site) return notFound("site");
    const before: Record<string, unknown> = {};
    const after: Record<string, unknown> = {};
    if (typeof body.display_name === "string" && body.display_name.trim() && body.display_name.trim() !== site.display_name) {
      before.display_name = site.display_name;
      after.display_name = body.display_name.trim();
    }
    if (typeof body.is_active === "boolean" && body.is_active !== site.is_active) {
      if (!body.is_active) {
        const inUse =
          state.workers.some((w) => w.site === site.key && w.is_active) ||
          state.users.some((u) => u.site === site.key && u.is_active);
        if (inUse) {
          return fail(409, "site_in_use", `${site.display_name} still has active workers or supervisors. Move or deactivate them first.`);
        }
      }
      before.is_active = site.is_active;
      after.is_active = body.is_active;
    }
    Object.assign(site, after);
    if (Object.keys(after).length) record(user, "site.update", `site:${site.key}`, site.key, before, after);
    return send(200, { site: siteView(site) });
  };

  // ---- items (§4) and jobs (§5) --------------------------------------------

  const listItems: Handler = () => send(200, state.items);

  const listJobs: Handler = () => send(200, sortedJobs());

  const createJob: Handler = ({ user, body }) => {
    if (!isAdmin(user)) return forbidden();
    const key = body.key;
    if (typeof key !== "string" || !KEY_PATTERN.test(key)) {
      return fail(422, "invalid_key", "A job key is lowercase letters, digits and underscores, starting with a letter.", { field: "key" });
    }
    if (typeof body.display_name !== "string" || !body.display_name.trim()) {
      return fail(422, "invalid", "A job needs a name.", { field: "display_name" });
    }
    if (readJob(key)) return fail(409, "already_exists", `The job key "${key}" is taken.`, { field: "key" });
    const bad = checkRequirements(body.requirements);
    if (bad) return bad;
    const job: Job = {
      key, display_name: body.display_name.trim(), position: state.jobs.length, is_active: true, version: 1,
      requirements: structuredClone(body.requirements as Requirement[]), updated_at: stamp(),
    };
    state.jobs.push(job);
    record(user, "job.create", `job:${key}`, null, null, {
      key, display_name: job.display_name, requirements: job.requirements,
    });
    return send(201, { job, warnings: requirementWarnings(job.requirements), applies_within_s: APPLIES_WITHIN_S });
  };

  const patchJob: Handler = ({ user, params, body }) => {
    if (!isAdmin(user)) return forbidden();
    const job = readJob(params[0]);
    if (!job) return notFound("job");
    if (body.version !== job.version) return versionConflict(job);
    const before: Record<string, unknown> = {};
    const after: Record<string, unknown> = {};
    if (typeof body.display_name === "string" && body.display_name.trim() && body.display_name.trim() !== job.display_name) {
      before.display_name = job.display_name;
      after.display_name = body.display_name.trim();
    }
    if (typeof body.is_active === "boolean" && body.is_active !== job.is_active) {
      if (!body.is_active && state.jobs.filter((j) => j.is_active).length === 1) {
        return fail(409, "last_active_job", "That would leave the app with no jobs to choose from.");
      }
      before.is_active = job.is_active;
      after.is_active = body.is_active;
    }
    if (Object.keys(after).length) {
      Object.assign(job, after);
      job.version += 1;
      job.updated_at = stamp();
      record(user, "job.update", `job:${job.key}`, null, before, after);
    }
    return send(200, { job, applies_within_s: APPLIES_WITHIN_S });
  };

  const putRequirements: Handler = ({ user, params, body }) => {
    if (!isAdmin(user)) return forbidden();
    const job = readJob(params[0]);
    if (!job) return notFound("job");
    if (body.version !== job.version) return versionConflict(job);
    const bad = checkRequirements(body.requirements);
    if (bad) return bad;
    const requirements = structuredClone(body.requirements as Requirement[]);
    if (JSON.stringify(requirements) !== JSON.stringify(job.requirements)) {
      record(user, "job.requirements", `job:${job.key}`, null, { requirements: job.requirements }, { requirements });
      job.requirements = requirements;
      job.version += 1;
      job.updated_at = stamp();
    }
    return send(200, { job, warnings: requirementWarnings(job.requirements), applies_within_s: APPLIES_WITHIN_S });
  };

  const putJobOrder: Handler = ({ user, body }) => {
    if (!isAdmin(user)) return forbidden();
    const keys = body.keys;
    if (!Array.isArray(keys)) return fail(422, "invalid", "keys must be a list.", { field: "keys" });
    const seen = new Set<string>();
    for (let i = 0; i < keys.length; i++) {
      if (!readJob(keys[i]) || seen.has(keys[i])) {
        return fail(422, "invalid", `"${keys[i]}" is unknown or listed twice.`, { field: `keys[${i}]` });
      }
      seen.add(keys[i]);
    }
    if (seen.size !== state.jobs.length) {
      return fail(422, "invalid", "Every job, active or not, must appear exactly once.", { field: "keys" });
    }
    const before = sortedJobs().map((j) => j.key);
    keys.forEach((key: string, position: number) => {
      readJob(key)!.position = position; // versions don't change
    });
    record(user, "job.order", "jobs", null, { keys: before }, { keys });
    return send(200, sortedJobs());
  };

  // ---- settings (§6) -------------------------------------------------------

  const listSettings: Handler = ({ user }) =>
    isAdmin(user) ? send(200, SETTING_DEFINITIONS.map((d) => readSetting(d.name))) : forbidden();

  const patchSettings: Handler = ({ user, body }) => {
    if (!isAdmin(user)) return forbidden();
    // All or nothing: check every value before saving any.
    const changes: Array<{ name: string; value: SettingValue | null }> = [];
    for (const [name, value] of Object.entries(body)) {
      const def = SETTING_DEFINITIONS.find((d) => d.name === name);
      if (!def) return fail(422, "invalid", `"${name}" isn't a setting.`, { field: name });
      if (value !== null) {
        if (def.type === "number") {
          if (typeof value !== "number") return fail(422, "invalid", `${name} must be a number.`, { field: name });
          if (value < def.min! || value > def.max!) {
            return fail(422, "out_of_range", `${name} must be between ${def.min} and ${def.max}.`, { field: name });
          }
        } else if (def.type === "boolean") {
          if (typeof value !== "boolean") return fail(422, "invalid", `${name} must be true or false.`, { field: name });
        } else if (!def.choices!.includes(value as string)) {
          return fail(422, "invalid", `${name} must be one of: ${def.choices!.join(", ")}.`, { field: name });
        }
      }
      changes.push({ name, value: value as SettingValue | null });
    }

    const warnings: Warning[] = [];
    for (const { name, value } of changes) {
      const was = readSetting(name);
      const savedBefore = state.settings[name]?.value ?? null;
      if (value === null) delete state.settings[name];
      else state.settings[name] = { value, updated_by: user.email, updated_at: stamp() };
      const effective = readSetting(name).value;
      if (effective === was.value && savedBefore === (value ?? null)) continue;
      record(user, "settings.update", `setting:${name}`, null, { value: savedBefore }, { value });
      if (name === "worker_auth" && effective === "required" && was.value !== "required") {
        warnings.push({
          code: "worker_auth_required", setting: name,
          message: "Checks from an app without login will now be refused.",
        });
      }
      if (name === "gemini_verify" && effective === false && was.value !== false) {
        warnings.push({
          code: "gemini_verify_switched_off", setting: name,
          message: "Gemini verification is off: anything the detector misses stays \"Needs review\".",
        });
      }
      if (name === "confidence_threshold" && effective !== was.value) {
        warnings.push({
          code: "threshold_changed", setting: name,
          message: "Every verdict from now on uses the new detector threshold.",
        });
      }
    }
    return send(200, {
      settings: SETTING_DEFINITIONS.map((d) => readSetting(d.name)),
      warnings,
      applies_within_s: APPLIES_WITHIN_S,
    });
  };

  // ---- dashboard users (§7) -------------------------------------------------

  const listUsers: Handler = ({ user }) => (isAdmin(user) ? send(200, state.users) : forbidden());

  /** A role and site that go together, or the response to send. */
  function checkRoleAndSite(role: unknown, site: unknown): Response | null {
    if (role !== "admin" && role !== "supervisor") return fail(422, "invalid", "A role is admin or supervisor.", { field: "role" });
    if (role === "admin") {
      return site ? fail(422, "invalid", "An admin doesn't belong to a site.", { field: "site" }) : null;
    }
    if (!site) return fail(422, "invalid", "A supervisor needs a site.", { field: "site" });
    return activeSite(site).error ?? null;
  }

  const createUser: Handler = ({ user, body }) => {
    if (!isAdmin(user)) return forbidden();
    const email = typeof body.email === "string" ? body.email.trim().toLowerCase() : "";
    if (!EMAIL_PATTERN.test(email)) return fail(422, "invalid", "That isn't a valid email address.", { field: "email" });
    if (typeof body.display_name !== "string" || !body.display_name.trim()) {
      return fail(422, "invalid", "A user needs a name.", { field: "display_name" });
    }
    const bad = checkRoleAndSite(body.role, body.site ?? null);
    if (bad) return bad;
    if (state.users.some((u) => u.email === email)) return fail(409, "already_exists", `${email} already has an account.`, { field: "email" });
    const created: AdminUser = {
      id: state.nextUserId++, email, display_name: body.display_name.trim(), role: body.role as AdminUser["role"],
      site: body.role === "admin" ? null : (body.site as string), is_active: true, must_change_password: true,
      locked_until: null, last_login_at: null, created_at: stamp(),
    };
    state.users.push(created);
    record(user, "user.create", `user:${email}`, created.site, null, {
      email, display_name: created.display_name, role: created.role, site: created.site,
    });
    return send(201, { user: created, temporary_password: temporaryPassword() });
  };

  const patchUser: Handler = ({ user, params, body }) => {
    if (!isAdmin(user)) return forbidden();
    const target = state.users.find((u) => String(u.id) === params[0]);
    if (!target) return notFound("user");
    const touchesSelf =
      target.email === user.email &&
      (("role" in body && body.role !== target.role) ||
        ("site" in body && (body.site ?? null) !== target.site) ||
        ("is_active" in body && body.is_active !== target.is_active));
    if (touchesSelf) return fail(409, "cannot_change_self", "You can't change your own role, site or active flag.");

    const role = (body.role ?? target.role) as AdminUser["role"];
    const site = role === "admin" ? null : "site" in body ? (body.site as string | null) : target.site;
    if ("role" in body || "site" in body) {
      const bad = checkRoleAndSite(role, "site" in body || role === "supervisor" ? site : null);
      if (bad) return bad;
    }
    const isActive = typeof body.is_active === "boolean" ? body.is_active : target.is_active;
    const losesAdmin = target.role === "admin" && target.is_active && (role !== "admin" || !isActive);
    if (losesAdmin && state.users.filter((u) => u.role === "admin" && u.is_active).length === 1) {
      return fail(409, "last_admin", "That would leave no active admin.");
    }

    const next: Partial<AdminUser> = {};
    if (typeof body.display_name === "string" && body.display_name.trim()) next.display_name = body.display_name.trim();
    next.role = role;
    next.site = site;
    next.is_active = isActive;
    const before: Record<string, unknown> = {};
    const after: Record<string, unknown> = {};
    for (const field of ["display_name", "role", "site", "is_active"] as const) {
      if (next[field] !== undefined && next[field] !== target[field]) {
        before[field] = target[field];
        after[field] = next[field];
      }
    }
    Object.assign(target, after);
    if (Object.keys(after).length) record(user, "user.update", `user:${target.email}`, target.site, before, after);
    return send(200, { user: target });
  };

  const resetUserPassword: Handler = ({ user, params }) => {
    if (!isAdmin(user)) return forbidden();
    const target = state.users.find((u) => String(u.id) === params[0]);
    if (!target) return notFound("user");
    if (target.email === user.email) {
      return fail(409, "cannot_change_self", "Use Change password to reset your own password.");
    }
    target.must_change_password = true;
    target.locked_until = null;
    record(user, "user.reset_password", `user:${target.email}`, target.site, null, null);
    return send(200, { temporary_password: temporaryPassword() });
  };

  // ---- workers (§8) ---------------------------------------------------------

  /** A supervisor's workers are their site's; for anyone else's they get a 404. */
  const visibleWorker = (user: DashboardUser, w: Worker | undefined) =>
    w && (isAdmin(user) || w.site === user.site) ? w : undefined;

  const findWorker = (id: string) => state.workers.find((w) => w.worker_id.toLowerCase() === id.toLowerCase());

  const listWorkers: Handler = ({ user, query }) => {
    const search = (query.get("search") ?? "").trim().toLowerCase();
    const siteFilter = isAdmin(user) ? query.get("site") : user.site; // a supervisor always gets their own site
    const active = query.get("active");
    const locked = query.get("locked") === "true";
    const rows = state.workers
      .filter((w) => isAdmin(user) || w.site === user.site)
      .filter((w) => !siteFilter || w.site === siteFilter)
      .filter((w) => active === null || String(w.is_active) === active)
      .filter((w) => !locked || (w.locked_until !== null && new Date(w.locked_until) > now()))
      .filter((w) => !search || w.worker_id.toLowerCase().includes(search) || w.display_name.toLowerCase().includes(search))
      .sort((a, b) => a.worker_id.localeCompare(b.worker_id));
    const page = pageOf(rows, query);
    return page instanceof Response ? page : send(200, page);
  };

  const getWorker: Handler = ({ user, params }) => {
    const worker = visibleWorker(user, findWorker(params[0]));
    return worker ? send(200, worker) : notFound("worker");
  };

  const createWorker: Handler = ({ user, body }) => {
    const id = typeof body.worker_id === "string" ? body.worker_id.trim() : "";
    if (!WORKER_ID_PATTERN.test(id)) {
      return fail(422, "invalid", "A worker ID is 2 to 32 letters, digits and hyphens, not starting with a hyphen.", { field: "worker_id" });
    }
    if (typeof body.display_name !== "string" || !body.display_name.trim()) {
      return fail(422, "invalid", "A worker needs a name.", { field: "display_name" });
    }
    let siteKey: string;
    if (isAdmin(user)) {
      if (!body.site) return fail(422, "invalid", "Choose the worker's site.", { field: "site" });
      siteKey = String(body.site);
    } else {
      if (body.site && body.site !== user.site) return forbidden();
      siteKey = user.site ?? "";
    }
    const checked = activeSite(siteKey);
    if (checked.error) return checked.error;
    // The one thing a supervisor can learn about another site: the ID is taken, not where.
    if (findWorker(id)) return fail(409, "already_exists", `The worker ID ${id.toUpperCase()} is taken.`, { field: "worker_id" });
    const worker: Worker = {
      worker_id: id.toUpperCase(), display_name: body.display_name.trim(), site: siteKey, is_active: true,
      must_change_pin: true, locked_until: null, last_login_at: null, created_at: stamp(), created_by: user.email,
    };
    state.workers.push(worker);
    record(user, "worker.create", `worker:${worker.worker_id}`, siteKey, null, {
      worker_id: worker.worker_id, display_name: worker.display_name, site: siteKey,
    });
    return send(201, { worker, one_time_pin: oneTimeSecret() });
  };

  const patchWorker: Handler = ({ user, params, body }) => {
    const worker = visibleWorker(user, findWorker(params[0]));
    if (!worker) return notFound("worker");
    if ("site" in body && !isAdmin(user)) return forbidden();
    const before: Record<string, unknown> = {};
    const after: Record<string, unknown> = {};
    if (typeof body.display_name === "string" && body.display_name.trim() && body.display_name.trim() !== worker.display_name) {
      before.display_name = worker.display_name;
      after.display_name = body.display_name.trim();
    }
    if ("site" in body && body.site !== worker.site) {
      const checked = activeSite(body.site);
      if (checked.error) return checked.error;
      before.site = worker.site;
      after.site = body.site;
    }
    if (typeof body.is_active === "boolean" && body.is_active !== worker.is_active) {
      before.is_active = worker.is_active;
      after.is_active = body.is_active;
    }
    Object.assign(worker, after);
    if (Object.keys(after).length) record(user, "worker.update", `worker:${worker.worker_id}`, worker.site, before, after);
    return send(200, { worker });
  };

  const resetWorkerPin: Handler = ({ user, params }) => {
    const worker = visibleWorker(user, findWorker(params[0]));
    if (!worker) return notFound("worker");
    worker.must_change_pin = true;
    worker.locked_until = null;
    record(user, "worker.reset_pin", `worker:${worker.worker_id}`, worker.site, null, null);
    return send(200, { one_time_pin: oneTimeSecret() });
  };

  // ---- audit log (§9) --------------------------------------------------------

  const listAudit: Handler = ({ user, query }) => {
    if (!isAdmin(user)) return forbidden();
    const exact = (value: string | null, wanted: string | null) => wanted === null || wanted === "" || value === wanted;
    const from = query.get("from");
    const to = query.get("to");
    const rows = state.audit
      .filter((e) => exact(e.action, query.get("action")))
      .filter((e) => exact(e.target, query.get("target")))
      .filter((e) => exact(e.actor, query.get("actor")))
      .filter((e) => exact(e.site, query.get("site")))
      .filter((e) => !from || e.at >= from)
      .filter((e) => !to || e.at < to)
      .sort((a, b) => b.id - a.id);
    const page = pageOf(rows, query);
    return page instanceof Response ? page : send(200, page);
  };

  // ---- routing ---------------------------------------------------------------

  const routes: Array<[string, RegExp, Handler]> = [
    ["GET", /^\/items$/, listItems],
    ["GET", /^\/jobs$/, listJobs],
    ["POST", /^\/jobs$/, createJob],
    ["PATCH", /^\/jobs\/([^/]+)$/, patchJob],
    ["PUT", /^\/jobs\/([^/]+)\/requirements$/, putRequirements],
    ["PUT", /^\/job-order$/, putJobOrder],
    ["GET", /^\/sites$/, listSites],
    ["POST", /^\/sites$/, createSite],
    ["PATCH", /^\/sites\/([^/]+)$/, patchSite],
    ["GET", /^\/settings$/, listSettings],
    ["PATCH", /^\/settings$/, patchSettings],
    ["GET", /^\/users$/, listUsers],
    ["POST", /^\/users$/, createUser],
    ["PATCH", /^\/users\/([^/]+)$/, patchUser],
    ["POST", /^\/users\/([^/]+)\/reset-password$/, resetUserPassword],
    ["GET", /^\/workers$/, listWorkers],
    ["GET", /^\/workers\/([^/]+)$/, getWorker],
    ["POST", /^\/workers$/, createWorker],
    ["PATCH", /^\/workers\/([^/]+)$/, patchWorker],
    ["POST", /^\/workers\/([^/]+)\/reset-pin$/, resetWorkerPin],
    ["GET", /^\/audit$/, listAudit],
  ];

  async function handle(url: string, init?: RequestInit): Promise<Response> {
    if (options.delayMs) await new Promise((resolve) => setTimeout(resolve, options.delayMs));
    const user = options.currentUser();
    if (!user) return fail(401, "not_authenticated", "You're not signed in, or your session ended.");
    if (user.must_change_password) {
      return fail(403, "password_change_required", "Choose a new password before continuing.");
    }
    ensureSelf(user);

    const parsed = new URL(url, "http://mock.local");
    const path = parsed.pathname.replace(/^\/api\/v1\/admin/, "");
    const method = (init?.method ?? "GET").toUpperCase();
    for (const [routeMethod, pattern, handler] of routes) {
      const match = pattern.exec(path);
      if (routeMethod === method && match) {
        let body: Record<string, unknown> = {};
        if (typeof init?.body === "string" && init.body) body = JSON.parse(init.body);
        return handler({ user, params: match.slice(1).map(decodeURIComponent), query: parsed.searchParams, body });
      }
    }
    return notFound("page");
  }

  return { fetch: handle, state };
}
