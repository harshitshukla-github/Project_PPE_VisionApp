import { beforeEach, describe, expect, it } from "vitest";

import type { DashboardUser } from "../api/client";
import { createMockBackend, type MockBackend } from "./backend";

const ADMIN: DashboardUser = {
  id: 1, email: "admin@example.com", display_name: "Test Admin", role: "admin", site: null, must_change_password: false,
};
const SUPERVISOR_B: DashboardUser = {
  id: 101, email: "supervisor.b@example.com", display_name: "Site Supervisor B", role: "supervisor",
  site: "pune_tower_b", must_change_password: false,
};

let signedIn: DashboardUser | null;
let backend: MockBackend;

beforeEach(() => {
  signedIn = ADMIN;
  backend = createMockBackend({ currentUser: () => signedIn, now: () => new Date("2026-10-25T12:00:00Z") });
});

async function call(method: string, path: string, body?: unknown) {
  const response = await backend.fetch(`/api/v1/admin${path}`, {
    method,
    body: body === undefined ? undefined : JSON.stringify(body),
  });
  const text = await response.text();
  return { status: response.status, body: text ? JSON.parse(text) : null };
}

const job = (key: string) => backend.state.jobs.find((j) => j.key === key)!;

describe("who may ask (§1, §2)", () => {
  it("401 not_authenticated when nobody is signed in", async () => {
    signedIn = null;
    expect((await call("GET", "/jobs")).body).toMatchObject({ code: "not_authenticated" });
  });

  it("403 password_change_required while the temporary password is still in use", async () => {
    signedIn = { ...ADMIN, must_change_password: true };
    const res = await call("GET", "/jobs");
    expect(res.status).toBe(403);
    expect(res.body.code).toBe("password_change_required");
  });

  it.each([
    ["POST", "/jobs", { key: "x_job", display_name: "X", requirements: [{ item: "hardhat", mode: "required" }] }],
    ["PUT", "/job-order", { keys: [] }],
    ["GET", "/settings", undefined],
    ["PATCH", "/settings", {}],
    ["GET", "/users", undefined],
    ["POST", "/sites", { key: "new_site", display_name: "New" }],
    ["GET", "/audit", undefined],
  ])("403 forbidden for a supervisor on %s %s", async (method, path, body) => {
    signedIn = SUPERVISOR_B;
    const res = await call(method, path, body);
    expect(res.status).toBe(403);
    expect(res.body.code).toBe("forbidden");
  });

  it("lets a supervisor read the matrix (items and jobs)", async () => {
    signedIn = SUPERVISOR_B;
    expect((await call("GET", "/items")).status).toBe(200);
    expect((await call("GET", "/jobs")).status).toBe(200);
  });
});

describe("items and jobs (§4, §5)", () => {
  it("lists items with detectable and detector_ap50 (null when untrained)", async () => {
    const items = (await call("GET", "/items")).body;
    expect(items.find((i: { key: string }) => i.key === "hardhat")).toMatchObject({ detectable: true, detector_ap50: 0.909 });
    expect(items.find((i: { key: string }) => i.key === "face_shield")).toMatchObject({ detectable: false, detector_ap50: null });
  });

  it("lists every job in display order, with a version", async () => {
    const jobs = (await call("GET", "/jobs")).body;
    expect(jobs.map((j: { position: number }) => j.position)).toEqual([0, 1, 2, 3, 4, 5]);
    expect(jobs[1]).toMatchObject({ key: "welder", version: 4, is_active: true });
  });

  it("PUT requirements saves, bumps the version by one, and warns about a Gemini-only required item", async () => {
    const res = await call("PUT", "/jobs/painter/requirements", {
      version: 2,
      requirements: [{ item: "safety_vest", mode: "required" }, { item: "face_shield", mode: "required" }],
    });
    expect(res.status).toBe(200);
    expect(res.body.job.version).toBe(3);
    expect(res.body.applies_within_s).toBe(30);
    expect(res.body.warnings).toEqual([
      expect.objectContaining({ code: "gemini_only_item", item: "face_shield" }),
    ]);
  });

  it("warns once when Gemini verification is switched off (no extra key)", async () => {
    await call("PATCH", "/settings", { gemini_verify: false });
    const res = await call("PUT", "/jobs/painter/requirements", {
      version: 2, requirements: [{ item: "safety_vest", mode: "required" }],
    });
    expect(res.body.warnings).toEqual([expect.objectContaining({ code: "gemini_verify_off" })]);
    expect(res.body.warnings[0]).not.toHaveProperty("setting");
  });

  it("409 version_conflict carries the job as it is now, and saves nothing", async () => {
    job("welder").version = 5; // someone else saved in between
    job("welder").display_name = "Welder (changed elsewhere)";
    const res = await call("PUT", "/jobs/welder/requirements", {
      version: 4, requirements: [{ item: "hardhat", mode: "required" }],
    });
    expect(res.status).toBe(409);
    expect(res.body.code).toBe("version_conflict");
    expect(res.body.current).toMatchObject({ key: "welder", version: 5, display_name: "Welder (changed elsewhere)" });
    expect(job("welder").requirements.length).toBeGreaterThan(1); // untouched
  });

  it.each([
    ["an unknown item", [{ item: "flippers", mode: "required" }], 422, "unknown_item", "requirements[0].item"],
    ["an item listed twice", [{ item: "hardhat", mode: "required" }, { item: "hardhat", mode: "pending" }], 422, "invalid", "requirements[1].item"],
    ["no required item", [{ item: "hardhat", mode: "pending" }], 422, "no_required_item", "requirements"],
  ])("422 for %s, naming the entry in `field`", async (_name, requirements, status, code, field) => {
    const res = await call("PUT", "/jobs/painter/requirements", { version: 2, requirements });
    expect(res.status).toBe(status);
    expect(res.body).toMatchObject({ code, field });
  });

  it("reordering changes positions but not versions", async () => {
    const before = Object.fromEntries(backend.state.jobs.map((j) => [j.key, j.version]));
    const keys = ["painter", "general_labor", "welder", "electrician", "height_work", "demolition"];
    const res = await call("PUT", "/job-order", { keys });
    expect(res.status).toBe(200);
    expect(res.body.map((j: { key: string }) => j.key)).toEqual(keys);
    expect(Object.fromEntries(backend.state.jobs.map((j) => [j.key, j.version]))).toEqual(before);
  });

  it("422 when the order leaves a job out", async () => {
    const res = await call("PUT", "/job-order", { keys: ["welder"] });
    expect(res.status).toBe(422);
    expect(res.body).toMatchObject({ code: "invalid", field: "keys" });
  });

  it("creates a job last, active, at version 1; refuses a bad or taken key", async () => {
    const created = await call("POST", "/jobs", {
      key: "scaffolder", display_name: "Scaffolder",
      requirements: [{ item: "hardhat", mode: "required" }, { item: "harness", mode: "pending" }],
    });
    expect(created.status).toBe(201);
    expect(created.body.job).toMatchObject({ key: "scaffolder", position: 6, is_active: true, version: 1 });
    expect((await call("POST", "/jobs", { key: "Bad Key", display_name: "x", requirements: [] })).body.code).toBe("invalid_key");
    expect((await call("POST", "/jobs", { key: "welder", display_name: "x", requirements: [{ item: "hardhat", mode: "required" }] })).body.code).toBe("already_exists");
  });

  it("refuses to deactivate the last active job, and a deactivated job stays in the list", async () => {
    for (const j of backend.state.jobs.slice(1)) {
      expect((await call("PATCH", `/jobs/${j.key}`, { version: j.version, is_active: false })).status).toBe(200);
    }
    const last = backend.state.jobs[0];
    const res = await call("PATCH", `/jobs/${last.key}`, { version: last.version, is_active: false });
    expect(res.body.code).toBe("last_active_job");
    expect((await call("GET", "/jobs")).body).toHaveLength(6);
  });
});

describe("settings (§6)", () => {
  it("describes each setting so the form can be generated", async () => {
    const settings = (await call("GET", "/settings")).body;
    expect(settings.find((s: { name: string }) => s.name === "confidence_threshold")).toMatchObject({
      type: "number", min: 0.25, max: 0.8, step: 0.05, confirm: true, value: 0.45, default: 0.45, overridden: false,
    });
    expect(settings.find((s: { name: string }) => s.name === "gemini_min_confidence")).toMatchObject({
      value: 0.8, default: 0.75, overridden: true,
    });
  });

  it("is a merge patch: sets a value, leaves the rest, and null returns to the default", async () => {
    await call("PATCH", "/settings", { confidence_threshold: 0.6 });
    const cleared = await call("PATCH", "/settings", { confidence_threshold: null, gemini_min_confidence: null });
    const byName = Object.fromEntries(cleared.body.settings.map((s: { name: string }) => [s.name, s]));
    expect(byName.confidence_threshold).toMatchObject({ value: 0.45, overridden: false });
    expect(byName.gemini_min_confidence).toMatchObject({ value: 0.75, overridden: false });
  });

  it("is all or nothing: one bad value saves none, and names the setting in `field`", async () => {
    const res = await call("PATCH", "/settings", { gemini_verify: false, confidence_threshold: 0.95 });
    expect(res.status).toBe(422);
    expect(res.body).toMatchObject({ code: "out_of_range", field: "confidence_threshold" });
    const after = (await call("GET", "/settings")).body;
    expect(after.find((s: { name: string }) => s.name === "gemini_verify").overridden).toBe(false);
  });

  it("422 invalid for the wrong type or a choice that isn't offered", async () => {
    expect((await call("PATCH", "/settings", { gemini_verify: "yes" })).body.code).toBe("invalid");
    expect((await call("PATCH", "/settings", { worker_auth: "sometimes" })).body.code).toBe("invalid");
  });

  it("warns, with the setting named, and says when the change applies", async () => {
    const res = await call("PATCH", "/settings", { worker_auth: "required", gemini_verify: false, confidence_threshold: 0.5 });
    expect(res.body.applies_within_s).toBe(30);
    expect(res.body.warnings.map((w: { code: string; setting: string }) => [w.code, w.setting]).sort()).toEqual([
      ["gemini_verify_switched_off", "gemini_verify"],
      ["threshold_changed", "confidence_threshold"],
      ["worker_auth_required", "worker_auth"],
    ]);
  });

  it("always sends `warnings`, empty when there is nothing to say", async () => {
    expect((await call("PATCH", "/settings", { gemini_min_confidence: 0.9 })).body.warnings).toEqual([]);
  });
});

describe("sites (§3)", () => {
  it("shows an admin every site and a supervisor only their own", async () => {
    expect((await call("GET", "/sites")).body.map((s: { key: string }) => s.key)).toEqual(["pune_tower_b", "pune_tower_c", "old_depot"]);
    signedIn = SUPERVISOR_B;
    expect((await call("GET", "/sites")).body.map((s: { key: string }) => s.key)).toEqual(["pune_tower_b"]);
  });

  it("409 site_in_use while a site still has active workers or supervisors", async () => {
    const res = await call("PATCH", "/sites/pune_tower_c", { is_active: false });
    expect(res.status).toBe(409);
    expect(res.body.code).toBe("site_in_use");
  });
});

describe("dashboard users (§7)", () => {
  it("creates a user with a temporary password shown once (never listed afterwards)", async () => {
    const res = await call("POST", "/users", { email: "new@example.com", display_name: "New", role: "supervisor", site: "pune_tower_b" });
    expect(res.status).toBe(201);
    expect(res.body.temporary_password).toMatch(/\w+-\w+-\d\d-\w+/);
    expect(res.body.user).toMatchObject({ must_change_password: true, site: "pune_tower_b" });
    expect(JSON.stringify((await call("GET", "/users")).body)).not.toContain(res.body.temporary_password);
  });

  it("422 for a supervisor without a site, an admin with one, or a deactivated site", async () => {
    const make = (extra: object) => call("POST", "/users", { email: "n@example.com", display_name: "N", ...extra });
    expect((await make({ role: "supervisor" })).body).toMatchObject({ code: "invalid", field: "site" });
    expect((await make({ role: "admin", site: "pune_tower_b" })).body).toMatchObject({ code: "invalid", field: "site" });
    expect((await make({ role: "supervisor", site: "old_depot" })).body).toMatchObject({ code: "invalid", field: "site" });
    expect((await make({ role: "supervisor", site: "nowhere" })).status).toBe(404);
  });

  it("409 cannot_change_self for your own role or active flag", async () => {
    await call("GET", "/users"); // makes the signed-in admin appear
    const me = backend.state.users.find((u) => u.email === ADMIN.email)!;
    expect((await call("PATCH", `/users/${me.id}`, { is_active: false })).body.code).toBe("cannot_change_self");
    expect((await call("POST", `/users/${me.id}/reset-password`)).body.code).toBe("cannot_change_self");
  });

  it("409 last_admin when the change would leave no active admin", async () => {
    await call("GET", "/users"); // makes the signed-in admin appear in the list
    // An admin deactivated a moment ago whose session is still open is the only way to get here:
    // any active actor is itself an active admin.
    backend.state.users.find((u) => u.email === ADMIN.email)!.is_active = false;
    const other = backend.state.users.find((u) => u.email === "second.admin@example.com")!;
    const res = await call("PATCH", `/users/${other.id}`, { is_active: false });
    expect(res.status).toBe(409);
    expect(res.body.code).toBe("last_admin");
  });

  it("lets one admin deactivate another while an active admin remains", async () => {
    const other = backend.state.users.find((u) => u.email === "second.admin@example.com")!;
    expect((await call("PATCH", `/users/${other.id}`, { is_active: false })).status).toBe(200);
  });

  it("clears the site when someone becomes an admin; a supervisor needs one", async () => {
    const res = await call("PATCH", "/users/101", { role: "admin" });
    expect(res.body.user).toMatchObject({ role: "admin", site: null });
    expect((await call("PATCH", "/users/101", { role: "supervisor" })).body).toMatchObject({ code: "invalid", field: "site" });
  });
});

describe("workers (§8)", () => {
  it("pages with a total and an opaque cursor", async () => {
    const first = (await call("GET", "/workers?limit=25")).body;
    expect(first.items).toHaveLength(25);
    expect(first.total).toBe(60);
    expect(first.next_cursor).toEqual(expect.any(String));
    const second = (await call("GET", `/workers?limit=25&cursor=${encodeURIComponent(first.next_cursor)}`)).body;
    expect(second.items[0].worker_id).not.toBe(first.items[0].worker_id);
    const last = (await call("GET", `/workers?limit=25&cursor=${encodeURIComponent(second.next_cursor)}`)).body;
    expect(last.items).toHaveLength(10);
    expect(last.next_cursor).toBeNull();
  });

  it("filters by search (ID or name), site, active and locked", async () => {
    expect((await call("GET", "/workers?search=w-0007")).body.items.map((w: { worker_id: string }) => w.worker_id)).toEqual(["W-0007"]);
    expect((await call("GET", "/workers?site=pune_tower_c")).body.total).toBe(20);
    expect((await call("GET", "/workers?active=false")).body.total).toBe(8);
    const locked = (await call("GET", "/workers?locked=true")).body;
    expect(locked.items.map((w: { worker_id: string }) => w.worker_id)).toEqual(["W-0004", "W-0009"]);
  });

  it("shows a supervisor only their own site, and answers 404 for another site's worker", async () => {
    signedIn = SUPERVISOR_B;
    const list = (await call("GET", "/workers?site=pune_tower_c")).body; // the site filter is ignored for a supervisor
    expect(list.items.every((w: { site: string }) => w.site === "pune_tower_b")).toBe(true);
    expect((await call("GET", "/workers/W-0003")).status).toBe(404); // W-0003 is at tower C
    expect((await call("PATCH", "/workers/W-0003", { display_name: "x" })).status).toBe(404);
    expect((await call("POST", "/workers/W-0003/reset-pin")).status).toBe(404);
  });

  it("creates a worker with a one-time PIN; a supervisor's go to their own site", async () => {
    signedIn = SUPERVISOR_B;
    const res = await call("POST", "/workers", { worker_id: "w-0100", display_name: "New Worker" });
    expect(res.status).toBe(201);
    expect(res.body.worker).toMatchObject({ worker_id: "W-0100", site: "pune_tower_b", must_change_pin: true, created_by: SUPERVISOR_B.email });
    expect(res.body.one_time_pin).toMatch(/^\d{6}$/);
    expect(res.body.one_time_pin).not.toMatch(/^(\d)\1{5}$/);
    expect(res.body.one_time_pin).not.toBe("123456");
  });

  it("403 when a supervisor names another site; 422 when an admin gives none; 422 for a deactivated site", async () => {
    signedIn = SUPERVISOR_B;
    expect((await call("POST", "/workers", { worker_id: "W-0101", display_name: "x", site: "pune_tower_c" })).status).toBe(403);
    signedIn = ADMIN;
    expect((await call("POST", "/workers", { worker_id: "W-0101", display_name: "x" })).body).toMatchObject({ code: "invalid", field: "site" });
    expect((await call("POST", "/workers", { worker_id: "W-0101", display_name: "x", site: "old_depot" })).body).toMatchObject({ code: "invalid", field: "site" });
    expect((await call("POST", "/workers", { worker_id: "W-0101", display_name: "x", site: "nowhere" })).status).toBe(404);
  });

  it("409 already_exists says the ID is taken, not where", async () => {
    signedIn = SUPERVISOR_B;
    const res = await call("POST", "/workers", { worker_id: "W-0003", display_name: "x" }); // exists at tower C
    expect(res.status).toBe(409);
    expect(res.body.code).toBe("already_exists");
    expect(JSON.stringify(res.body)).not.toContain("tower");
  });

  it("only an admin may move a worker; a supervisor sending `site` gets 403", async () => {
    signedIn = SUPERVISOR_B;
    expect((await call("PATCH", "/workers/W-0001", { site: "pune_tower_c" })).status).toBe(403);
    signedIn = ADMIN;
    expect((await call("PATCH", "/workers/W-0001", { site: "pune_tower_c" })).body.worker.site).toBe("pune_tower_c");
  });

  it("reset-pin unlocks the worker and gives a new one-time PIN", async () => {
    const res = await call("POST", "/workers/W-0004/reset-pin");
    expect(res.body.one_time_pin).toMatch(/^\d{6}$/);
    expect((await call("GET", "/workers/W-0004")).body).toMatchObject({ locked_until: null, must_change_pin: true });
  });
});

describe("audit log (§9)", () => {
  it("records every write, newest first, with before/after and the site", async () => {
    await call("PATCH", "/workers/W-0001", { is_active: false });
    await call("PUT", "/jobs/painter/requirements", { version: 2, requirements: [{ item: "safety_vest", mode: "required" }] });
    const entries = (await call("GET", "/audit")).body.items;
    expect(entries[0]).toMatchObject({ action: "job.requirements", target: "job:painter", actor: ADMIN.email });
    expect(entries[1]).toMatchObject({
      action: "worker.update", target: "worker:W-0001", site: "pune_tower_b", before: { is_active: true }, after: { is_active: false },
    });
  });

  it("never records a PIN or password", async () => {
    await call("POST", "/workers", { worker_id: "W-0100", display_name: "x", site: "pune_tower_b" });
    await call("POST", "/workers/W-0100/reset-pin");
    await call("POST", "/users", { email: "n@example.com", display_name: "N", role: "admin" });
    const text = JSON.stringify((await call("GET", "/audit")).body);
    expect(text).not.toMatch(/\b\d{6}\b.*one_time_pin|temporary_password|one_time_pin/);
  });

  it("filters exactly by action, target, actor and site, and by date range (from inclusive, to exclusive)", async () => {
    expect((await call("GET", "/audit?action=worker.create")).body.total).toBe(1);
    expect((await call("GET", "/audit?actor=cli")).body.total).toBe(1);
    expect((await call("GET", "/audit?site=pune_tower_b")).body.items.every((e: { site: string }) => e.site === "pune_tower_b")).toBe(true);
    const all = (await call("GET", "/audit")).body.items;
    const at = all[0].at;
    expect((await call("GET", `/audit?from=${at}`)).body.total).toBe(1);
    expect((await call("GET", `/audit?to=${at}`)).body.total).toBe(all.length - 1);
  });

  it("pages with a total", async () => {
    const page = (await call("GET", "/audit?limit=2")).body;
    expect(page).toMatchObject({ total: 4, next_cursor: expect.any(String) });
    expect(page.items).toHaveLength(2);
  });
});
