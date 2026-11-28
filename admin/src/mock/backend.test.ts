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
  it("lists the items head to toe, as the real backend does (docs/admin-api.md §4)", async () => {
    const items = (await call("GET", "/items")).body;

    expect(items.map((i: { key: string }) => i.key)).toEqual([
      "hardhat", "goggles", "face_shield", "ear_protection", "mask", "safety_vest", "harness", "gloves", "boots",
    ]);
  });

  it("has the same items Gemini-only as the real catalogue: Face shield, Ear protection and Harness", async () => {
    const items = (await call("GET", "/items")).body;

    const geminiOnly = items.filter((i: { detectable: boolean }) => !i.detectable).map((i: { key: string }) => i.key);
    expect(geminiOnly).toEqual(["face_shield", "ear_protection", "harness"]);
    // One direction only (docs/admin-api.md §4): an item the detector can't see has no accuracy figure.
    // The other way round isn't a rule: a detectable item may have none too, when the loaded model has no metrics file.
    for (const item of items) if (!item.detectable) expect(item.detector_ap50).toBeNull();
  });

  it("warns when Ear protection is made required, since only Gemini can judge it", async () => {
    const res = await call("PUT", "/jobs/painter/requirements", {
      version: 2,
      requirements: [{ item: "safety_vest", mode: "required" }, { item: "ear_protection", mode: "required" }],
    });

    expect(res.body.warnings).toEqual([expect.objectContaining({ code: "gemini_only_item", item: "ear_protection" })]);
  });

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

  describe("equipment head to toe, as the server keeps it (§5, decision 9 in §12)", () => {
    const items = (list: Array<{ item: string }>) => list.map((r) => r.item);

    it("a job ticked boots first is saved hardhat first, in the job and in its audit entry", async () => {
      const created = await call("POST", "/jobs", {
        key: "rigger", display_name: "Rigger",
        requirements: [{ item: "boots", mode: "required" }, { item: "harness", mode: "pending" }, { item: "hardhat", mode: "required" }],
      });
      expect(created.status).toBe(201);
      expect(items(created.body.job.requirements)).toEqual(["hardhat", "harness", "boots"]);
      const entry = backend.state.audit.find((e) => e.action === "job.create" && e.target === "job:rigger")!;
      expect(items((entry.after as { requirements: Array<{ item: string }> }).requirements)).toEqual(["hardhat", "harness", "boots"]);
    });

    it("an item switched on takes its place, not the end", async () => {
      const res = await call("PUT", "/jobs/general_labor/requirements", {
        version: 2,
        requirements: [...job("general_labor").requirements, { item: "goggles", mode: "required" }],
      });
      expect(items(res.body.job.requirements)).toEqual(["hardhat", "goggles", "safety_vest", "boots"]);
    });

    it("the same items in another order are no change: same version, no audit entry", async () => {
      const auditBefore = backend.state.audit.length;
      const res = await call("PUT", "/jobs/general_labor/requirements", {
        version: 2, requirements: [...job("general_labor").requirements].reverse(),
      });
      expect(res.status).toBe(200);
      expect(res.body.job.version).toBe(2);
      expect(items(res.body.job.requirements)).toEqual(["hardhat", "safety_vest", "boots"]);
      expect(backend.state.audit.length).toBe(auditBefore);
    });

    it("a 422 names the entry by its place in the list as sent, not as sorted", async () => {
      const res = await call("PUT", "/jobs/painter/requirements", {
        version: 2,
        requirements: [{ item: "boots", mode: "required" }, { item: "hardhat", mode: "required" }, { item: "boots", mode: "pending" }],
      });
      expect(res.body).toMatchObject({ code: "invalid", field: "requirements[2].item" });
    });
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

describe("checks (docs/checks-api.md)", () => {
  const check = (id: number) => backend.state.checks.find((c) => c.id === id)!;
  /** A NEEDS_REVIEW check nobody has reviewed yet. */
  const unreviewed = () => backend.state.checks.find((c) => c.verdict === "NEEDS_REVIEW" && !c.review)!;

  describe("the list (§4.1)", () => {
    it("is newest first, paged, with a total, and leaves out detections and verification", async () => {
      const page = (await call("GET", "/checks?limit=10")).body;
      expect(page.total).toBe(backend.state.checks.length);
      expect(page.items).toHaveLength(10);
      expect(page.next_cursor).toEqual(expect.any(String));
      const times = page.items.map((c: { at: string }) => c.at);
      expect([...times].sort().reverse()).toEqual(times);
      expect(page.items[0]).not.toHaveProperty("detections");
      expect(page.items[0]).not.toHaveProperty("verification");
      expect(page.items[0]).toHaveProperty("photo");
    });

    it("gives a supervisor only their own site's checks, whatever `site` asks for, and never one without a site", async () => {
      signedIn = SUPERVISOR_B;
      const all = (await call("GET", "/checks?limit=200")).body;
      expect(all.items.length).toBeGreaterThan(0);
      expect(all.items.every((c: { site: string }) => c.site === "pune_tower_b")).toBe(true);
      expect((await call("GET", "/checks?site=pune_tower_c&limit=200")).body.total).toBe(all.total);
    });

    it("filters by verdict, review, login, job, worker (any case), site and dates", async () => {
      const waiting = (await call("GET", "/checks?verdict=NEEDS_REVIEW&reviewed=false&limit=200")).body;
      expect(waiting.items.every((c: { verdict: string; review: unknown }) => c.verdict === "NEEDS_REVIEW" && c.review === null)).toBe(true);
      expect(waiting.total).toBe(backend.state.checks.filter((c) => c.verdict === "NEEDS_REVIEW" && !c.review).length);

      const noLogin = (await call("GET", "/checks?signed_in=false&limit=200")).body;
      expect(noLogin.total).toBe(3);
      expect(noLogin.items.every((c: { signed_in: boolean; site: null }) => !c.signed_in && c.site === null)).toBe(true);

      const someone = backend.state.checks.find((c) => c.signed_in)!;
      const theirs = (await call("GET", `/checks?worker=${someone.worker_id.toLowerCase()}&limit=200`)).body;
      expect(theirs.total).toBeGreaterThan(0);
      expect(theirs.items.every((c: { worker_id: string }) => c.worker_id === someone.worker_id)).toBe(true);

      expect((await call("GET", "/checks?job=painter&limit=200")).body.items.every((c: { job: string }) => c.job === "painter")).toBe(true);
      expect((await call("GET", "/checks?site=pune_tower_c&limit=200")).body.items.every((c: { site: string }) => c.site === "pune_tower_c")).toBe(true);

      const newest = backend.state.checks.map((c) => c.at).sort().at(-1)!;
      expect((await call("GET", `/checks?from=${newest}`)).body.total).toBe(1);
      expect((await call("GET", `/checks?to=${newest}&limit=200`)).body.total).toBe(backend.state.checks.length - 1);
    });

    it("refuses an unknown verdict", async () => {
      expect((await call("GET", "/checks?verdict=MAYBE")).body).toMatchObject({ code: "invalid", field: "verdict" });
    });
  });

  describe("one check and its photo (§4.2)", () => {
    it("gives the whole check, with detections", async () => {
      const res = await call("GET", "/checks/1001");
      expect(res.status).toBe(200);
      expect(res.body.detections.length).toBeGreaterThan(0);
      expect(res.body).toMatchObject({ id: 1001, gemini_min_confidence: expect.any(Number), required: expect.any(Array) });
    });

    it("answers 404 not_found to a supervisor for another site's check, or one without a site", async () => {
      signedIn = SUPERVISOR_B;
      const elsewhere = backend.state.checks.find((c) => c.site === "pune_tower_c")!;
      const noSite = backend.state.checks.find((c) => c.site === null)!;
      expect((await call("GET", `/checks/${elsewhere.id}`)).body.code).toBe("not_found");
      expect((await call("GET", `/checks/${noSite.id}`)).body.code).toBe("not_found");
      expect((await call("GET", `/checks/${elsewhere.id}/photo`)).status).toBe(404);
    });

    it("serves a kept photo as an image that isn't cached; 404 for one never stored, 410 for one deleted", async () => {
      const kept = backend.state.checks.find((c) => c.photo.state === "kept")!;
      const response = await backend.fetch(`/api/v1/admin/checks/${kept.id}/photo`);
      expect(response.status).toBe(200);
      expect(response.headers.get("Content-Type")).toMatch(/^image\//);
      expect(response.headers.get("Cache-Control")).toBe("private, no-store");

      const never = backend.state.checks.find((c) => c.photo.state === "not_stored")!;
      expect(await call("GET", `/checks/${never.id}/photo`)).toMatchObject({ status: 404, body: { code: "not_found" } });
      const deleted = backend.state.checks.find((c) => c.photo.state === "deleted")!;
      expect(await call("GET", `/checks/${deleted.id}/photo`)).toMatchObject({ status: 410, body: { code: "photo_deleted" } });
    });
  });

  describe("reviewing (§4.3)", () => {
    it("records a review beside the verdict, which doesn't change, and audits it", async () => {
      const c = unreviewed();
      const res = await call("POST", `/checks/${c.id}/review`, { outcome: "compliant", note: "  Gloves in hand.  " });

      expect(res.status).toBe(200);
      expect(res.body.check.review).toMatchObject({ outcome: "compliant", note: "Gloves in hand.", by: ADMIN.email });
      expect(res.body.check.verdict).toBe("NEEDS_REVIEW");
      expect(backend.state.audit.at(-1)).toMatchObject({
        action: "check.review",
        target: `check:${c.id}`,
        before: null,
        after: { outcome: "compliant", note: "Gloves in hand." },
      });
    });

    it("accepts any verdict: a wrong FAIL can be overturned, a wrong PASS flagged", async () => {
      const fail = backend.state.checks.find((c) => c.verdict === "FAIL" && c.site === "pune_tower_b")!;
      const pass = backend.state.checks.find((c) => c.verdict === "PASS" && c.site === "pune_tower_b")!;
      signedIn = SUPERVISOR_B;
      expect((await call("POST", `/checks/${fail.id}/review`, { outcome: "compliant" })).status).toBe(200);
      expect((await call("POST", `/checks/${pass.id}/review`, { outcome: "not_compliant" })).status).toBe(200);
    });

    it("answers a second review 409 already_reviewed with the check as it is now, so nobody overwrites anybody", async () => {
      const c = unreviewed();
      await call("POST", `/checks/${c.id}/review`, { outcome: "compliant" });
      const res = await call("POST", `/checks/${c.id}/review`, { outcome: "not_compliant" });
      expect(res.status).toBe(409);
      expect(res.body).toMatchObject({ code: "already_reviewed", current: { id: c.id, review: { outcome: "compliant" } } });
      expect(check(c.id).review!.outcome).toBe("compliant");
    });

    it("lets an admin replace a review, audited with the replaced one in `before`", async () => {
      const reviewed = backend.state.checks.find((c) => c.review)!;
      const was = reviewed.review!;
      const res = await call("POST", `/checks/${reviewed.id}/review`, { outcome: "not_compliant", replace: true });
      expect(res.status).toBe(200);
      expect(check(reviewed.id).review).toMatchObject({ outcome: "not_compliant", by: ADMIN.email });
      expect(backend.state.audit.at(-1)).toMatchObject({
        before: { outcome: was.outcome, note: was.note },
        after: { outcome: "not_compliant" },
      });
    });

    it("refuses `replace` from a supervisor: 422 invalid", async () => {
      const reviewed = backend.state.checks.find((c) => c.review && c.site === "pune_tower_b")!;
      signedIn = SUPERVISOR_B;
      expect((await call("POST", `/checks/${reviewed.id}/review`, { outcome: "compliant", replace: true })).body).toMatchObject({
        code: "invalid",
        field: "replace",
      });
    });

    it("refuses an unknown outcome and a note over 500 characters", async () => {
      const c = unreviewed();
      expect((await call("POST", `/checks/${c.id}/review`, { outcome: "maybe" })).body).toMatchObject({ code: "invalid", field: "outcome" });
      expect((await call("POST", `/checks/${c.id}/review`, { outcome: "compliant", note: "x".repeat(501) })).body).toMatchObject({
        code: "invalid",
        field: "note",
      });
      expect(check(c.id).review).toBeNull();
    });
  });

  describe("deleting a worker's photos (§4.4)", () => {
    const withPhotos = () => backend.state.checks.find((c) => c.photo.state === "kept" && c.signed_in)!.worker_id;

    it("is for admins only", async () => {
      signedIn = SUPERVISOR_B;
      expect((await call("DELETE", `/workers/${withPhotos()}/photos`)).status).toBe(403);
    });

    it("deletes every kept photo, keeps the checks, and audits the count", async () => {
      const worker = withPhotos();
      const kept = backend.state.checks.filter((c) => c.worker_id === worker && c.photo.state === "kept").length;
      const checks = backend.state.checks.filter((c) => c.worker_id === worker).length;

      const res = await call("DELETE", `/workers/${worker}/photos`);

      expect(res.body).toEqual({ deleted: kept });
      expect(backend.state.checks.filter((c) => c.worker_id === worker)).toHaveLength(checks);
      expect(backend.state.checks.some((c) => c.worker_id === worker && c.photo.state === "kept")).toBe(false);
      expect(backend.state.audit.at(-1)).toMatchObject({
        action: "worker.photos_delete",
        target: `worker:${worker}`,
        after: { deleted: kept },
      });
    });

    it("answers 404 for a worker who doesn't exist", async () => {
      expect((await call("DELETE", "/workers/NOBODY/photos")).status).toBe(404);
    });

    it("stops at a file it can't delete with a plain 500; the ones before stay deleted and audited, and again deletes the rest", async () => {
      // The sample data gives each worker one check; give this worker a second stored photo.
      const [first, second] = backend.state.checks.filter((c) => c.photo.state === "kept" && c.signed_in);
      second.worker_id = first.worker_id;
      const worker = first.worker_id;
      const kept = () => backend.state.checks.filter((c) => c.worker_id === worker && c.photo.state === "kept").length;
      const before = kept();
      expect(before).toBe(2);
      backend.state.photoFileFailsAfter = 1;

      const response = await backend.fetch(`/api/v1/admin/workers/${worker}/photos`, { method: "DELETE" });

      expect(response.status).toBe(500);
      expect(response.headers.get("Content-Type")).toBe("text/plain"); // not JSON
      expect(kept()).toBe(before - 1);
      expect(backend.state.audit.at(-1)).toMatchObject({ action: "worker.photos_delete", after: { deleted: 1 } });

      expect((await call("DELETE", `/workers/${worker}/photos`)).body).toEqual({ deleted: before - 1 });
      expect(kept()).toBe(0);
    });
  });

  describe("photo settings (§7)", () => {
    it("lists store_photos and the two retention periods, with their defaults", async () => {
      const settings = (await call("GET", "/settings")).body as Array<{ name: string }>;
      expect(settings.find((s) => s.name === "store_photos")).toMatchObject({ default: true });
      expect(settings.find((s) => s.name === "photo_review_days")).toMatchObject({ default: 90, min: 7, max: 3650 });
      expect(settings.find((s) => s.name === "photo_training_days")).toMatchObject({ default: 730, min: 30, max: 3650 });
    });

    it.each([
      [{ photo_training_days: 60 }, "photo_training_days"], // shorter than the default 90 for review
      [{ photo_review_days: 800 }, "photo_review_days"], // longer than the default 730 for training
      [{ photo_review_days: 100, photo_training_days: 90 }, "photo_training_days"],
    ])("refuses %j: training is never shorter than review (422 out_of_range on %s), and saves nothing", async (patch, field) => {
      const res = await call("PATCH", "/settings", patch);
      expect(res.status).toBe(422);
      expect(res.body).toMatchObject({ code: "out_of_range", field });
      expect(backend.state.settings).not.toHaveProperty("photo_review_days");
      expect(backend.state.settings).not.toHaveProperty("photo_training_days");
    });

    it("refuses a period that isn't a whole number of days", async () => {
      expect((await call("PATCH", "/settings", { photo_review_days: 30.5 })).body).toMatchObject({
        code: "invalid",
        field: "photo_review_days",
      });
    });

    it("warns retention_changed, with how many photos the next daily run deletes", async () => {
      const res = await call("PATCH", "/settings", { photo_review_days: 7 });
      expect(res.status).toBe(200);
      const warning = res.body.warnings.find((w: { code: string }) => w.code === "retention_changed");
      expect(warning).toMatchObject({
        setting: "photo_review_days",
        count: expect.any(Number),
        message: expect.stringMatching(/will delete \d+ photos?/),
      });
      expect(warning.message).toContain(String(warning.count));
    });

    it("warns once per period changed", async () => {
      const res = await call("PATCH", "/settings", { photo_review_days: 30, photo_training_days: 400 });
      const settings = res.body.warnings.filter((w: { code: string }) => w.code === "retention_changed").map((w: { setting: string }) => w.setting);
      expect(settings.sort()).toEqual(["photo_review_days", "photo_training_days"]);
    });

    it("stores a note that is only spaces as null (§4.3)", async () => {
      const c = backend.state.checks.find((x) => x.verdict === "NEEDS_REVIEW" && !x.review)!;
      expect((await call("POST", `/checks/${c.id}/review`, { outcome: "compliant", note: "   " })).body.check.review.note).toBeNull();
    });
  });
});
