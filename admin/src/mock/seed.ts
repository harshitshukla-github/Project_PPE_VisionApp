/**
 * Starting data for the mock backend. The six jobs are copied by hand from
 * configs/job_matrix.yaml, the file that seeds a new database: the same
 * equipment in the same order. `seed.test.ts` reads that file and fails if
 * they differ, so if it changes, change these to match. The AP50 figures,
 * versions, sites, people and audit entries are made up so every page has
 * something to show.
 */
import type { AdminUser, AuditEntry, Check, Detection, Item, Job, Requirement, Verdict, Worker } from "../api/types";

export interface SeedSite {
  key: string;
  display_name: string;
  is_active: boolean;
  created_at: string;
}

export interface MockState {
  sites: SeedSite[];
  items: Item[];
  jobs: Job[];
  /** A saved override per setting name; absent means "the .env default". */
  settings: Record<string, { value: number | boolean | string; updated_by: string; updated_at: string }>;
  users: AdminUser[];
  workers: Worker[];
  audit: AuditEntry[];
  /** Stored checks, as docs/checks-api.md §2 describes them. */
  checks: Check[];
  nextUserId: number;
  nextAuditId: number;
}

// Where each item sits on a 1080×1440 portrait photo of a worker, for the sample boxes.
const PHOTO = { width: 1080, height: 1440 };
const BOXES: Record<string, [number, number, number, number]> = {
  person: [210, 120, 870, 1420],
  hardhat: [400, 120, 680, 310],
  no_hardhat: [420, 150, 660, 360],
  goggles: [440, 300, 640, 370],
  face_shield: [420, 280, 660, 470],
  mask: [470, 380, 610, 460],
  ear_protection: [390, 280, 690, 380],
  safety_vest: [320, 520, 760, 960],
  no_vest: [330, 520, 750, 950],
  harness: [330, 500, 750, 1000],
  gloves: [230, 900, 370, 1040],
  boots: [360, 1300, 720, 1420],
};

/**
 * About 45 sample checks over the last ten days: mostly PASS, some FAIL and
 * NEEDS_REVIEW, a few already reviewed, a few sent without a login under an
 * ID nobody has (no site, no photo), and some photos deleted. Deterministic,
 * so tests can count on them.
 */
function seedChecks(now: Date, jobs: Job[], workers: Worker[], items: Item[]): Check[] {
  const iso = (hoursAgo: number) => new Date(now.getTime() - hoursAgo * 3_600_000).toISOString();
  const detectable = new Set(items.filter((i) => i.detectable).map((i) => i.key));
  const checks: Check[] = [];
  for (let i = 0; i < 45; i++) {
    const id = 1000 + i;
    const job = jobs[i % jobs.length];
    const signedIn = i % 15 !== 7; // three checks without a login
    const worker = workers[(i * 7) % workers.length];
    const required = job.requirements.filter((r) => r.mode === "required").map((r) => r.item);
    const pending = job.requirements.filter((r) => r.mode === "pending").map((r) => r.item);
    const verdict: Verdict = i % 5 === 1 ? "NEEDS_REVIEW" : i % 9 === 4 ? "FAIL" : "PASS";

    // NEEDS_REVIEW: one required, detectable item wasn't found. FAIL: a violation class was.
    const missing = verdict === "NEEDS_REVIEW" ? [required.filter((k) => detectable.has(k)).at(-1)!] : [];
    const violations = verdict === "FAIL" ? ["no_hardhat"] : [];
    const found = required.filter((k) => detectable.has(k) && !missing.includes(k) && !(verdict === "FAIL" && k === "hardhat"));
    const detections: Detection[] = ["person", ...found, ...violations].map((k, n) => ({
      class: k,
      confidence: Math.round((0.93 - n * 0.04) * 100) / 100,
      bbox: BOXES[k] ?? BOXES.person,
    }));

    const reviewed = verdict === "NEEDS_REVIEW" && i % 3 === 0;
    const supervisor = worker.site === "pune_tower_b" ? "supervisor.b@example.com" : "supervisor.c@example.com";
    const photoState = !signedIn ? "not_stored" : i >= 38 ? "deleted" : "kept";
    checks.push({
      id,
      at: iso(4 + i * 5),
      worker_id: signedIn ? worker.worker_id : `X-${900 + i}`,
      worker_name: signedIn ? worker.display_name : null,
      site: signedIn ? worker.site : null,
      signed_in: signedIn,
      job: job.key,
      job_name: job.display_name,
      job_version: job.version,
      required,
      pending,
      verdict,
      verdict_source: verdict === "NEEDS_REVIEW" ? "detector+gemini" : "detector",
      missing_equipment: verdict === "FAIL" ? ["hardhat"] : missing,
      violations_detected: violations,
      unverifiable_equipment: pending,
      explanation:
        verdict === "FAIL"
          ? `The worker isn't wearing a hardhat, which ${job.display_name.toLowerCase()} work requires against falling objects.`
          : null,
      model_used: "yolo11l",
      confidence_threshold: 0.45,
      gemini_verify: true,
      gemini_min_confidence: i < 20 ? 0.8 : 0.75,
      photo:
        photoState === "not_stored"
          ? { state: "not_stored", width: null, height: null, bytes: null, type: null }
          : { state: photoState, width: PHOTO.width, height: PHOTO.height, bytes: 380_000 + i * 1_013, type: "image/jpeg" },
      training_consent: signedIn && i % 4 === 0,
      review: reviewed
        ? { outcome: i % 2 === 0 ? "compliant" : "not_compliant", note: "Checked with the worker on site.", by: supervisor, at: iso(2 + i * 5) }
        : null,
      detections,
      verification:
        verdict === "NEEDS_REVIEW"
          ? {
              model: "gemini-2.5-flash",
              items: missing.map((item) => ({
                item,
                status: "uncertain" as const,
                confidence: 0.55,
                reason: "The area where it would be is out of frame.",
              })),
              elapsed_ms: 2140,
            }
          : null,
    });
  }
  return checks;
}

const req = (item: string, mode: Requirement["mode"] = "required"): Requirement => ({ item, mode });

const WORKER_NAMES = [
  "Asha Verma", "Ravi Kumar", "Meera Iyer", "Sanjay Patil", "Divya Nair", "Imran Shaikh",
  "Kavita Joshi", "Arjun Rao", "Nisha Gupta", "Vikram Singh", "Pooja Desai", "Rahul Mehta",
];

export function seedState(now: Date = new Date()): MockState {
  const iso = (daysAgo: number) => new Date(now.getTime() - daysAgo * 86_400_000).toISOString();

  const sites: SeedSite[] = [
    { key: "pune_tower_b", display_name: "Pune Tower B", is_active: true, created_at: iso(30) },
    { key: "pune_tower_c", display_name: "Pune Tower C", is_active: true, created_at: iso(20) },
    { key: "old_depot", display_name: "Old Depot", is_active: false, created_at: iso(90) },
  ];

  // The real catalogue (`ITEMS` in src/db/matrix.py, docs/admin-api.md §4): the same items, in
  // the same head-to-toe order, with the same ones the detector can't see (Face shield, Ear
  // protection and Harness are Gemini-only, so no AP50). Only the AP50 figures are made up.
  const items: Item[] = [
    { key: "hardhat", display_name: "Hardhat", detectable: true, detector_ap50: 0.909 },
    { key: "goggles", display_name: "Goggles", detectable: true, detector_ap50: 0.713 },
    { key: "face_shield", display_name: "Face shield", detectable: false, detector_ap50: null },
    { key: "ear_protection", display_name: "Ear protection", detectable: false, detector_ap50: null },
    { key: "mask", display_name: "Mask", detectable: true, detector_ap50: 0.781 },
    { key: "safety_vest", display_name: "Safety vest", detectable: true, detector_ap50: 0.874 },
    { key: "harness", display_name: "Harness", detectable: false, detector_ap50: null },
    { key: "gloves", display_name: "Gloves", detectable: true, detector_ap50: 0.652 },
    { key: "boots", display_name: "Boots", detectable: true, detector_ap50: 0.598 },
  ];

  const jobRows: Array<[string, string, number, Requirement[]]> = [
    ["general_labor", "General labor", 2, [req("hardhat"), req("safety_vest"), req("boots")]],
    ["welder", "Welder", 4, [req("hardhat"), req("safety_vest"), req("goggles"), req("gloves"), req("boots"), req("face_shield", "pending")]],
    ["electrician", "Electrician", 1, [req("hardhat"), req("safety_vest"), req("gloves"), req("goggles"), req("boots")]],
    ["height_work", "Height work", 3, [req("hardhat"), req("safety_vest"), req("boots"), req("harness", "pending")]],
    ["demolition", "Demolition", 1, [req("hardhat"), req("safety_vest"), req("goggles"), req("mask"), req("gloves"), req("boots"), req("ear_protection", "pending")]],
    ["painter", "Painter", 2, [req("hardhat"), req("safety_vest"), req("mask"), req("goggles")]],
  ];
  const jobs: Job[] = jobRows.map(([key, display_name, version, requirements], position) => ({
    key,
    display_name,
    position,
    is_active: true,
    version,
    requirements,
    updated_at: iso(7 - position),
  }));

  const users: AdminUser[] = [
    {
      id: 101, email: "supervisor.b@example.com", display_name: "Site Supervisor B", role: "supervisor",
      site: "pune_tower_b", is_active: true, must_change_password: true, locked_until: null,
      last_login_at: null, created_at: iso(14),
    },
    {
      id: 102, email: "supervisor.c@example.com", display_name: "Site Supervisor C", role: "supervisor",
      site: "pune_tower_c", is_active: true, must_change_password: false, locked_until: null,
      last_login_at: iso(1), created_at: iso(14),
    },
    {
      id: 103, email: "second.admin@example.com", display_name: "Second Admin", role: "admin",
      site: null, is_active: true, must_change_password: false, locked_until: null,
      last_login_at: iso(2), created_at: iso(40),
    },
  ];

  // 60 workers, so the default page size of 50 shows paging. Every 7th is
  // deactivated; two are locked out right now.
  const workers: Worker[] = Array.from({ length: 60 }, (_, i) => {
    const n = i + 1;
    const site = n % 3 === 0 ? "pune_tower_c" : "pune_tower_b";
    return {
      worker_id: `W-${String(n).padStart(4, "0")}`,
      display_name: `${WORKER_NAMES[i % WORKER_NAMES.length]}${i >= WORKER_NAMES.length ? ` ${Math.floor(i / WORKER_NAMES.length) + 1}` : ""}`,
      site,
      is_active: n % 7 !== 0,
      must_change_pin: n % 11 === 0,
      locked_until: n === 4 || n === 9 ? new Date(now.getTime() + 12 * 60_000).toISOString() : null,
      last_login_at: n % 5 === 0 ? null : iso(n % 6),
      created_at: iso(30 - (n % 25)),
      created_by: n % 10 === 0 ? null : site === "pune_tower_b" ? "supervisor.b@example.com" : "supervisor.c@example.com",
    };
  });

  const audit: AuditEntry[] = [
    {
      id: 300, at: iso(6), actor: "cli", action: "worker.create", target: "worker:W-0001", site: "pune_tower_b",
      before: null, after: { worker_id: "W-0001", display_name: "Asha Verma", site: "pune_tower_b" }, ip: null,
    },
    {
      id: 301, at: iso(5), actor: "second.admin@example.com", action: "job.requirements", target: "job:welder",
      site: null,
      before: { requirements: [req("hardhat"), req("safety_vest"), req("goggles")] },
      after: { requirements: jobRows[1][3] }, ip: "192.168.29.20",
    },
    {
      id: 302, at: iso(3), actor: "second.admin@example.com", action: "settings.update", target: "setting:gemini_min_confidence",
      site: null, before: { value: null }, after: { value: 0.8 }, ip: "192.168.29.20",
    },
    {
      id: 303, at: iso(1), actor: "supervisor.b@example.com", action: "worker.reset_pin", target: "worker:W-0004",
      site: "pune_tower_b", before: null, after: null, ip: "192.168.29.31",
    },
  ];

  return {
    sites,
    items,
    jobs,
    settings: {
      gemini_min_confidence: { value: 0.8, updated_by: "second.admin@example.com", updated_at: iso(3) },
    },
    users,
    workers,
    audit,
    checks: seedChecks(now, jobs, workers, items),
    nextUserId: 200,
    nextAuditId: 304,
  };
}
