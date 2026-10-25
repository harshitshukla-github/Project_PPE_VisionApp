/**
 * Starting data for the mock backend. Illustrative, not real: the jobs echo
 * the live matrix, but the AP50 figures, sites, people and audit entries are
 * made up so every page has something to show.
 */
import type { AdminUser, AuditEntry, Item, Job, Requirement, Worker } from "../api/types";

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
  nextUserId: number;
  nextAuditId: number;
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

  const items: Item[] = [
    { key: "hardhat", display_name: "Hardhat", detectable: true, detector_ap50: 0.909 },
    { key: "safety_vest", display_name: "Safety vest", detectable: true, detector_ap50: 0.874 },
    { key: "goggles", display_name: "Goggles", detectable: true, detector_ap50: 0.713 },
    { key: "mask", display_name: "Mask", detectable: true, detector_ap50: 0.781 },
    { key: "gloves", display_name: "Gloves", detectable: true, detector_ap50: 0.652 },
    { key: "boots", display_name: "Boots", detectable: true, detector_ap50: 0.598 },
    { key: "ear_protection", display_name: "Ear protection", detectable: true, detector_ap50: 0.544 },
    { key: "face_shield", display_name: "Face shield", detectable: false, detector_ap50: null },
    { key: "harness", display_name: "Harness", detectable: false, detector_ap50: null },
  ];

  const jobRows: Array<[string, string, number, Requirement[]]> = [
    ["general_labor", "General labor", 2, [req("hardhat"), req("safety_vest"), req("boots")]],
    ["welder", "Welder", 4, [req("hardhat"), req("safety_vest"), req("goggles"), req("gloves"), req("boots"), req("face_shield", "pending")]],
    ["electrician", "Electrician", 1, [req("hardhat"), req("safety_vest"), req("gloves"), req("boots")]],
    ["height_work", "Height work", 3, [req("hardhat"), req("safety_vest"), req("boots"), req("harness", "pending")]],
    ["demolition", "Demolition", 1, [req("hardhat"), req("safety_vest"), req("mask"), req("gloves"), req("boots"), req("ear_protection", "pending")]],
    ["painter", "Painter", 2, [req("safety_vest"), req("mask"), req("gloves")]],
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
    nextUserId: 200,
    nextAuditId: 304,
  };
}
