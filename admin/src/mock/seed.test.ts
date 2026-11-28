/**
 * The stand-in's jobs are copied by hand from configs/job_matrix.yaml, the
 * file that seeds a new database. They drifted once (three of six), so this
 * reads the yaml and compares: the same jobs in the same order, and the same
 * equipment, head to toe as the server seeds it (docs/admin-api.md §5,
 * decision 9 in §12). The head-to-toe order is read from the server's own
 * catalogue (`ITEMS` in src/db/matrix.py), so the stand-in's can't drift either.
 */
import { existsSync, readFileSync } from "node:fs";
import { dirname, join } from "node:path";

import { describe, expect, it } from "vitest";

import { seedState } from "./seed";

/**
 * configs/job_matrix.yaml, found from wherever the tests were started (`admin/`
 * for `npm test` and CI) by looking upwards. `import.meta.url` isn't a file
 * URL under the jsdom environment, so it can't be used for this.
 */
function findRepoFile(...parts: string[]): string {
  for (let dir = process.cwd(); ; dir = dirname(dir)) {
    const candidate = join(dir, ...parts);
    if (existsSync(candidate)) return candidate;
    if (dirname(dir) === dir) throw new Error(`${parts.join("/")} not found above ${process.cwd()}`);
  }
}

const MATRIX_FILE = findRepoFile("configs", "job_matrix.yaml");

/**
 * The item keys of `ITEMS` in src/db/matrix.py, in order: one `("key", "Name", True),`
 * per line between `ITEMS ... = (` and `)`. Anything else in there fails the test.
 */
function readItemOrder(text: string): string[] {
  const block = /^ITEMS\b[^\n]*= \(\r?\n([\s\S]*?)^\)/m.exec(text);
  if (!block) throw new Error("ITEMS not found in src/db/matrix.py. Update readItemOrder.");
  return block[1]
    .split(/\r?\n/)
    .filter((line) => line.trim() !== "")
    .map((line) => {
      const item = /^ {4}\("([a-z_]+)", "[^"]*", (True|False)\),$/.exec(line.trimEnd());
      if (!item) throw new Error(`src/db/matrix.py ITEMS has a line this test can't read: "${line}". Update readItemOrder.`);
      return item[1];
    });
}

const ITEM_ORDER = readItemOrder(readFileSync(findRepoFile("src", "db", "matrix.py"), "utf-8"));

interface YamlJob {
  required: string[];
  pending: string[];
}

/**
 * Reads the jobs from job_matrix.yaml. Only the shape that file has today is
 * understood (`jobs:`, then a key per job, each with `required: [...]` and
 * `pending: [...]` on one line), to keep a yaml library out of the dashboard.
 * Any other line under `jobs:` fails the test and says which, so a change of
 * layout is noticed instead of read as "no jobs".
 */
function readJobMatrix(text: string): Map<string, YamlJob> {
  const jobs = new Map<string, YamlJob>();
  let inJobs = false;
  let current: YamlJob | null = null;

  for (const raw of text.split(/\r?\n/)) {
    const line = raw.replace(/#.*$/, "").trimEnd();
    if (line.trim() === "") continue;
    if (line === "jobs:") {
      inJobs = true;
      continue;
    }
    if (/^\S/.test(line)) {
      inJobs = false; // another top-level key
      continue;
    }
    if (!inJobs) continue;

    const job = /^ {2}([a-z_]+):$/.exec(line);
    if (job) {
      current = { required: [], pending: [] };
      jobs.set(job[1], current);
      continue;
    }
    const list = /^ {4}(required|pending): \[(.*)\]$/.exec(line);
    if (list && current) {
      current[list[1] as keyof YamlJob] = list[2].split(",").map((s) => s.trim()).filter(Boolean);
      continue;
    }
    throw new Error(`configs/job_matrix.yaml has a line this test can't read: "${raw}". Update readJobMatrix.`);
  }
  return jobs;
}

describe("the stand-in's sample jobs match configs/job_matrix.yaml", () => {
  const yaml = readJobMatrix(readFileSync(MATRIX_FILE, "utf-8"));
  const seed = seedState();

  it("reads the file (a reader that found nothing would pass everything below)", () => {
    expect(yaml.size).toBeGreaterThanOrEqual(6);
    for (const job of yaml.values()) expect(job.required.length).toBeGreaterThan(0);
  });

  it("has the same jobs, in the same order", () => {
    expect(seed.jobs.map((j) => j.key)).toEqual([...yaml.keys()]);
  });

  it("lists the server's items, in the server's head-to-toe order", () => {
    expect(ITEM_ORDER.length).toBeGreaterThanOrEqual(9);
    expect(seed.items.map((i) => i.key)).toEqual(ITEM_ORDER);
  });

  it.each([...readJobMatrix(readFileSync(MATRIX_FILE, "utf-8")).keys()])(
    "%s: the same required and pending items, head to toe as the server seeds them",
    (key) => {
      // seed_from_yaml: required then pending, sorted together by the items' order.
      const expected = [
        ...yaml.get(key)!.required.map((item) => ({ item, mode: "required" })),
        ...yaml.get(key)!.pending.map((item) => ({ item, mode: "pending" })),
      ].sort((a, b) => ITEM_ORDER.indexOf(a.item) - ITEM_ORDER.indexOf(b.item));
      expect(seed.jobs.find((j) => j.key === key)?.requirements).toEqual(expected);
    },
  );

  it("names only items the stand-in's catalogue has", () => {
    const known = new Set(seed.items.map((i) => i.key));
    for (const job of yaml.values()) {
      for (const item of [...job.required, ...job.pending]) expect(known).toContain(item);
    }
  });
});
