/**
 * The stand-in's jobs are copied by hand from configs/job_matrix.yaml, the
 * file that seeds a new database. They drifted once (three of six), so this
 * reads the yaml and compares: the same jobs in the same order, the same
 * required equipment in the same order, then the pending items.
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
function findMatrixFile(): string {
  for (let dir = process.cwd(); ; dir = dirname(dir)) {
    const candidate = join(dir, "configs", "job_matrix.yaml");
    if (existsSync(candidate)) return candidate;
    if (dirname(dir) === dir) throw new Error(`configs/job_matrix.yaml not found above ${process.cwd()}`);
  }
}

const MATRIX_FILE = findMatrixFile();

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

  it.each([...readJobMatrix(readFileSync(MATRIX_FILE, "utf-8")).keys()])(
    "%s: the same required items in the same order, then the same pending ones",
    (key) => {
      const expected = [
        ...yaml.get(key)!.required.map((item) => ({ item, mode: "required" })),
        ...yaml.get(key)!.pending.map((item) => ({ item, mode: "pending" })),
      ];
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
