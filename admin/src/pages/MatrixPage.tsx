import { useState } from "react";

import { ApiError, getItems, getJobs, putRequirements, type DashboardUser } from "../api/client";
import { problemFrom, type Problem } from "../api/problems";
import type { Item, Job, Requirement, RequirementMode, Warning } from "../api/types";
import { ProblemBanner } from "../components/ProblemBanner";
import { Loading, Tag } from "../components/widgets";
import { useResource } from "../hooks/useResource";

/** What a cell shows. "Manual" is the contract's `pending`: listed for the worker as "verify manually", never scored. */
type Mode = RequirementMode | "off";

const MODE_LABEL: Record<Mode, string> = { required: "Required", pending: "Manual", off: "Off" };

function modeOf(requirements: Requirement[], item: string): Mode {
  return requirements.find((r) => r.item === item)?.mode ?? "off";
}

/** Sets one cell. An item already in the list keeps its place; a new one goes last. */
function withMode(requirements: Requirement[], item: string, mode: Mode): Requirement[] {
  if (mode === "off") return requirements.filter((r) => r.item !== item);
  if (requirements.some((r) => r.item === item)) {
    return requirements.map((r) => (r.item === item ? { ...r, mode } : r));
  }
  return [...requirements, { item, mode }];
}

/** Two lists are the same matrix row if every item has the same mode; order doesn't matter here. */
function sameRow(a: Requirement[], b: Requirement[]): boolean {
  return a.length === b.length && a.every((r) => modeOf(b, r.item) === r.mode);
}

interface RowState {
  busy?: boolean;
  saved?: string;
  warnings?: Warning[];
  problem?: Problem;
  /** The item whose cell the server's error is about. */
  errorItem?: string;
  /** Someone else saved this job first: what it looks like now. */
  conflict?: Job;
}

/** What changed under the user's feet, in words. */
function describeConflict(items: Item[], loaded: Job, mine: Requirement[], theirs: Job): string[] {
  const lines: string[] = [];
  if (theirs.display_name !== loaded.display_name) lines.push(`It was renamed to "${theirs.display_name}".`);
  if (theirs.is_active !== loaded.is_active) lines.push(theirs.is_active ? "It was reactivated." : "It was deactivated.");
  for (const item of items) {
    const yours = modeOf(mine, item.key);
    const now = modeOf(theirs.requirements, item.key);
    const before = modeOf(loaded.requirements, item.key);
    if (now !== before) {
      lines.push(
        `${item.display_name}: was ${MODE_LABEL[before]}, now ${MODE_LABEL[now]}` +
          (yours !== before && yours !== now ? ` (you set ${MODE_LABEL[yours]})` : ""),
      );
    }
  }
  return lines.length ? lines : ["Someone saved this job without changing the equipment list."];
}

/**
 * The matrix (docs/admin-api.md §5): jobs down the side, items across the top,
 * each cell Required, Manual or Off, saved a row at a time with the version
 * the row was loaded at. A supervisor sees it but can't change it (§1).
 */
export function MatrixPage({ user }: { user: DashboardUser }) {
  const readOnly = user.role !== "admin";
  const { data, problem, loading, reload, setData } = useResource(
    async () => ({ items: await getItems(), jobs: await getJobs() }),
    [],
  );
  const [drafts, setDrafts] = useState<Record<string, Requirement[]>>({});
  const [rows, setRows] = useState<Record<string, RowState>>({});

  const patchRow = (key: string, state: RowState | null) =>
    setRows((current) => {
      const next = { ...current };
      if (state) next[key] = state;
      else delete next[key];
      return next;
    });

  const replaceJob = (job: Job) =>
    setData((previous) => (previous ? { ...previous, jobs: previous.jobs.map((j) => (j.key === job.key ? job : j)) } : previous!));

  function choose(job: Job, item: string, mode: Mode) {
    const next = withMode(drafts[job.key] ?? job.requirements, item, mode);
    setDrafts((current) => {
      const copy = { ...current };
      if (sameRow(next, job.requirements)) delete copy[job.key];
      else copy[job.key] = next;
      return copy;
    });
    patchRow(job.key, null);
  }

  function discard(job: Job) {
    setDrafts((current) => {
      const copy = { ...current };
      delete copy[job.key];
      return copy;
    });
    patchRow(job.key, null);
  }

  async function save(job: Job) {
    const requirements = drafts[job.key];
    if (!requirements) return;
    patchRow(job.key, { busy: true });
    try {
      const result = await putRequirements(job.key, job.version, requirements);
      replaceJob(result.job);
      discard(result.job);
      patchRow(job.key, {
        saved: `Saved. It takes effect within ${result.applies_within_s} seconds.`,
        warnings: result.warnings,
      });
    } catch (err) {
      if (err instanceof ApiError && err.code === "version_conflict" && err.current) {
        patchRow(job.key, { conflict: err.current });
        return;
      }
      // `requirements[2].item` -> the cell for the third item in the list we sent.
      const index = err instanceof ApiError ? /^requirements\[(\d+)\]/.exec(err.field ?? "")?.[1] : undefined;
      patchRow(job.key, {
        problem: problemFrom(err, "Couldn't save this row."),
        errorItem: index === undefined ? undefined : requirements[Number(index)]?.item,
      });
    }
  }

  /** Take the other person's version, then either drop my edits or keep them on top. */
  function resolve(job: Job, keepMine: boolean) {
    const theirs = rows[job.key]?.conflict;
    if (!theirs) return;
    replaceJob(theirs);
    if (!keepMine) discard(theirs);
    patchRow(job.key, keepMine ? { problem: { message: "Showing their version with your edits on top. Save again when you're happy.", tone: "notice" } } : null);
  }

  const items = data?.items ?? [];
  const jobs = data?.jobs ?? [];

  return (
    <section className="page">
      <h1>Job matrix</h1>
      <p className="hint">
        {readOnly
          ? "You can see which equipment each job needs. Only an admin can change it."
          : "For each job, choose whether each item is required, shown to the worker as “verify manually”, or not asked for. Save a row when you're done with it."}
      </p>
      {problem && <ProblemBanner problem={problem} />}
      <Loading loading={loading} hasData={!!data} />

      {data && (
        <div className="table-scroll">
          <table className="matrix">
            <thead>
              <tr>
                <th scope="col">Job</th>
                {items.map((item) => (
                  <th scope="col" key={item.key}>
                    {item.display_name}
                    <small className="muted">
                      {item.detector_ap50 !== null ? `AP50 ${item.detector_ap50.toFixed(2)}` : <Tag tone="warn">Gemini only</Tag>}
                    </small>
                  </th>
                ))}
                {!readOnly && <th scope="col">Row</th>}
              </tr>
            </thead>
            <tbody>
              {jobs.map((job) => {
                const requirements = drafts[job.key] ?? job.requirements;
                const state = rows[job.key] ?? {};
                const dirty = !!drafts[job.key];
                const cols = items.length + (readOnly ? 1 : 2);
                return [
                  <tr key={job.key} className={job.is_active ? undefined : "inactive"}>
                    <th scope="row">
                      {job.display_name} {!job.is_active && <Tag>inactive</Tag>}
                      <small className="muted">version {job.version}</small>
                    </th>
                    {items.map((item) => {
                      const mode = modeOf(requirements, item.key);
                      const warned = state.warnings?.some((w) => w.item === item.key);
                      return (
                        <td key={item.key} className={state.errorItem === item.key ? "cell-error" : undefined}>
                          {readOnly ? (
                            MODE_LABEL[mode]
                          ) : (
                            <select
                              aria-label={`${job.display_name}: ${item.display_name}`}
                              value={mode}
                              disabled={state.busy}
                              onChange={(e) => choose(job, item.key, e.target.value as Mode)}
                            >
                              {(["required", "pending", "off"] as Mode[]).map((m) => (
                                <option key={m} value={m}>
                                  {MODE_LABEL[m]}
                                </option>
                              ))}
                            </select>
                          )}
                          {warned && <Tag tone="warn">Gemini only</Tag>}
                        </td>
                      );
                    })}
                    {!readOnly && (
                      <td className="actions">
                        <button type="button" aria-label={`Save ${job.display_name}`} disabled={!dirty || state.busy} onClick={() => void save(job)}>
                          {state.busy ? "Saving…" : "Save"}
                        </button>
                        <button type="button" className="secondary" aria-label={`Discard ${job.display_name}`} disabled={!dirty || state.busy} onClick={() => discard(job)}>
                          Discard
                        </button>
                      </td>
                    )}
                  </tr>,
                  (state.saved || state.warnings?.length || state.problem || state.conflict) && (
                    <tr key={`${job.key}-notes`} className="notes">
                      <td colSpan={cols}>
                        {state.problem && <ProblemBanner problem={state.problem} />}
                        {state.conflict && (
                          <div className="panel conflict" role="alert" data-testid={`conflict-${job.key}`}>
                            <strong>Someone changed {state.conflict.display_name} while you were editing it.</strong>
                            <ul>
                              {describeConflict(items, job, requirements, state.conflict).map((line) => (
                                <li key={line}>{line}</li>
                              ))}
                            </ul>
                            <div className="actions">
                              <button type="button" className="secondary" onClick={() => resolve(job, false)}>
                                Use their version
                              </button>
                              <button type="button" onClick={() => resolve(job, true)}>
                                Keep my edits on top
                              </button>
                            </div>
                          </div>
                        )}
                        {state.saved && <p className="saved">{state.saved}</p>}
                        {state.warnings?.map((w) => (
                          <p className="warning" key={`${w.code}-${w.item ?? ""}`}>
                            <Tag tone="warn">Heads up</Tag> {w.message}
                          </p>
                        ))}
                      </td>
                    </tr>
                  ),
                ];
              })}
            </tbody>
          </table>
        </div>
      )}
      {data && (
        <p>
          <button type="button" className="secondary" onClick={reload}>
            Reload
          </button>
        </p>
      )}
    </section>
  );
}
