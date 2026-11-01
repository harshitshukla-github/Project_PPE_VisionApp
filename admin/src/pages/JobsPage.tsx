import { useState, type FormEvent } from "react";

import { ApiError, createJob, getItems, getJobs, putJobOrder, updateJob } from "../api/client";
import { problemFrom, type Problem } from "../api/problems";
import type { Job, Warning } from "../api/types";
import { ProblemBanner } from "../components/ProblemBanner";
import { Field, Loading, Tag } from "../components/widgets";
import { useResource } from "../hooks/useResource";

/**
 * Jobs (docs/admin-api.md §5, admin only): add, rename, reorder, deactivate.
 * Which equipment each job needs is the Matrix page. A job is never deleted,
 * because checks refer to it by key; "deactivate" takes it out of the app's
 * list instead.
 */
export function JobsPage() {
  const { data, problem, loading, setData, reload } = useResource(
    async () => ({ items: await getItems(), jobs: await getJobs() }),
    [],
  );
  const [message, setMessage] = useState<Problem | null>(null);
  const [warnings, setWarnings] = useState<Warning[]>([]);
  const [busy, setBusy] = useState(false);
  const [renaming, setRenaming] = useState<{ key: string; name: string } | null>(null);

  // The add-a-job form.
  const [key, setKey] = useState("");
  const [name, setName] = useState("");
  const [required, setRequired] = useState<string[]>([]);
  const [fieldErrors, setFieldErrors] = useState<Record<string, string>>({});

  const jobs = data?.jobs ?? [];
  const replaceJob = (job: Job) =>
    setData((previous) => ({ ...previous!, jobs: previous!.jobs.map((j) => (j.key === job.key ? job : j)) }));

  /** Runs one change and turns any failure into a message; a version conflict shows their version. */
  async function run(action: () => Promise<void>) {
    setBusy(true);
    setMessage(null);
    try {
      await action();
    } catch (err) {
      if (err instanceof ApiError && err.code === "version_conflict" && err.current) {
        replaceJob(err.current);
        setMessage({
          message: `Someone changed ${err.current.display_name} while you were working. It now shows their version; do it again if you still want to.`,
          tone: "notice",
        });
      } else {
        setMessage(problemFrom(err, "Couldn't save that."));
      }
    } finally {
      setBusy(false);
    }
  }

  const move = (index: number, by: -1 | 1) =>
    run(async () => {
      const keys = jobs.map((j) => j.key);
      [keys[index], keys[index + by]] = [keys[index + by], keys[index]];
      const ordered = await putJobOrder(keys);
      setData((previous) => ({ ...previous!, jobs: ordered }));
    });

  const toggle = (job: Job) =>
    run(async () => replaceJob((await updateJob(job.key, { version: job.version, is_active: !job.is_active })).job));

  const rename = (job: Job, display_name: string) =>
    run(async () => {
      replaceJob((await updateJob(job.key, { version: job.version, display_name })).job);
      setRenaming(null);
    });

  async function add(event: FormEvent) {
    event.preventDefault();
    setFieldErrors({});
    setWarnings([]);
    await run(async () => {
      try {
        const result = await createJob({
          key: key.trim(),
          display_name: name.trim(),
          requirements: required.map((item) => ({ item, mode: "required" as const })),
        });
        setWarnings(result.warnings);
        setKey("");
        setName("");
        setRequired([]);
        reload();
      } catch (err) {
        // Put the server's sentence next to the field it is about.
        if (err instanceof ApiError && err.field && ["key", "display_name", "requirements"].includes(err.field)) {
          setFieldErrors({ [err.field]: err.message });
          return;
        }
        throw err;
      }
    });
  }

  return (
    <section className="page">
      <h1>Jobs</h1>
      <p className="hint">
        The jobs workers choose from, in the order the app shows them. Which equipment each job needs is on the Matrix page.
      </p>
      {problem && <ProblemBanner problem={problem} />}
      {message && <ProblemBanner problem={message} />}
      <Loading loading={loading} hasData={!!data} />

      {data && (
        <div className="table-scroll">
        <table className="table">
          <thead>
            <tr>
              <th scope="col">Order</th>
              <th scope="col">Job</th>
              <th scope="col">Key</th>
              <th scope="col">Status</th>
              <th scope="col">Actions</th>
            </tr>
          </thead>
          <tbody>
            {jobs.map((job, index) => (
              <tr key={job.key} className={job.is_active ? undefined : "inactive"}>
                <td className="actions">
                  <button type="button" className="secondary" aria-label={`Move ${job.display_name} up`} disabled={busy || index === 0} onClick={() => void move(index, -1)}>
                    ↑
                  </button>
                  <button type="button" className="secondary" aria-label={`Move ${job.display_name} down`} disabled={busy || index === jobs.length - 1} onClick={() => void move(index, 1)}>
                    ↓
                  </button>
                </td>
                <td>
                  {renaming?.key === job.key ? (
                    <form
                      className="inline-form"
                      onSubmit={(e) => {
                        e.preventDefault();
                        void rename(job, renaming.name.trim());
                      }}
                    >
                      <input
                        aria-label={`New name for ${job.display_name}`}
                        value={renaming.name}
                        onChange={(e) => setRenaming({ key: job.key, name: e.target.value })}
                        autoFocus
                      />
                      <button type="submit" disabled={busy || !renaming.name.trim()}>
                        Save name
                      </button>
                      <button type="button" className="secondary" onClick={() => setRenaming(null)}>
                        Cancel
                      </button>
                    </form>
                  ) : (
                    job.display_name
                  )}
                </td>
                <td>
                  <code>{job.key}</code>
                </td>
                <td>{job.is_active ? <Tag tone="good">Active</Tag> : <Tag>Inactive</Tag>}</td>
                <td className="actions">
                  <button type="button" className="secondary" disabled={busy} onClick={() => setRenaming({ key: job.key, name: job.display_name })}>
                    Rename
                  </button>
                  <button type="button" className="secondary" aria-label={`${job.is_active ? "Deactivate" : "Reactivate"} ${job.display_name}`} disabled={busy} onClick={() => void toggle(job)}>
                    {job.is_active ? "Deactivate" : "Reactivate"}
                  </button>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
        </div>
      )}

      {data && (
        <form className="panel grid-form" onSubmit={(e) => void add(e)}>
          <h2>Add a job</h2>
          <Field label="Key" hint="Lowercase letters, digits and underscores. It never changes afterwards." error={fieldErrors.key}>
            <input value={key} onChange={(e) => setKey(e.target.value)} placeholder="scaffolder" />
          </Field>
          <Field label="Name" error={fieldErrors.display_name}>
            <input value={name} onChange={(e) => setName(e.target.value)} placeholder="Scaffolder" />
          </Field>
          <fieldset>
            <legend>Required equipment</legend>
            <div className="checks">
              {data.items.map((item) => (
                <label key={item.key} className="check">
                  <input
                    type="checkbox"
                    checked={required.includes(item.key)}
                    onChange={(e) => setRequired((r) => (e.target.checked ? [...r, item.key] : r.filter((k) => k !== item.key)))}
                  />
                  {item.display_name}
                </label>
              ))}
            </div>
            <small className="muted">Items to be shown as “verify manually” are set on the Matrix page.</small>
            {fieldErrors.requirements && <span className="field-error">{fieldErrors.requirements}</span>}
          </fieldset>
          <div className="actions">
            <button type="submit" disabled={busy || !key.trim() || !name.trim()}>
              Add job
            </button>
          </div>
          {warnings.map((w) => (
            <p className="warning" key={`${w.code}-${w.item ?? ""}`}>
              <Tag tone="warn">Heads up</Tag> {w.message}
            </p>
          ))}
        </form>
      )}
    </section>
  );
}
