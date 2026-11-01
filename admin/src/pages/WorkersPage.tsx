import { useState, type FormEvent } from "react";

import {
  ApiError,
  createWorker,
  getSites,
  getWorkers,
  resetWorkerPin,
  updateWorker,
  type DashboardUser,
} from "../api/client";
import { problemFrom, type Problem } from "../api/problems";
import type { Worker } from "../api/types";
import { ProblemBanner } from "../components/ProblemBanner";
import { Field, Loading, OnceSecret, Pager, Tag, Time } from "../components/widgets";
import { usePages, useResource } from "../hooks/useResource";

const PAGE_SIZE = 25;

interface Filters {
  search: string;
  site: string;
  status: "" | "true" | "false";
  locked: boolean;
}

const NO_FILTERS: Filters = { search: "", site: "", status: "", locked: false };

/**
 * Workers (docs/admin-api.md §8). An admin sees every site; a supervisor
 * sees, adds and manages only their own site's workers. A worker's one-time
 * PIN is shown once, when they are added or their PIN is reset, and the
 * worker replaces it at their first sign-in.
 *
 * `now` is only a seam for tests: it decides which "locked until" times are
 * still in the future.
 */
export function WorkersPage({ user, now = () => new Date() }: { user: DashboardUser; now?: () => Date }) {
  const isAdmin = user.role === "admin";
  const sites = useResource(getSites, []);
  const siteName = (key: string) => sites.data?.find((s) => s.key === key)?.display_name ?? key;

  const [draft, setDraft] = useState<Filters>(NO_FILTERS);
  const [applied, setApplied] = useState<Filters>(NO_FILTERS);
  const list = usePages(
    (cursor) =>
      getWorkers({
        search: applied.search.trim(),
        site: applied.site,
        active: applied.status === "" ? undefined : applied.status === "true",
        locked: applied.locked || undefined,
        limit: PAGE_SIZE,
        cursor,
      }),
    PAGE_SIZE,
    [applied],
  );

  const [message, setMessage] = useState<Problem | null>(null);
  const [secret, setSecret] = useState<{ title: string; label: string; secret: string } | null>(null);
  const [busy, setBusy] = useState(false);
  const [editing, setEditing] = useState<{ id: string; name: string; site: string } | null>(null);

  // The add-a-worker form.
  const [newId, setNewId] = useState("");
  const [newName, setNewName] = useState("");
  const [newSite, setNewSite] = useState("");
  const [fieldErrors, setFieldErrors] = useState<Record<string, string>>({});

  const replaceWorker = (worker: Worker) =>
    list.setData((page) =>
      page ? { ...page, items: page.items.map((w) => (w.worker_id === worker.worker_id ? worker : w)) } : page!,
    );

  async function run(action: () => Promise<void>) {
    setBusy(true);
    setMessage(null);
    try {
      await action();
    } catch (err) {
      setMessage(problemFrom(err, "Couldn't save that."));
    } finally {
      setBusy(false);
    }
  }

  const showPin = (worker: Worker, pin: string, title: string) =>
    setSecret({
      title,
      label: `Give this PIN to ${worker.display_name} (${worker.worker_id}). They choose their own at first sign-in.`,
      secret: pin,
    });

  async function add(event: FormEvent) {
    event.preventDefault();
    setFieldErrors({});
    await run(async () => {
      try {
        const { worker, one_time_pin } = await createWorker({
          worker_id: newId.trim(),
          display_name: newName.trim(),
          ...(isAdmin ? { site: newSite } : {}),
        });
        showPin(worker, one_time_pin, `${worker.display_name} was added`);
        setNewId("");
        setNewName("");
        list.reload();
      } catch (err) {
        if (err instanceof ApiError && err.field && ["worker_id", "display_name", "site"].includes(err.field)) {
          setFieldErrors({ [err.field]: err.message });
          return;
        }
        throw err;
      }
    });
  }

  const isLocked = (w: Worker) => w.locked_until !== null && new Date(w.locked_until) > now();

  function applyFilters(event: FormEvent) {
    event.preventDefault();
    setApplied(draft);
  }

  const activeSites = (sites.data ?? []).filter((s) => s.is_active);

  return (
    <section className="page">
      <h1>Workers</h1>
      <p className="hint">
        {isAdmin ? "Workers at every site." : `Workers at ${siteName(user.site ?? "")}.`} A worker signs in to the app with their ID and a PIN you give them.
      </p>
      {message && <ProblemBanner problem={message} />}
      {secret && <OnceSecret title={secret.title} label={secret.label} secret={secret.secret} onDone={() => setSecret(null)} />}

      <form className="toolbar" onSubmit={applyFilters} aria-label="Filters">
        <label>
          Search
          <input value={draft.search} onChange={(e) => setDraft({ ...draft, search: e.target.value })} placeholder="ID or name" />
        </label>
        {isAdmin && (
          <label>
            Filter by site
            <select value={draft.site} onChange={(e) => setDraft({ ...draft, site: e.target.value })}>
              <option value="">All sites</option>
              {(sites.data ?? []).map((s) => (
                <option key={s.key} value={s.key}>
                  {s.display_name}
                </option>
              ))}
            </select>
          </label>
        )}
        <label>
          Status
          <select value={draft.status} onChange={(e) => setDraft({ ...draft, status: e.target.value as Filters["status"] })}>
            <option value="">Active and inactive</option>
            <option value="true">Active</option>
            <option value="false">Inactive</option>
          </select>
        </label>
        <label className="check">
          <input type="checkbox" checked={draft.locked} onChange={(e) => setDraft({ ...draft, locked: e.target.checked })} />
          Locked out now
        </label>
        <button type="submit">Apply</button>
        <button
          type="button"
          className="secondary"
          onClick={() => {
            setDraft(NO_FILTERS);
            setApplied(NO_FILTERS);
          }}
        >
          Clear
        </button>
      </form>

      {list.problem && <ProblemBanner problem={list.problem} />}
      <Loading loading={list.loading} hasData={!!list.page} />

      {list.page && (
        <>
          <div className="table-scroll">
        <table className="table">
            <thead>
              <tr>
                <th scope="col">Worker ID</th>
                <th scope="col">Name</th>
                {isAdmin && <th scope="col">Site</th>}
                <th scope="col">Status</th>
                <th scope="col">Last sign-in</th>
                <th scope="col">Actions</th>
              </tr>
            </thead>
            <tbody>
              {list.page.items.length === 0 && (
                <tr>
                  <td colSpan={isAdmin ? 6 : 5} className="muted">
                    No workers match.
                  </td>
                </tr>
              )}
              {list.page.items.map((w) => (
                <tr key={w.worker_id} className={w.is_active ? undefined : "inactive"}>
                  <th scope="row">{w.worker_id}</th>
                  <td>
                    {editing?.id === w.worker_id ? (
                      <form
                        className="inline-form"
                        onSubmit={(e) => {
                          e.preventDefault();
                          void run(async () => {
                            const patch: { display_name: string; site?: string } = { display_name: editing.name.trim() };
                            if (isAdmin && editing.site !== w.site) patch.site = editing.site;
                            replaceWorker((await updateWorker(w.worker_id, patch)).worker);
                            setEditing(null);
                          });
                        }}
                      >
                        <input aria-label={`Name for ${w.worker_id}`} value={editing.name} onChange={(e) => setEditing({ ...editing, name: e.target.value })} autoFocus />
                        {isAdmin && (
                          <select aria-label={`Site for ${w.worker_id}`} value={editing.site} onChange={(e) => setEditing({ ...editing, site: e.target.value })}>
                            {(sites.data ?? []).filter((s) => s.is_active || s.key === w.site).map((s) => (
                              <option key={s.key} value={s.key}>
                                {s.display_name}
                              </option>
                            ))}
                          </select>
                        )}
                        <button type="submit" disabled={busy || !editing.name.trim()}>
                          Save
                        </button>
                        <button type="button" className="secondary" onClick={() => setEditing(null)}>
                          Cancel
                        </button>
                      </form>
                    ) : (
                      w.display_name
                    )}
                  </td>
                  {isAdmin && <td>{siteName(w.site)}</td>}
                  <td>
                    {w.is_active ? <Tag tone="good">Active</Tag> : <Tag>Inactive</Tag>}
                    {isLocked(w) && <Tag tone="bad">Locked until <Time iso={w.locked_until} /></Tag>}
                    {w.must_change_pin && <Tag tone="warn">PIN not changed yet</Tag>}
                  </td>
                  <td>
                    <Time iso={w.last_login_at} />
                  </td>
                  <td className="actions">
                    <button
                      type="button"
                      className="secondary"
                      aria-label={`Reset PIN for ${w.worker_id}`}
                      disabled={busy}
                      onClick={() =>
                        void run(async () => {
                          const { one_time_pin } = await resetWorkerPin(w.worker_id);
                          showPin(w, one_time_pin, `New PIN for ${w.display_name}`);
                          list.reload();
                        })
                      }
                    >
                      Reset PIN
                    </button>
                    <button type="button" className="secondary" aria-label={`Edit ${w.worker_id}`} disabled={busy} onClick={() => setEditing({ id: w.worker_id, name: w.display_name, site: w.site })}>
                      Edit
                    </button>
                    <button
                      type="button"
                      className="secondary"
                      aria-label={`${w.is_active ? "Deactivate" : "Reactivate"} ${w.worker_id}`}
                      disabled={busy}
                      onClick={() => void run(async () => replaceWorker((await updateWorker(w.worker_id, { is_active: !w.is_active })).worker))}
                    >
                      {w.is_active ? "Deactivate" : "Reactivate"}
                    </button>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
          <Pager range={list.range} hasPrevious={list.hasPrevious} hasNext={list.hasNext} onPrevious={list.previous} onNext={list.next} busy={list.loading} />
        </>
      )}

      <form className="panel grid-form" onSubmit={(e) => void add(e)}>
        <h2>Add a worker</h2>
        <Field label="Worker ID" hint="2 to 32 letters, digits and hyphens. Printed on the worker's badge." error={fieldErrors.worker_id}>
          <input value={newId} onChange={(e) => setNewId(e.target.value)} placeholder="W-0043" />
        </Field>
        <Field label="Name" error={fieldErrors.display_name}>
          <input value={newName} onChange={(e) => setNewName(e.target.value)} placeholder="Ravi Kumar" />
        </Field>
        {isAdmin ? (
          <Field label="Site" error={fieldErrors.site}>
            <select value={newSite} onChange={(e) => setNewSite(e.target.value)}>
              <option value="">Choose a site</option>
              {activeSites.map((s) => (
                <option key={s.key} value={s.key}>
                  {s.display_name}
                </option>
              ))}
            </select>
          </Field>
        ) : (
          <p className="hint">Site: {siteName(user.site ?? "")}</p>
        )}
        <div className="actions">
          <button type="submit" disabled={busy || !newId.trim() || !newName.trim() || (isAdmin && !newSite)}>
            Add worker
          </button>
        </div>
      </form>
    </section>
  );
}
