import { Fragment, useState, type FormEvent } from "react";

import { getAudit, getSites } from "../api/client";
import { ProblemBanner } from "../components/ProblemBanner";
import { Loading, Pager, Time } from "../components/widgets";
import { usePages, useResource } from "../hooks/useResource";

const PAGE_SIZE = 25;

/** Every action in the audit log (docs/admin-api.md §9). */
const ACTIONS = [
  "site.create", "site.update",
  "job.create", "job.update", "job.requirements", "job.order",
  "settings.update",
  "user.create", "user.update", "user.reset_password",
  "worker.create", "worker.update", "worker.reset_pin",
];

interface Filters {
  action: string;
  target: string;
  actor: string;
  site: string;
  /** A calendar day, yyyy-mm-dd. */
  from: string;
  to: string;
}

const NO_FILTERS: Filters = { action: "", target: "", actor: "", site: "", from: "", to: "" };

/** `from` is the start of that day, UTC. `to` is the day after: the API's `to` is exclusive, the person's isn't. */
const startOfDay = (day: string) => (day ? `${day}T00:00:00Z` : undefined);
const endOfDay = (day: string) =>
  day ? new Date(Date.parse(`${day}T00:00:00Z`) + 86_400_000).toISOString().replace(".000Z", "Z") : undefined;

function Details({ label, value }: { label: string; value: Record<string, unknown> | null }) {
  return (
    <div>
      <h3>{label}</h3>
      {value ? <pre>{JSON.stringify(value, null, 2)}</pre> : <p className="muted">Nothing recorded.</p>}
    </div>
  );
}

/**
 * Audit log (docs/admin-api.md §9, admin only): every change made through the
 * API, and every account or site made from the command line, newest first.
 * Passwords, PINs, hashes and tokens are never in it.
 */
export function AuditPage() {
  const sites = useResource(getSites, []);
  const [draft, setDraft] = useState<Filters>(NO_FILTERS);
  const [applied, setApplied] = useState<Filters>(NO_FILTERS);
  const [open, setOpen] = useState<number | null>(null);
  const list = usePages(
    (cursor) =>
      getAudit({
        action: applied.action,
        target: applied.target.trim(),
        actor: applied.actor.trim(),
        site: applied.site,
        from: startOfDay(applied.from),
        to: endOfDay(applied.to),
        limit: PAGE_SIZE,
        cursor,
      }),
    PAGE_SIZE,
    [applied],
  );

  function apply(event: FormEvent) {
    event.preventDefault();
    setApplied(draft);
    setOpen(null);
  }

  return (
    <section className="page">
      <h1>Audit log</h1>
      <p className="hint">Who changed what, newest first. The filters match exactly.</p>

      <form className="toolbar" onSubmit={apply} aria-label="Filters">
        <label>
          Action
          <select value={draft.action} onChange={(e) => setDraft({ ...draft, action: e.target.value })}>
            <option value="">Any action</option>
            {ACTIONS.map((a) => (
              <option key={a} value={a}>
                {a}
              </option>
            ))}
          </select>
        </label>
        <label>
          Target
          <input value={draft.target} onChange={(e) => setDraft({ ...draft, target: e.target.value })} placeholder="job:welder" />
        </label>
        <label>
          Who
          <input value={draft.actor} onChange={(e) => setDraft({ ...draft, actor: e.target.value })} placeholder="email, or cli" />
        </label>
        <label>
          Site
          <select value={draft.site} onChange={(e) => setDraft({ ...draft, site: e.target.value })}>
            <option value="">Any site</option>
            {(sites.data ?? []).map((s) => (
              <option key={s.key} value={s.key}>
                {s.display_name}
              </option>
            ))}
          </select>
        </label>
        <label>
          From
          <input type="date" value={draft.from} onChange={(e) => setDraft({ ...draft, from: e.target.value })} />
        </label>
        <label>
          To
          <input type="date" value={draft.to} onChange={(e) => setDraft({ ...draft, to: e.target.value })} />
        </label>
        <button type="submit">Apply</button>
        <button
          type="button"
          className="secondary"
          onClick={() => {
            setDraft(NO_FILTERS);
            setApplied(NO_FILTERS);
            setOpen(null);
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
                <th scope="col">When</th>
                <th scope="col">Who</th>
                <th scope="col">Action</th>
                <th scope="col">Target</th>
                <th scope="col">Site</th>
                <th scope="col">Details</th>
              </tr>
            </thead>
            <tbody>
              {list.page.items.length === 0 && (
                <tr>
                  <td colSpan={6} className="muted">
                    Nothing in the log matches.
                  </td>
                </tr>
              )}
              {list.page.items.map((entry) => (
                <Fragment key={entry.id}>
                  <tr>
                    <td>
                      <Time iso={entry.at} />
                    </td>
                    <td>{entry.actor}</td>
                    <td>
                      <code>{entry.action}</code>
                    </td>
                    <td>
                      <code>{entry.target}</code>
                    </td>
                    <td>{entry.site ?? <span className="muted">—</span>}</td>
                    <td>
                      <button
                        type="button"
                        className="secondary"
                        aria-expanded={open === entry.id}
                        aria-label={`${open === entry.id ? "Hide" : "Show"} details of ${entry.action} on ${entry.target}`}
                        onClick={() => setOpen(open === entry.id ? null : entry.id)}
                      >
                        {open === entry.id ? "Hide" : "Show"}
                      </button>
                    </td>
                  </tr>
                  {open === entry.id && (
                    <tr className="notes" data-testid={`details-${entry.id}`}>
                      <td colSpan={6}>
                        <div className="diff">
                          <Details label="Before" value={entry.before} />
                          <Details label="After" value={entry.after} />
                        </div>
                        {entry.ip && <p className="muted">From {entry.ip}</p>}
                      </td>
                    </tr>
                  )}
                </Fragment>
              ))}
            </tbody>
          </table>
        </div>
          <Pager range={list.range} hasPrevious={list.hasPrevious} hasNext={list.hasNext} onPrevious={list.previous} onNext={list.next} busy={list.loading} />
        </>
      )}
    </section>
  );
}
