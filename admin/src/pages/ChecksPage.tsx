import { useState, type FormEvent } from "react";

import { getChecks, getJobs, getSites, type DashboardUser } from "../api/client";
import type { CheckSummary, Verdict } from "../api/types";
import { isUsableDay, startOfDay, startOfNextDay } from "../dates";
import { ProblemBanner } from "../components/ProblemBanner";
import { Loading, Pager, Tag, Time } from "../components/widgets";
import { usePages, useResource } from "../hooks/useResource";
import { currentQuery } from "../router";

const PAGE_SIZE = 25;

interface Filters {
  verdict: "" | Verdict;
  reviewed: "" | "true" | "false";
  signedIn: "" | "true" | "false";
  site: string;
  worker: string;
  job: string;
  /** Calendar days, yyyy-mm-dd, in the person's own timezone (src/dates.ts). */
  from: string;
  to: string;
}

const NO_FILTERS: Filters = { verdict: "", reviewed: "", signedIn: "", site: "", worker: "", job: "", from: "", to: "" };

/** A supervisor's default view: the checks waiting for a person (docs/checks-api.md §4.1). */
const WAITING: Filters = { ...NO_FILTERS, verdict: "NEEDS_REVIEW", reviewed: "false" };

/**
 * Where the list starts: a worker's checks when the address names one (the
 * Workers page links here with `?worker=`), the checks waiting for review for
 * a supervisor, everything for an admin.
 */
function startingFilters(user: DashboardUser): Filters {
  const worker = currentQuery().get("worker");
  if (worker) return { ...NO_FILTERS, worker };
  return user.role === "supervisor" ? WAITING : NO_FILTERS;
}

const VERDICT_LABEL: Record<Verdict, string> = { PASS: "Pass", FAIL: "Fail", NEEDS_REVIEW: "Needs review" };

export function VerdictTag({ verdict }: { verdict: Verdict }) {
  const tone = verdict === "PASS" ? "good" : verdict === "FAIL" ? "bad" : "warn";
  return <Tag tone={tone}>{VERDICT_LABEL[verdict]}</Tag>;
}

/** "Compliant" / "Not compliant" once reviewed; "Waiting" for a NEEDS_REVIEW nobody has looked at yet. */
export function ReviewTag({ check }: { check: Pick<CheckSummary, "review" | "verdict"> }) {
  if (check.review) {
    return check.review.outcome === "compliant" ? <Tag tone="good">Compliant</Tag> : <Tag tone="bad">Not compliant</Tag>;
  }
  return check.verdict === "NEEDS_REVIEW" ? <Tag tone="warn">Waiting</Tag> : <span className="muted">—</span>;
}

/**
 * Checks (docs/checks-api.md §4.1): every check the server stored, newest
 * first. An admin sees every site; a supervisor sees their own, and starts on
 * the checks waiting for them. Any check can be opened and reviewed.
 */
export function ChecksPage({ user }: { user: DashboardUser }) {
  const isAdmin = user.role === "admin";
  const sites = useResource(getSites, []);
  const jobs = useResource(getJobs, []);
  const siteName = (key: string | null) => (key ? sites.data?.find((s) => s.key === key)?.display_name ?? key : "—");

  const [draft, setDraft] = useState<Filters>(() => startingFilters(user));
  const [applied, setApplied] = useState<Filters>(draft);
  const [badDates, setBadDates] = useState(false);
  const list = usePages(
    (cursor) =>
      getChecks({
        verdict: applied.verdict || undefined,
        reviewed: applied.reviewed === "" ? undefined : applied.reviewed === "true",
        signed_in: applied.signedIn === "" ? undefined : applied.signedIn === "true",
        site: isAdmin ? applied.site : undefined,
        worker: applied.worker.trim(),
        job: applied.job,
        from: startOfDay(applied.from),
        to: startOfNextDay(applied.to),
        limit: PAGE_SIZE,
        cursor,
      }),
    PAGE_SIZE,
    [applied],
  );

  function apply(event: FormEvent) {
    event.preventDefault();
    // A date box with something in it that isn't a whole day yet: said so, not applied (as on the Audit log).
    const bad = (day: string) => day !== "" && !isUsableDay(day);
    setBadDates(bad(draft.from) || bad(draft.to));
    if (bad(draft.from) || bad(draft.to)) return;
    setApplied(draft);
  }

  const showingWaiting = applied.verdict === "NEEDS_REVIEW" && applied.reviewed === "false";
  const columns = isAdmin ? 7 : 6;

  return (
    <section className="page">
      <h1>Checks</h1>
      <p className="hint">
        {isAdmin ? "Checks from every site" : `Checks at ${siteName(user.site)}`}, newest first. Open one to see the photo
        and what was found, and to review it.
        {showingWaiting && " Showing the ones waiting for review."}
      </p>

      <form className="toolbar" onSubmit={apply} aria-label="Filters" noValidate>
        <label>
          Verdict
          <select value={draft.verdict} onChange={(e) => setDraft({ ...draft, verdict: e.target.value as Filters["verdict"] })}>
            <option value="">Any verdict</option>
            <option value="NEEDS_REVIEW">Needs review</option>
            <option value="FAIL">Fail</option>
            <option value="PASS">Pass</option>
          </select>
        </label>
        <label>
          Review
          <select value={draft.reviewed} onChange={(e) => setDraft({ ...draft, reviewed: e.target.value as Filters["reviewed"] })}>
            <option value="">Reviewed or not</option>
            <option value="false">Not reviewed</option>
            <option value="true">Reviewed</option>
          </select>
        </label>
        <label>
          Worker
          <input value={draft.worker} onChange={(e) => setDraft({ ...draft, worker: e.target.value })} placeholder="W-0042" />
        </label>
        <label>
          Job
          <select value={draft.job} onChange={(e) => setDraft({ ...draft, job: e.target.value })}>
            <option value="">Any job</option>
            {(jobs.data ?? []).map((j) => (
              <option key={j.key} value={j.key}>
                {j.display_name}
              </option>
            ))}
          </select>
        </label>
        <label>
          Login
          <select value={draft.signedIn} onChange={(e) => setDraft({ ...draft, signedIn: e.target.value as Filters["signedIn"] })}>
            <option value="">Signed in or not</option>
            <option value="true">Signed in</option>
            <option value="false">Without a login</option>
          </select>
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
          From
          <input
            type="date"
            min="1900-01-01"
            max="9999-12-31"
            value={draft.from}
            onChange={(e) => setDraft({ ...draft, from: e.target.value })}
            aria-invalid={badDates && draft.from !== "" && !isUsableDay(draft.from) ? true : undefined}
          />
        </label>
        <label>
          To
          <input
            type="date"
            min="1900-01-01"
            max="9999-12-31"
            value={draft.to}
            onChange={(e) => setDraft({ ...draft, to: e.target.value })}
            aria-invalid={badDates && draft.to !== "" && !isUsableDay(draft.to) ? true : undefined}
          />
        </label>
        <button type="submit">Apply</button>
        <button
          type="button"
          className="secondary"
          onClick={() => {
            setDraft(NO_FILTERS);
            setApplied(NO_FILTERS);
            setBadDates(false);
          }}
        >
          Show all
        </button>
      </form>
      {badDates && (
        <p className="field-error" role="alert">
          Finish typing the date, such as 2026-11-02.
        </p>
      )}

      {list.problem && <ProblemBanner problem={list.problem} />}
      <Loading loading={list.loading} hasData={!!list.page} />

      {list.page && (
        <>
          <div className="table-scroll">
            <table className="table">
              <thead>
                <tr>
                  <th scope="col">Check</th>
                  <th scope="col">When</th>
                  <th scope="col">Worker</th>
                  {isAdmin && <th scope="col">Site</th>}
                  <th scope="col">Job</th>
                  <th scope="col">Verdict</th>
                  <th scope="col">Review</th>
                </tr>
              </thead>
              <tbody>
                {list.page.items.length === 0 && (
                  <tr>
                    <td colSpan={columns} className="muted">
                      {showingWaiting ? "Nothing is waiting for review." : "No checks match."}
                    </td>
                  </tr>
                )}
                {list.page.items.map((c) => (
                  <tr key={c.id}>
                    <th scope="row">
                      <a href={`#/checks/${c.id}`}>#{c.id}</a>
                    </th>
                    <td>
                      <Time iso={c.at} />
                    </td>
                    <td>
                      {c.worker_name ?? <span className="muted">Unknown worker</span>} <small className="muted">{c.worker_id}</small>
                      {!c.signed_in && (
                        <Tag tone="warn" title="Sent from an app without login: the worker ID was typed, not checked.">
                          No login
                        </Tag>
                      )}
                    </td>
                    {isAdmin && <td>{siteName(c.site)}</td>}
                    <td>{c.job_name}</td>
                    <td>
                      <VerdictTag verdict={c.verdict} />
                    </td>
                    <td>
                      <ReviewTag check={c} />
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          <Pager
            range={list.range}
            hasPrevious={list.hasPrevious}
            hasNext={list.hasNext}
            onPrevious={list.previous}
            onNext={list.next}
            busy={list.loading}
          />
        </>
      )}
    </section>
  );
}
