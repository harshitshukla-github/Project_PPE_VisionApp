import { useState, type FormEvent } from "react";

import { ApiError, checkPhotoSrc, getCheck, getItems, reviewCheck, type DashboardUser } from "../api/client";
import { problemFrom, type Problem } from "../api/problems";
import type { Check, Detection, ReviewOutcome } from "../api/types";
import { ProblemBanner } from "../components/ProblemBanner";
import { Loading, Tag, Time } from "../components/widgets";
import { useResource } from "../hooks/useResource";
import { ReviewTag, VerdictTag } from "./ChecksPage";

const NOTE_MAX = 500;

/** Classes that are evidence of a violation, not equipment: drawn in red. */
const VIOLATIONS = new Set(["no_hardhat", "no_vest"]);

/**
 * One check (docs/checks-api.md §2, §4.2, §4.3): the photo with what the
 * detector found drawn on it, what the job required then, Gemini's findings,
 * the settings the verdict was decided with, and the review.
 */
export function CheckPage({ user, id }: { user: DashboardUser; id: string }) {
  const valid = /^\d+$/.test(id);
  const check = useResource(
    () => (valid ? getCheck(Number(id)) : Promise.reject(new ApiError(404, "No such check.", { code: "not_found" }))),
    [id],
  );
  const items = useResource(getItems, []);
  const name = (key: string) =>
    items.data?.find((i) => i.key === key)?.display_name ??
    key.charAt(0).toUpperCase() + key.slice(1).replace(/_/g, " ");

  const c = check.data;
  return (
    <section className="page">
      <p>
        <a href="#/checks">← All checks</a>
      </p>
      <h1>Check #{id}</h1>
      {check.problem && <ProblemBanner problem={check.problem} />}
      <Loading loading={check.loading} hasData={!!c} />

      {c && (
        <>
          <div className="check-summary">
            <VerdictTag verdict={c.verdict} /> <ReviewTag check={c} />
            {c.verdict_source === "detector+gemini" && <Tag title="The detector was unsure, and Gemini's look at the photo settled or confirmed it.">Detector and Gemini</Tag>}
          </div>
          <dl className="facts">
            <dt>When</dt>
            <dd>
              <Time iso={c.at} />
            </dd>
            <dt>Worker</dt>
            <dd>
              {c.worker_name ?? "Unknown worker"} ({c.worker_id})
              {!c.signed_in && (
                <>
                  {" "}
                  <Tag tone="warn">No login</Tag> <span className="muted">The worker ID was typed in an app without login, not checked.</span>
                </>
              )}
            </dd>
            <dt>Site</dt>
            <dd>{c.site ?? "—"}</dd>
            <dt>Job</dt>
            <dd>
              {c.job_name} <small className="muted">(matrix version {c.job_version} at the time)</small>
            </dd>
          </dl>

          <div className="check-columns">
            <Photo check={c} itemName={name} />

            <div>
              <h2>What the job required then</h2>
              <ul className="requirements">
                {c.required.map((item) => (
                  <li key={item}>
                    {name(item)}{" "}
                    {c.missing_equipment.includes(item) ? (
                      <Tag tone={c.verdict === "FAIL" ? "bad" : "warn"}>{c.verdict === "FAIL" ? "Not worn" : "Not confirmed"}</Tag>
                    ) : c.unverifiable_equipment.includes(item) ? (
                      <Tag tone="warn">Couldn&apos;t be checked</Tag>
                    ) : (
                      <Tag tone="good">Found</Tag>
                    )}
                  </li>
                ))}
                {c.pending.map((item) => (
                  <li key={item}>
                    {name(item)} <Tag title="Listed for the worker to check by hand; never scored.">Check by hand</Tag>
                  </li>
                ))}
              </ul>
              {c.violations_detected.length > 0 && (
                <p>
                  <strong>Violation seen:</strong> {c.violations_detected.map(name).join(", ")}
                </p>
              )}
              {c.explanation && (
                <>
                  <h2>Explanation the worker was given</h2>
                  <p>{c.explanation}</p>
                </>
              )}

              {c.verification && (
                <>
                  <h2>Gemini&apos;s second opinion</h2>
                  <table className="table compact">
                    <thead>
                      <tr>
                        <th scope="col">Item</th>
                        <th scope="col">Finding</th>
                        <th scope="col">Sure</th>
                        <th scope="col">Why</th>
                      </tr>
                    </thead>
                    <tbody>
                      {c.verification.items.map((v) => (
                        <tr key={v.item}>
                          <th scope="row">{name(v.item)}</th>
                          <td>{v.status === "worn" ? "Worn" : v.status === "not_worn" ? "Not worn" : "Uncertain"}</td>
                          <td>{Math.round(v.confidence * 100)}%</td>
                          <td>{v.reason}</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                  <p className="hint">
                    {c.verification.model}. Findings below {Math.round(c.gemini_min_confidence * 100)}% count as uncertain.
                  </p>
                </>
              )}

              <h2>Decided with</h2>
              <dl className="facts" data-testid="decided-with">
                <dt>Detector threshold</dt>
                <dd>{c.confidence_threshold}</dd>
                <dt>Gemini second opinion</dt>
                <dd>{c.gemini_verify ? "On" : "Off"}</dd>
                <dt>Gemini minimum confidence</dt>
                <dd>{c.gemini_min_confidence}</dd>
                <dt>Model</dt>
                <dd>{c.model_used}</dd>
                <dt>Photo kept for training</dt>
                <dd>{c.training_consent ? "Yes, the worker agreed" : "No"}</dd>
              </dl>
            </div>
          </div>

          <ReviewSection user={user} check={c} onChange={(next) => check.setData(next)} />
        </>
      )}
    </section>
  );
}

/** The photo, with the detector's boxes drawn over it; or why there is no photo. */
function Photo({ check, itemName }: { check: Check; itemName: (key: string) => string }) {
  const kept = check.photo.state === "kept";
  const src = useResource(() => (kept ? checkPhotoSrc(check.id) : Promise.resolve(null)), [check.id, kept]);
  const [broken, setBroken] = useState(false);

  if (!kept) {
    const why =
      check.photo.state === "deleted"
        ? "The photo has been deleted: on request, or because it was older than the time photos are kept."
        : check.signed_in
          ? "No photo was stored for this check: photos were switched off, or storing it failed."
          : "No photo: this check was sent without a login, so its photo is never kept.";
    return (
      <figure className="photo" data-testid="no-photo">
        <p className="hint">{why}</p>
      </figure>
    );
  }

  const width = check.photo.width ?? 1;
  const height = check.photo.height ?? 1;
  return (
    <figure className="photo">
      {src.problem && <ProblemBanner problem={src.problem} />}
      {broken && <ProblemBanner problem={{ message: "The photo couldn't be loaded.", tone: "error" }} />}
      {src.data && !broken && (
        <div className="photo-frame">
          <img src={src.data} alt={`The photo from check #${check.id}`} width={width} height={height} onError={() => setBroken(true)} />
          {/* The boxes are in the stored photo's pixels (after EXIF orientation), the same space as the image. */}
          <svg className="boxes" viewBox={`0 0 ${width} ${height}`} preserveAspectRatio="none" aria-hidden="true">
            {check.detections.map((d, i) => (
              <Box key={i} detection={d} label={itemName(d.class)} fontSize={Math.round(width / 36)} />
            ))}
          </svg>
        </div>
      )}
      <figcaption>
        <ul className="detections" aria-label="What the detector found">
          {check.detections.map((d, i) => (
            <li key={i}>
              {VIOLATIONS.has(d.class) ? <Tag tone="bad">{itemName(d.class)}</Tag> : itemName(d.class)}{" "}
              <small className="muted">{Math.round(d.confidence * 100)}%</small>
            </li>
          ))}
        </ul>
      </figcaption>
    </figure>
  );
}

function Box({ detection, label, fontSize }: { detection: Detection; label: string; fontSize: number }) {
  const [x1, y1, x2, y2] = detection.bbox;
  const bad = VIOLATIONS.has(detection.class);
  return (
    <g className={bad ? "box box-bad" : detection.class === "person" ? "box box-person" : "box"} data-testid="box">
      <rect x={x1} y={y1} width={x2 - x1} height={y2 - y1} vectorEffect="non-scaling-stroke" />
      <text x={x1 + 6} y={y1 + fontSize + 2} fontSize={fontSize}>
        {label} {Math.round(detection.confidence * 100)}%
      </text>
    </g>
  );
}

/**
 * The review (§4.3): one decision for the whole check, on any verdict. Once
 * someone has reviewed it, a supervisor sees the review; an admin can replace
 * it. If someone else reviews it while this page is open, the server answers
 * `409 already_reviewed` with the check as it is now, and their review is
 * shown instead of being overwritten.
 */
function ReviewSection({ user, check, onChange }: { user: DashboardUser; check: Check; onChange: (check: Check) => void }) {
  const isAdmin = user.role === "admin";
  const [replacing, setReplacing] = useState(false);
  const [outcome, setOutcome] = useState<ReviewOutcome | "">("");
  const [note, setNote] = useState("");
  const [busy, setBusy] = useState(false);
  const [message, setMessage] = useState<Problem | null>(null);
  const [saved, setSaved] = useState(false);

  const formOpen = !check.review || replacing;

  async function submit(event: FormEvent) {
    event.preventDefault();
    if (!outcome) return;
    setBusy(true);
    setMessage(null);
    setSaved(false);
    try {
      const result = await reviewCheck(check.id, {
        outcome,
        ...(note.trim() ? { note: note.trim() } : {}),
        ...(replacing ? { replace: true } : {}),
      });
      onChange(result.check);
      setReplacing(false);
      setOutcome("");
      setNote("");
      setSaved(true);
    } catch (err) {
      if (err instanceof ApiError && err.code === "already_reviewed" && err.current) {
        // `current` is the check as it is now (as for version_conflict, but a check, not a job).
        onChange(err.current as unknown as Check);
        setMessage({
          message: "Someone reviewed this check while you had it open. Their review is shown below; yours wasn't saved.",
          tone: "notice",
        });
      } else {
        setMessage(problemFrom(err, "Couldn't save the review."));
      }
    } finally {
      setBusy(false);
    }
  }

  return (
    <section className="panel" aria-labelledby="review-heading">
      <h2 id="review-heading">Review</h2>
      {message && <ProblemBanner problem={message} />}
      {saved && (
        <p className="saved" role="status">
          Review saved.
        </p>
      )}

      {check.review && (
        <div data-testid="current-review">
          <p>
            <ReviewTag check={check} /> by {check.review.by}, <Time iso={check.review.at} />
          </p>
          {check.review.note && <p className="note">{check.review.note}</p>}
          {isAdmin && !replacing && (
            <button type="button" className="secondary" onClick={() => setReplacing(true)}>
              Replace review
            </button>
          )}
        </div>
      )}

      {!check.review && (
        <p className="hint">
          The verdict stays as the system decided it; your review is recorded beside it.
          {check.verdict === "FAIL" && " If the worker was compliant after all, say so here."}
          {check.verdict === "PASS" && " If something was missed, say so here."}
        </p>
      )}

      {formOpen && (
        <form className="grid-form" onSubmit={(e) => void submit(e)} aria-label={replacing ? "Replace the review" : "Review this check"}>
          <fieldset>
            <legend>Was the worker compliant?</legend>
            <label className="check">
              <input type="radio" name="outcome" value="compliant" checked={outcome === "compliant"} onChange={() => setOutcome("compliant")} />
              Compliant
            </label>
            <label className="check">
              <input type="radio" name="outcome" value="not_compliant" checked={outcome === "not_compliant"} onChange={() => setOutcome("not_compliant")} />
              Not compliant
            </label>
          </fieldset>
          <label>
            Note (optional)
            <textarea value={note} maxLength={NOTE_MAX} rows={3} onChange={(e) => setNote(e.target.value)} />
          </label>
          <small className="muted">
            {note.length}/{NOTE_MAX}
          </small>
          <div className="actions">
            <button type="submit" disabled={busy || !outcome}>
              {busy ? "Saving…" : replacing ? "Replace review" : "Save review"}
            </button>
            {replacing && (
              <button type="button" className="secondary" onClick={() => setReplacing(false)}>
                Cancel
              </button>
            )}
          </div>
        </form>
      )}
    </section>
  );
}
