import { cloneElement, useId, useState, type ReactElement, type ReactNode } from "react";

/**
 * A labelled form control. The hint and the error sit outside the `<label>`,
 * so the control's name stays "Email" and doesn't become "Email … is taken";
 * `aria-describedby` ties them to the control, and the error is announced.
 */
export function Field({
  label,
  hint,
  error,
  children,
}: {
  label: string;
  hint?: string;
  error?: string;
  /** The one input, select or textarea; it receives the id and the aria attributes. */
  children: ReactElement<Record<string, unknown>>;
}) {
  const id = useId();
  const describedBy = [hint && `${id}-hint`, error && `${id}-error`].filter(Boolean).join(" ") || undefined;
  return (
    <div className="field">
      <label htmlFor={id}>{label}</label>
      {cloneElement(children, { id, "aria-invalid": error ? true : undefined, "aria-describedby": describedBy })}
      {hint && (
        <small id={`${id}-hint`} className="muted">
          {hint}
        </small>
      )}
      {error && (
        <span id={`${id}-error`} className="field-error" role="alert">
          {error}
        </span>
      )}
    </div>
  );
}

/** A small coloured label: "Active", "Locked", "Gemini only". */
export function Tag({ tone = "neutral", children }: { tone?: "neutral" | "good" | "warn" | "bad"; children: ReactNode }) {
  return <span className={`tag tag-${tone}`}>{children}</span>;
}

/** The "51–100 of 312" line with Previous and Next (docs/admin-api.md §2). */
export function Pager({
  range,
  hasPrevious,
  hasNext,
  onPrevious,
  onNext,
  busy,
}: {
  range: { first: number; last: number; total: number } | null;
  hasPrevious: boolean;
  hasNext: boolean;
  onPrevious: () => void;
  onNext: () => void;
  busy?: boolean;
}) {
  if (!range) return null;
  return (
    <nav className="pager" aria-label="Pages">
      <span data-testid="range">
        {range.total === 0 ? "Nothing to show" : `${range.first}–${range.last} of ${range.total}`}
      </span>
      <button type="button" className="secondary" onClick={onPrevious} disabled={!hasPrevious || busy}>
        Previous
      </button>
      <button type="button" className="secondary" onClick={onNext} disabled={!hasNext || busy}>
        Next
      </button>
    </nav>
  );
}

/**
 * A secret that exists only on this screen: a worker's one-time PIN, or a
 * user's temporary password (docs/admin-api.md §7, §8). The server never
 * shows it again, so the panel stays until the person confirms they have
 * handed it over.
 */
export function OnceSecret({
  title,
  label,
  secret,
  onDone,
}: {
  title: string;
  label: string;
  secret: string;
  onDone: () => void;
}) {
  const [copied, setCopied] = useState(false);

  async function copy() {
    try {
      await navigator.clipboard.writeText(secret);
      setCopied(true);
    } catch {
      // No clipboard permission: the secret is on screen to be read or typed.
    }
  }

  return (
    <section className="panel secret" role="alert" data-testid="once-secret">
      <h2>{title}</h2>
      <p className="hint">
        {label} It is shown <strong>once</strong>: it can&apos;t be looked up again afterwards.
      </p>
      <p className="secret-value" data-testid="secret-value">
        {secret}
      </p>
      <div className="actions">
        <button type="button" className="secondary" onClick={() => void copy()}>
          {copied ? "Copied" : "Copy"}
        </button>
        <button type="button" onClick={onDone}>
          I&apos;ve handed it over
        </button>
      </div>
    </section>
  );
}

/** Asks before a change that deserves it (docs/admin-api.md §6 `confirm`). */
export function ConfirmDialog({
  title,
  children,
  confirmLabel,
  busy,
  onConfirm,
  onCancel,
}: {
  title: string;
  children: ReactNode;
  confirmLabel: string;
  busy?: boolean;
  onConfirm: () => void;
  onCancel: () => void;
}) {
  return (
    <div className="overlay">
      <div className="dialog" role="dialog" aria-modal="true" aria-label={title}>
        <h2>{title}</h2>
        {children}
        <div className="actions">
          <button type="button" className="secondary" onClick={onCancel} disabled={busy}>
            Cancel
          </button>
          <button type="button" onClick={onConfirm} disabled={busy}>
            {busy ? "Saving…" : confirmLabel}
          </button>
        </div>
      </div>
    </div>
  );
}

/** "Loading…" until a page has something to show; after that a reload keeps the old data in place. */
export function Loading({ loading, hasData }: { loading: boolean; hasData: boolean }) {
  if (hasData || !loading) return null;
  return (
    <p className="hint" aria-busy="true">
      Loading…
    </p>
  );
}

/** "25 Oct 2026, 14:03" in the viewer's time, with the exact UTC time on hover. */
export function Time({ iso }: { iso: string | null }) {
  if (!iso) return <span className="muted">never</span>;
  const date = new Date(iso);
  return <time dateTime={iso} title={iso}>{date.toLocaleString(undefined, { dateStyle: "medium", timeStyle: "short" })}</time>;
}
