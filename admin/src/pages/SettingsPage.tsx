import { useState } from "react";

import { ApiError, getSettings, patchSettings } from "../api/client";
import { problemFrom, type Problem } from "../api/problems";
import type { Setting, SettingValue, SettingsPatch, Warning } from "../api/types";
import { ProblemBanner } from "../components/ProblemBanner";
import { ConfirmDialog, Loading, Tag, Time } from "../components/widgets";
import { useResource } from "../hooks/useResource";

/** One edit in progress: a new value (as typed), or "go back to the default". */
type Edit = { kind: "value"; text: string } | { kind: "default" };

const label = (name: string) => name.charAt(0).toUpperCase() + name.slice(1).replace(/_/g, " ");

const show = (value: SettingValue) => (typeof value === "boolean" ? (value ? "On" : "Off") : String(value));

/** What the typed text means for this setting, or undefined if it isn't a valid value yet. */
function parse(setting: Setting, text: string): SettingValue | undefined {
  if (setting.type === "boolean") return text === "true";
  if (setting.type === "choice") return text;
  if (text.trim() === "") return undefined;
  const number = Number(text);
  return Number.isFinite(number) ? number : undefined;
}

/**
 * Runtime settings (docs/admin-api.md §6, admin only). The form is built from
 * `GET /settings`: each setting says its own type, range, choices and whether
 * saving needs a confirmation, so nothing about them is written down twice.
 * Saving sends a merge patch of only what changed: a value sets it, "default"
 * (`null`) clears the override. It is all or nothing.
 */
export function SettingsPage() {
  const { data, problem, loading, setData } = useResource(getSettings, []);
  const [edits, setEdits] = useState<Record<string, Edit>>({});
  const [confirming, setConfirming] = useState(false);
  const [busy, setBusy] = useState(false);
  const [message, setMessage] = useState<Problem | null>(null);
  const [fieldErrors, setFieldErrors] = useState<Record<string, string>>({});
  const [saved, setSaved] = useState<{ warnings: Warning[]; appliesWithinS: number } | null>(null);

  const settings = data ?? [];

  /** What a setting will be once saved, and what it is now. */
  const nextValue = (s: Setting): SettingValue | undefined => {
    const edit = edits[s.name];
    if (!edit) return s.value;
    return edit.kind === "default" ? s.default : parse(s, edit.text);
  };

  /** The edits that actually change something, as a merge patch. */
  const changes: Array<{ setting: Setting; to: SettingValue | null }> = [];
  let invalid = false;
  for (const s of settings) {
    const edit = edits[s.name];
    if (!edit) continue;
    if (edit.kind === "default") {
      if (s.overridden) changes.push({ setting: s, to: null });
      continue;
    }
    const value = parse(s, edit.text);
    if (value === undefined) invalid = true;
    else if (value !== s.value) changes.push({ setting: s, to: value });
  }
  const needsConfirm = changes.filter((c) => c.setting.confirm);

  function edit(setting: Setting, next: Edit | undefined) {
    setEdits((current) => {
      const copy = { ...current };
      if (next) copy[setting.name] = next;
      else delete copy[setting.name];
      return copy;
    });
    setSaved(null);
  }

  async function save() {
    setBusy(true);
    setMessage(null);
    setFieldErrors({});
    try {
      const patch: SettingsPatch = Object.fromEntries(changes.map((c) => [c.setting.name, c.to]));
      const result = await patchSettings(patch);
      setData(result.settings);
      setEdits({});
      setSaved({ warnings: result.warnings, appliesWithinS: result.applies_within_s });
    } catch (err) {
      if (err instanceof ApiError && err.field) {
        setFieldErrors({ [err.field]: err.message });
        setMessage({ message: "Nothing was saved. Fix the setting marked below and try again.", tone: "error" });
      } else {
        setMessage(problemFrom(err, "Couldn't save the settings."));
      }
    } finally {
      setBusy(false);
      setConfirming(false);
    }
  }

  return (
    <section className="page">
      <h1>Settings</h1>
      <p className="hint">
        Changes apply to every worker&apos;s next check. Each is recorded in the audit log. Keys, the model and the database stay in the server&apos;s configuration.
      </p>
      {problem && <ProblemBanner problem={problem} />}
      {message && <ProblemBanner problem={message} />}
      <Loading loading={loading} hasData={!!data} />

      {saved && (
        <div className="panel saved-panel" role="status">
          <p className="saved">Saved. It takes effect within {saved.appliesWithinS} seconds.</p>
          {saved.warnings.map((w) => (
            <p className="warning" key={`${w.code}-${w.setting ?? ""}`}>
              <Tag tone="warn">Heads up</Tag> {w.message}
            </p>
          ))}
        </div>
      )}

      {data && (
        <form
          className="settings"
          // The server decides what is in range (and says so); the browser's own pop-up would hide that.
          noValidate
          onSubmit={(e) => {
            e.preventDefault();
            if (needsConfirm.length) setConfirming(true);
            else void save();
          }}
        >
          {settings.map((s) => {
            const current = edits[s.name];
            const text = current?.kind === "value" ? current.text : String(nextValue(s) ?? "");
            const pending = !!current && changes.some((c) => c.setting.name === s.name);
            return (
              <fieldset key={s.name} className={pending ? "setting changed" : "setting"}>
                <legend>
                  {label(s.name)} {s.overridden ? <Tag tone="warn">Changed here</Tag> : <Tag>Default</Tag>}
                </legend>
                <p className="hint">{s.description}</p>
                <div className="setting-input">
                  {s.type === "number" && (
                    <input
                      type="number"
                      aria-label={label(s.name)}
                      min={s.min ?? undefined}
                      max={s.max ?? undefined}
                      step={s.step ?? undefined}
                      value={text}
                      aria-invalid={!!fieldErrors[s.name]}
                      onChange={(e) => edit(s, { kind: "value", text: e.target.value })}
                    />
                  )}
                  {s.type === "boolean" && (
                    <input
                      type="checkbox"
                      aria-label={label(s.name)}
                      checked={nextValue(s) === true}
                      onChange={(e) => edit(s, { kind: "value", text: String(e.target.checked) })}
                    />
                  )}
                  {s.type === "choice" && (
                    <select aria-label={label(s.name)} value={text} onChange={(e) => edit(s, { kind: "value", text: e.target.value })}>
                      {s.choices!.map((c) => (
                        <option key={c} value={c}>
                          {c}
                        </option>
                      ))}
                    </select>
                  )}
                  {s.type === "number" && (
                    <small className="muted">
                      {s.min} to {s.max}
                    </small>
                  )}
                  <small className="muted">Default: {show(s.default)}</small>
                  <button
                    type="button"
                    className="secondary"
                    aria-label={`Use the default for ${label(s.name)}`}
                    disabled={!s.overridden || current?.kind === "default"}
                    onClick={() => edit(s, { kind: "default" })}
                  >
                    Use default
                  </button>
                </div>
                {current?.kind === "default" && s.overridden && <p className="hint">Will go back to the default ({show(s.default)}).</p>}
                {s.overridden && s.updated_by && (
                  <p className="hint">
                    Last changed by {s.updated_by}, <Time iso={s.updated_at} />.
                  </p>
                )}
                {fieldErrors[s.name] && <span className="field-error">{fieldErrors[s.name]}</span>}
              </fieldset>
            );
          })}
          <div className="actions">
            <button type="submit" disabled={busy || changes.length === 0 || invalid}>
              Save changes
            </button>
            <button type="button" className="secondary" disabled={busy || Object.keys(edits).length === 0} onClick={() => setEdits({})}>
              Discard changes
            </button>
          </div>
        </form>
      )}

      {confirming && (
        <ConfirmDialog title="Confirm these changes" confirmLabel="Yes, save" busy={busy} onConfirm={() => void save()} onCancel={() => setConfirming(false)}>
          <p>These change how every check is judged. Check them before saving:</p>
          <ul>
            {changes.map(({ setting, to }) => (
              <li key={setting.name}>
                <strong>{label(setting.name)}</strong>: {show(setting.value)} → {to === null ? `${show(setting.default)} (the default)` : show(to)}
              </li>
            ))}
          </ul>
        </ConfirmDialog>
      )}
    </section>
  );
}
