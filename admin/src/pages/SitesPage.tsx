import { useState, type FormEvent } from "react";

import { ApiError, createSite, getSites, updateSite } from "../api/client";
import { problemFrom, type Problem } from "../api/problems";
import type { Site } from "../api/types";
import { ProblemBanner } from "../components/ProblemBanner";
import { Field, Loading, Tag, Time } from "../components/widgets";
import { useResource } from "../hooks/useResource";

/**
 * Sites (docs/admin-api.md §3, admin only): the places workers are checked.
 * Every worker and every supervisor belongs to one. A site can't be
 * deactivated while it still has active workers or supervisors, and is never
 * deleted, because checks will refer to it.
 */
export function SitesPage() {
  const { data, problem, loading, setData } = useResource(getSites, []);
  const [message, setMessage] = useState<Problem | null>(null);
  const [busy, setBusy] = useState(false);
  const [renaming, setRenaming] = useState<{ key: string; name: string } | null>(null);
  const [key, setKey] = useState("");
  const [name, setName] = useState("");
  const [fieldErrors, setFieldErrors] = useState<Record<string, string>>({});

  const replace = (site: Site) => setData((sites) => (sites ?? []).map((s) => (s.key === site.key ? site : s)));

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

  async function add(event: FormEvent) {
    event.preventDefault();
    setFieldErrors({});
    await run(async () => {
      try {
        const { site } = await createSite({ key: key.trim(), display_name: name.trim() });
        setData((sites) => [...(sites ?? []), site]);
        setKey("");
        setName("");
      } catch (err) {
        if (err instanceof ApiError && err.field && ["key", "display_name"].includes(err.field)) {
          setFieldErrors({ [err.field]: err.message });
          return;
        }
        throw err;
      }
    });
  }

  return (
    <section className="page">
      <h1>Sites</h1>
      <p className="hint">Where workers are checked. Each worker and each supervisor belongs to one site.</p>
      {problem && <ProblemBanner problem={problem} />}
      {message && <ProblemBanner problem={message} />}
      <Loading loading={loading} hasData={!!data} />

      {data && (
        <div className="table-scroll">
        <table className="table">
          <thead>
            <tr>
              <th scope="col">Site</th>
              <th scope="col">Key</th>
              <th scope="col">Active workers</th>
              <th scope="col">Status</th>
              <th scope="col">Created</th>
              <th scope="col">Actions</th>
            </tr>
          </thead>
          <tbody>
            {data.map((site) => (
              <tr key={site.key} className={site.is_active ? undefined : "inactive"}>
                <td>
                  {renaming?.key === site.key ? (
                    <form
                      className="inline-form"
                      onSubmit={(e) => {
                        e.preventDefault();
                        void run(async () => {
                          replace((await updateSite(site.key, { display_name: renaming.name.trim() })).site);
                          setRenaming(null);
                        });
                      }}
                    >
                      <input aria-label={`New name for ${site.display_name}`} value={renaming.name} onChange={(e) => setRenaming({ key: site.key, name: e.target.value })} autoFocus />
                      <button type="submit" disabled={busy || !renaming.name.trim()}>
                        Save name
                      </button>
                      <button type="button" className="secondary" onClick={() => setRenaming(null)}>
                        Cancel
                      </button>
                    </form>
                  ) : (
                    site.display_name
                  )}
                </td>
                <td>
                  <code>{site.key}</code>
                </td>
                <td>{site.worker_count}</td>
                <td>{site.is_active ? <Tag tone="good">Active</Tag> : <Tag>Inactive</Tag>}</td>
                <td>
                  <Time iso={site.created_at} />
                </td>
                <td className="actions">
                  <button type="button" className="secondary" disabled={busy} onClick={() => setRenaming({ key: site.key, name: site.display_name })}>
                    Rename
                  </button>
                  <button
                    type="button"
                    className="secondary"
                    aria-label={`${site.is_active ? "Deactivate" : "Reactivate"} ${site.display_name}`}
                    disabled={busy}
                    onClick={() => void run(async () => replace((await updateSite(site.key, { is_active: !site.is_active })).site))}
                  >
                    {site.is_active ? "Deactivate" : "Reactivate"}
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
          <h2>Add a site</h2>
          <Field label="Key" hint="Lowercase letters, digits and underscores. It never changes afterwards." error={fieldErrors.key}>
            <input value={key} onChange={(e) => setKey(e.target.value)} placeholder="pune_tower_d" />
          </Field>
          <Field label="Name" error={fieldErrors.display_name}>
            <input value={name} onChange={(e) => setName(e.target.value)} placeholder="Pune Tower D" />
          </Field>
          <div className="actions">
            <button type="submit" disabled={busy || !key.trim() || !name.trim()}>
              Add site
            </button>
          </div>
        </form>
      )}
    </section>
  );
}
