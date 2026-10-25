import { useState, type FormEvent } from "react";

import {
  ApiError,
  createUser,
  getSites,
  getUsers,
  resetUserPassword,
  updateUser,
  type DashboardUser,
  type Role,
} from "../api/client";
import { problemFrom, type Problem } from "../api/problems";
import type { AdminUser } from "../api/types";
import { ProblemBanner } from "../components/ProblemBanner";
import { Field, Loading, OnceSecret, Tag, Time } from "../components/widgets";
import { useResource } from "../hooks/useResource";

/**
 * Dashboard users (docs/admin-api.md §7, admin only). A supervisor belongs to
 * one site; an admin works across all of them. A new user, or a password
 * reset, gets a temporary password that is shown once: the person replaces it
 * at their first sign-in. You can't change your own role, site or active
 * flag, or reset your own password here; the server refuses, and the buttons
 * say so before you try.
 *
 * `now` is only a seam for tests.
 */
export function UsersPage({ user, now = () => new Date() }: { user: DashboardUser; now?: () => Date }) {
  const users = useResource(getUsers, []);
  const sites = useResource(getSites, []);
  const siteName = (key: string | null) => (key ? (sites.data?.find((s) => s.key === key)?.display_name ?? key) : "All sites");

  const [message, setMessage] = useState<Problem | null>(null);
  const [secret, setSecret] = useState<{ title: string; label: string; secret: string } | null>(null);
  const [busy, setBusy] = useState(false);
  const [editing, setEditing] = useState<{ id: number; name: string; role: Role; site: string } | null>(null);

  const [email, setEmail] = useState("");
  const [name, setName] = useState("");
  const [role, setRole] = useState<Role>("supervisor");
  const [site, setSite] = useState("");
  const [fieldErrors, setFieldErrors] = useState<Record<string, string>>({});

  const replace = (updated: AdminUser) => users.setData((all) => (all ?? []).map((u) => (u.id === updated.id ? updated : u)));
  const activeSites = (sites.data ?? []).filter((s) => s.is_active);

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

  const showPassword = (target: { display_name: string; email: string }, password: string, title: string) =>
    setSecret({
      title,
      label: `Give this temporary password to ${target.display_name} (${target.email}). They must replace it when they first sign in.`,
      secret: password,
    });

  async function add(event: FormEvent) {
    event.preventDefault();
    setFieldErrors({});
    await run(async () => {
      try {
        const result = await createUser({
          email: email.trim(),
          display_name: name.trim(),
          role,
          ...(role === "supervisor" ? { site } : {}),
        });
        showPassword(result.user, result.temporary_password, `${result.user.display_name} was added`);
        users.setData((all) => [...(all ?? []), result.user]);
        setEmail("");
        setName("");
      } catch (err) {
        if (err instanceof ApiError && err.field && ["email", "display_name", "site", "role"].includes(err.field)) {
          setFieldErrors({ [err.field]: err.message });
          return;
        }
        throw err;
      }
    });
  }

  const isMe = (u: AdminUser) => u.email === user.email;
  const isLocked = (u: AdminUser) => u.locked_until !== null && new Date(u.locked_until) > now();

  return (
    <section className="page">
      <h1>Users</h1>
      <p className="hint">People who sign in to this dashboard. An admin works across all sites; a supervisor looks after one.</p>
      {users.problem && <ProblemBanner problem={users.problem} />}
      {message && <ProblemBanner problem={message} />}
      {secret && <OnceSecret title={secret.title} label={secret.label} secret={secret.secret} onDone={() => setSecret(null)} />}
      <Loading loading={users.loading} hasData={!!users.data} />

      {users.data && (
        <div className="table-scroll">
        <table className="table">
          <thead>
            <tr>
              <th scope="col">Name</th>
              <th scope="col">Role</th>
              <th scope="col">Site</th>
              <th scope="col">Status</th>
              <th scope="col">Last sign-in</th>
              <th scope="col">Actions</th>
            </tr>
          </thead>
          <tbody>
            {users.data.map((u) => (
              <tr key={u.id} className={u.is_active ? undefined : "inactive"}>
                <th scope="row">
                  {editing?.id === u.id ? (
                    <form
                      className="inline-form"
                      onSubmit={(e) => {
                        e.preventDefault();
                        void run(async () => {
                          replace(
                            (
                              await updateUser(u.id, {
                                display_name: editing.name.trim(),
                                role: editing.role,
                                site: editing.role === "admin" ? null : editing.site,
                              })
                            ).user,
                          );
                          setEditing(null);
                        });
                      }}
                    >
                      <input aria-label={`Name for ${u.email}`} value={editing.name} onChange={(e) => setEditing({ ...editing, name: e.target.value })} autoFocus />
                      <select aria-label={`Role for ${u.email}`} value={editing.role} onChange={(e) => setEditing({ ...editing, role: e.target.value as Role })}>
                        <option value="supervisor">Supervisor</option>
                        <option value="admin">Admin</option>
                      </select>
                      {editing.role === "supervisor" && (
                        <select aria-label={`Site for ${u.email}`} value={editing.site} onChange={(e) => setEditing({ ...editing, site: e.target.value })}>
                          <option value="">Choose a site</option>
                          {(sites.data ?? []).filter((s) => s.is_active || s.key === u.site).map((s) => (
                            <option key={s.key} value={s.key}>
                              {s.display_name}
                            </option>
                          ))}
                        </select>
                      )}
                      <button type="submit" disabled={busy || !editing.name.trim() || (editing.role === "supervisor" && !editing.site)}>
                        Save
                      </button>
                      <button type="button" className="secondary" onClick={() => setEditing(null)}>
                        Cancel
                      </button>
                    </form>
                  ) : (
                    <>
                      {u.display_name} {isMe(u) && <Tag tone="good">You</Tag>}
                      <small className="muted">{u.email}</small>
                    </>
                  )}
                </th>
                <td>{u.role === "admin" ? "Admin" : "Supervisor"}</td>
                <td>{siteName(u.site)}</td>
                <td>
                  {u.is_active ? <Tag tone="good">Active</Tag> : <Tag>Inactive</Tag>}
                  {isLocked(u) && <Tag tone="bad">Locked</Tag>}
                  {u.must_change_password && <Tag tone="warn">Password not changed yet</Tag>}
                </td>
                <td>
                  <Time iso={u.last_login_at} />
                </td>
                <td className="actions">
                  <button
                    type="button"
                    className="secondary"
                    aria-label={`Reset password for ${u.email}`}
                    disabled={busy || isMe(u)}
                    title={isMe(u) ? "Use Change password to reset your own." : undefined}
                    onClick={() =>
                      void run(async () => {
                        const { temporary_password } = await resetUserPassword(u.id);
                        showPassword(u, temporary_password, `New password for ${u.display_name}`);
                        users.reload();
                      })
                    }
                  >
                    Reset password
                  </button>
                  <button
                    type="button"
                    className="secondary"
                    aria-label={`Edit ${u.email}`}
                    disabled={busy || isMe(u)}
                    title={isMe(u) ? "You can't change your own role or site." : undefined}
                    onClick={() => setEditing({ id: u.id, name: u.display_name, role: u.role, site: u.site ?? "" })}
                  >
                    Edit
                  </button>
                  <button
                    type="button"
                    className="secondary"
                    aria-label={`${u.is_active ? "Deactivate" : "Reactivate"} ${u.email}`}
                    disabled={busy || isMe(u)}
                    title={isMe(u) ? "You can't deactivate yourself." : undefined}
                    onClick={() => void run(async () => replace((await updateUser(u.id, { is_active: !u.is_active })).user))}
                  >
                    {u.is_active ? "Deactivate" : "Reactivate"}
                  </button>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
        </div>
      )}

      <form className="panel grid-form" onSubmit={(e) => void add(e)}>
        <h2>Add a user</h2>
        <Field label="Email" error={fieldErrors.email}>
          <input type="email" value={email} onChange={(e) => setEmail(e.target.value)} placeholder="supervisor@example.com" />
        </Field>
        <Field label="Name" error={fieldErrors.display_name}>
          <input value={name} onChange={(e) => setName(e.target.value)} placeholder="Site Supervisor" />
        </Field>
        <Field label="Role" error={fieldErrors.role}>
          <select value={role} onChange={(e) => setRole(e.target.value as Role)}>
            <option value="supervisor">Supervisor</option>
            <option value="admin">Admin</option>
          </select>
        </Field>
        {role === "supervisor" && (
          <Field label="Site" error={fieldErrors.site}>
            <select value={site} onChange={(e) => setSite(e.target.value)}>
              <option value="">Choose a site</option>
              {activeSites.map((s) => (
                <option key={s.key} value={s.key}>
                  {s.display_name}
                </option>
              ))}
            </select>
          </Field>
        )}
        <div className="actions">
          <button type="submit" disabled={busy || !email.trim() || !name.trim() || (role === "supervisor" && !site)}>
            Add user
          </button>
        </div>
      </form>
    </section>
  );
}
