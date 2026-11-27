import type { ReactElement } from "react";

import type { DashboardUser, Role } from "../api/client";
import { AuditPage } from "./AuditPage";
import { CheckPage } from "./CheckPage";
import { ChecksPage } from "./ChecksPage";
import { JobsPage } from "./JobsPage";
import { MatrixPage } from "./MatrixPage";
import { SettingsPage } from "./SettingsPage";
import { SitesPage } from "./SitesPage";
import { UsersPage } from "./UsersPage";
import { WorkersPage } from "./WorkersPage";

export interface PageDef {
  path: string;
  label: string;
  /** Who gets a link to it. The server's `403 forbidden` is still the real rule (docs/admin-api.md §1). */
  roles: Role[];
  /** `rest` is what follows the page's path, such as `1006` in `/checks/1006`; empty for the page itself. */
  render: (user: DashboardUser, rest: string) => ReactElement;
}

const BOTH: Role[] = ["admin", "supervisor"];
const ADMIN_ONLY: Role[] = ["admin"];

/** Every page, in the order of the navigation. Roles follow the table in docs/admin-api.md §1. */
export const PAGES: PageDef[] = [
  { path: "/matrix", label: "Matrix", roles: BOTH, render: (user) => <MatrixPage user={user} /> },
  {
    path: "/checks",
    label: "Checks",
    roles: BOTH,
    render: (user, rest) => (rest ? <CheckPage user={user} id={rest} /> : <ChecksPage user={user} />),
  },
  { path: "/jobs", label: "Jobs", roles: ADMIN_ONLY, render: () => <JobsPage /> },
  { path: "/workers", label: "Workers", roles: BOTH, render: (user) => <WorkersPage user={user} /> },
  { path: "/sites", label: "Sites", roles: ADMIN_ONLY, render: () => <SitesPage /> },
  { path: "/users", label: "Users", roles: ADMIN_ONLY, render: (user) => <UsersPage user={user} /> },
  { path: "/settings", label: "Settings", roles: ADMIN_ONLY, render: () => <SettingsPage /> },
  { path: "/audit", label: "Audit log", roles: ADMIN_ONLY, render: () => <AuditPage /> },
];

/** The page an address belongs to, and what follows its path: `/checks/1006` → Checks, `1006`. */
export function pageFor(path: string): { page: PageDef; rest: string } | undefined {
  for (const page of PAGES) {
    if (path === page.path) return { page, rest: "" };
    if (path.startsWith(`${page.path}/`)) return { page, rest: path.slice(page.path.length + 1) };
  }
  return undefined;
}

/** Where "/" goes: the one page everybody can see. */
export const DEFAULT_PATH = "/matrix";

/** The signed-in user changing their own password: opened from the header, not listed in the navigation. */
export const PASSWORD_PATH = "/password";
