import type { ReactElement } from "react";

import type { DashboardUser, Role } from "../api/client";
import { AuditPage } from "./AuditPage";
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
  render: (user: DashboardUser) => ReactElement;
}

const BOTH: Role[] = ["admin", "supervisor"];
const ADMIN_ONLY: Role[] = ["admin"];

/** Every page, in the order of the navigation. Roles follow the table in docs/admin-api.md §1. */
export const PAGES: PageDef[] = [
  { path: "/matrix", label: "Matrix", roles: BOTH, render: (user) => <MatrixPage user={user} /> },
  { path: "/jobs", label: "Jobs", roles: ADMIN_ONLY, render: () => <JobsPage /> },
  { path: "/workers", label: "Workers", roles: BOTH, render: (user) => <WorkersPage user={user} /> },
  { path: "/sites", label: "Sites", roles: ADMIN_ONLY, render: () => <SitesPage /> },
  { path: "/users", label: "Users", roles: ADMIN_ONLY, render: (user) => <UsersPage user={user} /> },
  { path: "/settings", label: "Settings", roles: ADMIN_ONLY, render: () => <SettingsPage /> },
  { path: "/audit", label: "Audit log", roles: ADMIN_ONLY, render: () => <AuditPage /> },
];

/** Where "/" goes: the one page everybody can see. */
export const DEFAULT_PATH = "/matrix";
