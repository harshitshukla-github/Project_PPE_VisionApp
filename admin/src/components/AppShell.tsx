import type { ReactNode } from "react";

import { adminApiIsMocked, type DashboardUser } from "../api/client";
import { useAuth } from "../auth/AuthContext";
import { PAGES } from "../pages/registry";

/**
 * The frame around every page: the navigation (only the pages this role can
 * use), who is signed in, and Sign out. While the admin API is faked, a
 * banner says so on every page, so nobody mistakes sample data for the real
 * thing.
 */
export function AppShell({ user, path, children }: { user: DashboardUser; path: string; children: ReactNode }) {
  const { signOut } = useAuth();
  const visible = PAGES.filter((page) => page.roles.includes(user.role));

  return (
    <div className="shell">
      {adminApiIsMocked() && (
        <div className="mock-banner" role="note" data-testid="mock-banner">
          Sample data: the admin API isn&apos;t built yet, so nothing you change here reaches the server, and a reload starts again.
        </div>
      )}
      <header className="topbar">
        <strong className="brand">PPB Admin</strong>
        <nav aria-label="Pages">
          {visible.map((page) => (
            <a key={page.path} href={`#${page.path}`} aria-current={path === page.path ? "page" : undefined}>
              {page.label}
            </a>
          ))}
        </nav>
        <div className="who">
          <span data-testid="whoami">
            {user.display_name} · {user.role}
          </span>
          <button type="button" className="link" onClick={() => void signOut()}>
            Sign out
          </button>
        </div>
      </header>
      <main className="content">{children}</main>
    </div>
  );
}
