import { useEffect, useState, type ReactNode } from "react";

import { adminApiIsMocked, getChecks, type DashboardUser } from "../api/client";
import { useAuth } from "../auth/AuthContext";
import { PAGES, PASSWORD_PATH } from "../pages/registry";

/**
 * The frame around every page: the navigation (only the pages this role can
 * use), who is signed in, and Sign out. While the admin API is faked (opt-in:
 * `VITE_ADMIN_API=mock`), a banner says so on every page, so nobody mistakes
 * sample data for the real thing.
 *
 * The Checks link carries how many checks are waiting for a person
 * (docs/checks-api.md §9): `NEEDS_REVIEW` and not reviewed, the supervisor's
 * default view. It is asked again whenever the address changes, so it drops
 * after a review once the person moves on.
 */
export function AppShell({ user, path, children }: { user: DashboardUser; path: string; children: ReactNode }) {
  const { signOut } = useAuth();
  const visible = PAGES.filter((page) => page.roles.includes(user.role));
  const waiting = useWaitingCount(path);
  const isCurrent = (pagePath: string) => path === pagePath || path.startsWith(`${pagePath}/`);

  return (
    <div className="shell">
      {adminApiIsMocked() && (
        <div className="mock-banner" role="note" data-testid="mock-banner">
          Sample data: these pages are showing made-up data from your browser, not the server. Nothing you change here is saved, and a reload starts again.
        </div>
      )}
      <header className="topbar">
        <strong className="brand">PPB Admin</strong>
        <nav aria-label="Pages">
          {visible.map((page) => (
            <a key={page.path} href={`#${page.path}`} aria-current={isCurrent(page.path) ? "page" : undefined}>
              {page.label}
              {page.path === "/checks" && waiting !== null && waiting > 0 && (
                <span className="count" data-testid="waiting-count" title="Waiting for review">
                  {waiting}
                  <span className="sr-only"> waiting for review</span>
                </span>
              )}
            </a>
          ))}
        </nav>
        <div className="who">
          <span data-testid="whoami">
            {user.display_name} · {user.role}
          </span>
          <a href={`#${PASSWORD_PATH}`} aria-current={path === PASSWORD_PATH ? "page" : undefined}>
            Change password
          </a>
          <button type="button" className="link" onClick={() => void signOut()}>
            Sign out
          </button>
        </div>
      </header>
      <main className="content">{children}</main>
    </div>
  );
}

/**
 * `total` of the checks waiting for review, or null when it can't be had: a
 * server without the checks endpoints yet, an outage. The navigation then
 * simply shows no number; the Checks page itself says what's wrong.
 */
function useWaitingCount(path: string): number | null {
  const [count, setCount] = useState<number | null>(null);
  useEffect(() => {
    let cancelled = false;
    getChecks({ verdict: "NEEDS_REVIEW", reviewed: false, limit: 1 }).then(
      (page) => {
        if (!cancelled) setCount(typeof page?.total === "number" ? page.total : null);
      },
      () => {
        if (!cancelled) setCount(null);
      },
    );
    return () => {
      cancelled = true;
    };
  }, [path]);
  return count;
}
