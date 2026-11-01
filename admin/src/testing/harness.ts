import type { DashboardUser } from "../api/client";
import { resetClientState, setAdminTransport } from "../api/client";
import { createMockBackend, type MockBackend } from "../mock/backend";

export const ADMIN: DashboardUser = {
  id: 1, email: "admin@example.com", display_name: "Test Admin", role: "admin", site: null, must_change_password: false,
};

/** The supervisor of Pune Tower B, as seeded in the mock backend. */
export const SUPERVISOR_B: DashboardUser = {
  id: 101, email: "supervisor.b@example.com", display_name: "Site Supervisor B", role: "supervisor",
  site: "pune_tower_b", must_change_password: false,
};

/**
 * A fresh mock backend for one test, wired in as the `/admin/*` transport.
 * Pass the same user to the page you render, so the page and the server agree
 * on who is signed in.
 */
export function startMockBackend(user: DashboardUser = ADMIN): MockBackend {
  resetClientState();
  const backend = createMockBackend({
    currentUser: () => user,
    now: () => new Date("2026-10-25T12:00:00Z"),
  });
  setAdminTransport(backend.fetch);
  return backend;
}
