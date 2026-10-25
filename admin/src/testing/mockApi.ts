import { vi } from "vitest";

import type { SessionResponse } from "../api/client";

export const ADMIN_SESSION: SessionResponse = {
  user: {
    id: 1,
    email: "admin@example.com",
    display_name: "Test Admin",
    role: "admin",
    site: null,
    must_change_password: false,
  },
  csrf_token: "csrf-1",
  expires_at: "2026-10-16T03:12:00Z",
};

/** A session for a user who still has the temporary password. */
export const MUST_CHANGE_SESSION: SessionResponse = {
  ...ADMIN_SESSION,
  user: { ...ADMIN_SESSION.user, must_change_password: true },
};

export function ok(body: unknown, status = 200): Response {
  return new Response(JSON.stringify(body), {
    status,
    headers: { "Content-Type": "application/json" },
  });
}

/** The backend's error body: `detail` for a person, `code` for the app (auth-api §3). */
export function fail(status: number, body: Record<string, unknown>): Response {
  return ok(body, status);
}

export function noContent(): Response {
  return new Response(null, { status: 204 });
}

type Handler = () => Response | Promise<Response>;

/**
 * Fakes the backend. Keys are "METHOD /api/v1/path"; a request nobody
 * planned for fails loudly instead of quietly returning nothing.
 */
export function mockApi(routes: Record<string, Handler>) {
  const fetchMock = vi.fn(async (url: string, init?: RequestInit) => {
    const key = `${init?.method ?? "GET"} ${url}`;
    const handler = routes[key];
    if (!handler) throw new TypeError(`no mock route for ${key}`);
    return handler();
  });
  vi.stubGlobal("fetch", fetchMock);
  return fetchMock;
}

/** The headers a given call carried. */
export function headersOf(fetchMock: ReturnType<typeof mockApi>, index: number): Record<string, string> {
  return (fetchMock.mock.calls[index]?.[1]?.headers ?? {}) as Record<string, string>;
}
