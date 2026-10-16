import type { AuthState } from "../context/SessionContext";

/** Starts SessionProvider already signed in, so screen tests need no login round-trip. */
export const SIGNED_IN: AuthState = {
  kind: "signedIn",
  worker: { worker_id: "W-0042", display_name: "Test Worker" },
  mustChangePin: false,
};

export const SIGNED_OUT: AuthState = { kind: "signedOut" };

/** A fetch Response stand-in for a 2xx JSON answer. */
export function ok(body: unknown = {}) {
  return { ok: true, status: 200, json: async () => body };
}

/** A fetch Response stand-in for a failed request, with the backend's error body. */
export function fail(status: number, body: Record<string, unknown>) {
  return { ok: false, status, json: async () => body };
}

type FakeResponse = ReturnType<typeof ok> | ReturnType<typeof fail> | { ok: true; status: 204 };

export const noContent: FakeResponse = { ok: true, status: 204 };

/**
 * Answers fetch by URL ending, so a test doesn't depend on the order in which
 * SessionProvider's jobs request and the screen's own request happen. A
 * function value is called on every request, letting a test change its answer.
 */
export function mockRoutes(
  mockFetch: jest.Mock,
  routes: Record<string, FakeResponse | (() => FakeResponse)>,
): void {
  mockFetch.mockImplementation(async (url: string) => {
    for (const [suffix, response] of Object.entries(routes)) {
      if (url.endsWith(suffix)) {
        return typeof response === "function" ? response() : response;
      }
    }
    throw new TypeError(`Network request failed (no mock route for ${url})`);
  });
}
