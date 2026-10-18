import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import { ADMIN_SESSION, fail, headersOf, mockApi, noContent, ok } from "../testing/mockApi";
import {
  ApiError,
  changePassword,
  getSession,
  login,
  logout,
  resetClientState,
  setSessionEndedHandler,
} from "./client";

beforeEach(() => resetClientState());
afterEach(() => vi.unstubAllGlobals());

describe("CSRF token (auth-api §6)", () => {
  it("is kept from login and sent on writes, never on reads", async () => {
    const fetchMock = mockApi({
      "POST /api/v1/auth/login": () => ok(ADMIN_SESSION),
      "GET /api/v1/auth/me": () => ok(ADMIN_SESSION),
      "POST /api/v1/auth/password": () => noContent(),
    });

    await login("admin@example.com", "pw");
    await getSession();
    await changePassword("old", "new");

    expect(headersOf(fetchMock, 0)["X-CSRF-Token"]).toBeUndefined(); // login: nothing to send yet
    expect(headersOf(fetchMock, 1)["X-CSRF-Token"]).toBeUndefined(); // GET
    expect(headersOf(fetchMock, 2)["X-CSRF-Token"]).toBe("csrf-1");
  });

  it("comes back from GET /auth/me after a page reload", async () => {
    const fetchMock = mockApi({
      "GET /api/v1/auth/me": () => ok({ ...ADMIN_SESSION, csrf_token: "csrf-after-reload" }),
      "POST /api/v1/auth/password": () => noContent(),
    });

    await getSession();
    await changePassword("old", "new");

    expect(headersOf(fetchMock, 1)["X-CSRF-Token"]).toBe("csrf-after-reload");
  });

  it("on 403 csrf_failed, fetches a fresh token and retries the write once", async () => {
    let attempts = 0;
    const fetchMock = mockApi({
      "GET /api/v1/auth/me": () => ok({ ...ADMIN_SESSION, csrf_token: "csrf-fresh" }),
      "POST /api/v1/auth/password": () =>
        ++attempts === 1 ? fail(403, { detail: "Stale.", code: "csrf_failed" }) : noContent(),
    });

    await expect(changePassword("old", "new")).resolves.toBeUndefined();

    const writes = fetchMock.mock.calls.filter(([, init]) => init?.method === "POST");
    const passwordWrites = writes.filter(([url]) => url === "/api/v1/auth/password");
    expect(passwordWrites).toHaveLength(2);
    expect((passwordWrites[1][1]?.headers as Record<string, string>)["X-CSRF-Token"]).toBe("csrf-fresh");
  });

  it("gives up after one retry rather than looping", async () => {
    const fetchMock = mockApi({
      "GET /api/v1/auth/me": () => ok(ADMIN_SESSION),
      "POST /api/v1/auth/password": () => fail(403, { detail: "Stale.", code: "csrf_failed" }),
    });

    await expect(changePassword("old", "new")).rejects.toMatchObject({ code: "csrf_failed" });

    const passwordWrites = fetchMock.mock.calls.filter(([url]) => url === "/api/v1/auth/password");
    expect(passwordWrites).toHaveLength(2);
  });

  it("is forgotten on logout, whatever the server answers", async () => {
    const fetchMock = mockApi({
      "GET /api/v1/auth/me": () => ok(ADMIN_SESSION),
      "POST /api/v1/auth/logout": () => fail(500, { detail: "boom" }),
      "POST /api/v1/auth/password": () => noContent(),
    });
    await getSession();

    await expect(logout()).rejects.toBeInstanceOf(ApiError);
    await changePassword("old", "new");

    expect(headersOf(fetchMock, 2)["X-CSRF-Token"]).toBeUndefined();
  });
});

describe("requests", () => {
  it("go to the same origin under /api/v1 with the cookie, as JSON", async () => {
    const fetchMock = mockApi({ "POST /api/v1/auth/login": () => ok(ADMIN_SESSION) });

    await login("admin@example.com", "pw");

    const [url, init] = fetchMock.mock.calls[0];
    expect(url).toBe("/api/v1/auth/login");
    expect(init?.credentials).toBe("same-origin");
    expect((init?.headers as Record<string, string>)["Content-Type"]).toBe("application/json");
    expect(JSON.parse(init?.body as string)).toEqual({ email: "admin@example.com", password: "pw" });
  });

  it("treats 204 No Content as success", async () => {
    mockApi({ "POST /api/v1/auth/password": () => noContent() });

    await expect(changePassword("old", "new")).resolves.toBeUndefined();
  });
});

describe("errors", () => {
  it("carry the `code`, `retry_after_s` and `field` from the backend's body", async () => {
    mockApi({
      "POST /api/v1/auth/login": () =>
        fail(423, {
          detail: "Too many wrong passwords. Try again in 14 minutes.",
          code: "account_locked",
          retry_after_s: 840,
          field: "email",
        }),
    });

    await expect(login("a@b.c", "pw")).rejects.toMatchObject({
      name: "ApiError",
      status: 423,
      code: "account_locked",
      retryAfterS: 840,
      field: "email",
      message: "Too many wrong passwords. Try again in 14 minutes.",
    });
  });

  it("join FastAPI's own validation list into one sentence", async () => {
    mockApi({
      "POST /api/v1/auth/login": () => fail(422, { detail: [{ msg: "value is not a valid email address" }] }),
    });

    await expect(login("nope", "pw")).rejects.toMatchObject({
      status: 422,
      message: "value is not a valid email address",
    });
  });

  it("report an unreachable server as status 0", async () => {
    vi.stubGlobal("fetch", vi.fn().mockRejectedValue(new TypeError("Failed to fetch")));

    await expect(getSession()).rejects.toMatchObject({ status: 0, message: "Can't reach the server." });
  });
});

describe("a 401 not_authenticated", () => {
  it("tells the app the session ended", async () => {
    const ended = vi.fn();
    setSessionEndedHandler(ended);
    mockApi({
      "POST /api/v1/auth/password": () => fail(401, { detail: "Sign in again.", code: "not_authenticated" }),
    });

    await expect(changePassword("old", "new")).rejects.toMatchObject({ status: 401 });

    expect(ended).toHaveBeenCalledTimes(1);
  });

  it("is not triggered by a wrong password at login (that is invalid_credentials)", async () => {
    const ended = vi.fn();
    setSessionEndedHandler(ended);
    mockApi({
      "POST /api/v1/auth/login": () => fail(401, { detail: "Wrong.", code: "invalid_credentials" }),
    });

    await expect(login("a@b.c", "pw")).rejects.toMatchObject({ code: "invalid_credentials" });

    expect(ended).not.toHaveBeenCalled();
  });
});
