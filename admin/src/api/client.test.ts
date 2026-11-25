import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import { ADMIN_SESSION, fail, headersOf, mockApi, noContent, ok } from "../testing/mockApi";
import {
  ApiError,
  REQUEST_TIMEOUT_MS,
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

describe("a 502 or 504 from whatever sits in front of the server", () => {
  // The proxy gave up on the server, which may still have done the work.
  it.each([
    [502, "Bad Gateway"],
    [504, "Gateway Timeout"],
  ])("on a write (%i) says it may or may not have been saved, and is marked unsure", async (status, text) => {
    mockApi({ "POST /api/v1/auth/password": () => new Response(text, { status, statusText: text }) });

    const err = await changePassword("old", "new").catch((e: unknown) => e);

    expect(err).toMatchObject({ status, unsure: true });
    expect((err as ApiError).message).toContain("may or may not have been saved");
  });

  it("on a read is an ordinary outage: nothing was changed", async () => {
    mockApi({ "GET /api/v1/auth/me": () => new Response("Gateway Timeout", { status: 504, statusText: "Gateway Timeout" }) });

    const err = await getSession().catch((e: unknown) => e);

    expect(err).toMatchObject({ status: 504, unsure: false });
  });

  it("a 503 on a write is not unsure: our server refused it, so nothing was saved", async () => {
    mockApi({
      "POST /api/v1/auth/password": () =>
        fail(503, { detail: "The server can't reach its database right now. Try again shortly.", code: "auth_unavailable" }),
    });

    const err = await changePassword("old", "new").catch((e: unknown) => e);

    expect(err).toMatchObject({ status: 503, unsure: false, code: "auth_unavailable" });
  });

  it.each([
    [502, "Bad Gateway"],
    [504, "Gateway Timeout"],
  ])("on sign-in (%i) is an ordinary outage: signing in saves nothing, and trying again is harmless", async (status, text) => {
    mockApi({ "POST /api/v1/auth/login": () => new Response(text, { status, statusText: text }) });

    const err = await login("admin@example.com", "pw").catch((e: unknown) => e);

    expect(err).toMatchObject({ status, unsure: false });
    expect((err as ApiError).message).not.toContain("may or may not have been saved");
  });
});

describe("a server that takes the request and never answers", () => {
  /** A fetch that hangs until it is aborted, as a stuck server or proxy does. */
  function hangingFetch() {
    const fetchMock = vi.fn(
      (_url: string, init?: RequestInit) =>
        new Promise<Response>((_resolve, reject) => {
          init?.signal?.addEventListener("abort", () => reject(new DOMException("Aborted.", "AbortError")));
        }),
    );
    vi.stubGlobal("fetch", fetchMock);
    return fetchMock;
  }
  beforeEach(() => vi.useFakeTimers());
  afterEach(() => vi.useRealTimers());

  it("a read gives up after REQUEST_TIMEOUT_MS, as an outage (status 0) that says what happened", async () => {
    hangingFetch();
    let settled = false;
    const result = getSession().finally(() => (settled = true));
    const rejected = expect(result).rejects.toMatchObject({ status: 0, message: "The server didn't answer in time." });

    await vi.advanceTimersByTimeAsync(REQUEST_TIMEOUT_MS - 1);
    expect(settled).toBe(false); // not a moment sooner
    await vi.advanceTimersByTimeAsync(1);
    await rejected;
  });

  it("a write that times out says it may or may not have been saved: the server may have done it", async () => {
    hangingFetch();
    const failed = changePassword("old", "new").catch((err: unknown) => err);

    await vi.advanceTimersByTimeAsync(REQUEST_TIMEOUT_MS);

    const err = await failed;
    expect(err).toBeInstanceOf(ApiError);
    expect(err).toMatchObject({ status: 0, unsure: true });
    expect((err as ApiError).message).toContain("may or may not have been saved");
  });

  it("a read that times out is not marked unsure: reading again is harmless", async () => {
    hangingFetch();
    const failed = getSession().catch((err: unknown) => err);

    await vi.advanceTimersByTimeAsync(REQUEST_TIMEOUT_MS);

    expect(await failed).toMatchObject({ status: 0, unsure: false });
  });

  it("a sign-in that times out says only that it timed out: it saves nothing", async () => {
    hangingFetch();
    const failed = login("admin@example.com", "pw").catch((err: unknown) => err);

    await vi.advanceTimersByTimeAsync(REQUEST_TIMEOUT_MS);

    expect(await failed).toMatchObject({ status: 0, unsure: false, message: "The server didn't answer in time." });
  });

  it("an answer in time leaves no timer running", async () => {
    mockApi({ "GET /api/v1/auth/me": () => ok(ADMIN_SESSION) });

    await getSession();

    expect(vi.getTimerCount()).toBe(0);
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

  it("on a quiet check, is reported to the caller only: the check right after a sign-in isn't a session ending", async () => {
    const ended = vi.fn();
    setSessionEndedHandler(ended);
    const fetchMock = mockApi({
      "POST /api/v1/auth/login": () => ok(ADMIN_SESSION),
      "GET /api/v1/auth/me": () => fail(401, { detail: "Sign in.", code: "not_authenticated" }),
      "POST /api/v1/auth/password": () => noContent(),
    });
    await login("admin@example.com", "pw");

    await expect(getSession({ quiet: true })).rejects.toMatchObject({ status: 401, code: "not_authenticated" });

    expect(ended).not.toHaveBeenCalled();
    // The CSRF token from the login is still forgotten: there is no session for it to belong to.
    await changePassword("old", "new");
    expect(headersOf(fetchMock, 2)["X-CSRF-Token"]).toBeUndefined();
  });

  it("on an ordinary check, still tells the app", async () => {
    const ended = vi.fn();
    setSessionEndedHandler(ended);
    mockApi({ "GET /api/v1/auth/me": () => fail(401, { detail: "Sign in.", code: "not_authenticated" }) });

    await expect(getSession()).rejects.toMatchObject({ status: 401 });

    expect(ended).toHaveBeenCalledTimes(1);
  });
});
