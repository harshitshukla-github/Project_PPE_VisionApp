import { describe, expect, it } from "vitest";

import { ApiError } from "./client";
import { formatWait, isOutage, problemFrom, SessionNotKeptError } from "./problems";

describe("formatWait", () => {
  it.each([
    [1, "1 second"],
    [40, "40 seconds"],
    [60, "1 minute"],
    [61, "2 minutes"],
    [840, "14 minutes"],
  ])("%i s -> %s", (seconds, expected) => {
    expect(formatWait(seconds)).toBe(expected);
  });
});

describe("isOutage: the server couldn't be reached, or couldn't answer", () => {
  it.each([0, 502, 503, 504])("status %i is an outage", (status) => {
    expect(isOutage(new ApiError(status, "x"))).toBe(true);
  });

  it.each([401, 403, 404, 422, 500])("status %i is the server answering, not an outage", (status) => {
    expect(isOutage(new ApiError(status, "x"))).toBe(false);
  });

  it("an error that didn't come from a request isn't one", () => {
    expect(isOutage(new SyntaxError("Unexpected token <"))).toBe(false);
  });
});

describe("problemFrom (auth-api §3, by `code`)", () => {
  it("401 invalid_credentials -> wrong email or password", () => {
    const err = new ApiError(401, "whatever the server wrote", { code: "invalid_credentials" });
    expect(problemFrom(err, "x")).toEqual({ message: "Wrong email or password.", tone: "error" });
  });

  it("423 account_locked shows when to retry", () => {
    const err = new ApiError(423, "locked", { code: "account_locked", retryAfterS: 840 });
    expect(problemFrom(err, "x").message).toContain("Try again in 14 minutes");
  });

  it("423 without retry_after_s still points to an admin", () => {
    const err = new ApiError(423, "locked", { code: "account_locked" });
    expect(problemFrom(err, "x").message).toContain("Ask an admin");
  });

  it("429 says how long to wait", () => {
    const err = new ApiError(429, "slow down", { code: "too_many_attempts", retryAfterS: 30 });
    expect(problemFrom(err, "x").message).toBe("Too many attempts. Wait 30 seconds and try again.");
  });

  it("403 account_disabled points to an admin", () => {
    const err = new ApiError(403, "off", { code: "account_disabled" });
    expect(problemFrom(err, "x").message).toContain("Ask an admin");
  });

  it("403 csrf_failed asks for a reload", () => {
    const err = new ApiError(403, "stale", { code: "csrf_failed" });
    expect(problemFrom(err, "x").message).toContain("Reload");
  });

  it("422 weak_password shows the server's own sentence", () => {
    const err = new ApiError(422, "Use at least 12 characters.", { code: "weak_password" });
    expect(problemFrom(err, "x")).toEqual({ message: "Use at least 12 characters.", tone: "error" });
  });

  it("any 503 is an amber 'try again shortly' notice", () => {
    const problem = problemFrom(new ApiError(503, "Service Unavailable"), "x");
    expect(problem.tone).toBe("notice");
    expect(problem.message).toContain("Try again shortly");
  });

  it.each([
    // The backend's two sentences (src/auth/errors.py, src/auth/deps.py): a moment, or not at all.
    ["The server can't reach its database right now. Try again shortly."],
    ["Sign-in is unavailable: the server has no database configured."],
  ])("503 auth_unavailable keeps the server's own reason, in amber: %s", (detail) => {
    const problem = problemFrom(new ApiError(503, detail, { code: "auth_unavailable" }), "x");

    expect(problem).toEqual({ message: detail, tone: "notice" });
  });

  it("a 502 or 504 on a save that may have gone through keeps the client's 'may or may not have been saved'", () => {
    const err = new ApiError(504, "The server didn't answer properly, so this may or may not have been saved.", {
      unsure: true,
    });

    expect(problemFrom(err, "x")).toEqual({ message: err.message, tone: "notice" });
  });

  it.each([
    [502, "Bad Gateway"],
    [504, "Gateway Timeout"],
  ])("a %i from whatever sits in front of the server is the same notice, not the raw '%s'", (status, text) => {
    const problem = problemFrom(new ApiError(status, text), "x");

    expect(problem).toEqual({ message: "The server can't answer right now. Try again shortly.", tone: "notice" });
  });

  it("an unreachable server keeps its own message, in amber: an outage isn't the person's doing", () => {
    const err = new ApiError(0, "Can't reach the server.");
    expect(problemFrom(err, "x")).toEqual({ message: "Can't reach the server.", tone: "notice" });
  });

  describe("a sign-in the browser didn't keep", () => {
    it("on a page that isn't HTTPS: names the address and says to use HTTPS", () => {
      const problem = problemFrom(
        new SessionNotKeptError(true, "http://192.168.29.210:8000", "http://localhost:8000/admin/"),
        "x",
      );

      expect(problem.tone).toBe("error");
      expect(problem.message).toContain("You signed in, but your browser didn't keep the session");
      expect(problem.message).toContain("http://192.168.29.210:8000 isn't a secure (HTTPS) address");
      expect(problem.message).toContain("https://");
      expect(problem.message).not.toContain("session ended"); // not the ordinary expiry
    });

    it("offers localhost only for the computer that runs the server, with the port and path to open there", () => {
      // Read on another device, a bare "http://localhost" would be that device, not the server.
      const problem = problemFrom(
        new SessionNotKeptError(true, "http://192.168.29.210:8000", "http://localhost:8000/admin/"),
        "x",
      );

      expect(problem.message).toContain("on the computer that runs the server, at http://localhost:8000/admin/");
    });

    it("on a secure page: says to check cookies, and doesn't blame the address", () => {
      const problem = problemFrom(
        new SessionNotKeptError(false, "https://dashboard.example.com", "http://localhost/admin/"),
        "x",
      );

      expect(problem.message).toContain("didn't keep the session");
      expect(problem.message).toContain("allows cookies");
      expect(problem.message).not.toContain("isn't a secure");
      expect(problem.message).not.toContain("localhost");
    });
  });

  it("a 404 without a code (FastAPI's own 'no such route') says the server doesn't have this part yet", () => {
    const problem = problemFrom(new ApiError(404, "Not Found"), "x");
    expect(problem).toEqual({
      message: "This server doesn't have this part of the dashboard yet. It needs updating.",
      tone: "notice",
    });
  });

  it("a 404 with the contract's not_found code keeps the server's own sentence", () => {
    expect(problemFrom(new ApiError(404, "No such check.", { code: "not_found" }), "x")).toEqual({
      message: "No such check.",
      tone: "error",
    });
  });

  it("uses the fallback for errors that aren't ApiErrors", () => {
    expect(problemFrom(new Error("boom"), "Something broke.")).toEqual({
      message: "Something broke.",
      tone: "error",
    });
  });
});
