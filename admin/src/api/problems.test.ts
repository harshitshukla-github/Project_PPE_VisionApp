import { describe, expect, it } from "vitest";

import { ApiError } from "./client";
import { formatWait, problemFrom } from "./problems";

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
    const problem = problemFrom(new ApiError(503, "no database", { code: "auth_unavailable" }), "x");
    expect(problem.tone).toBe("notice");
    expect(problem.message).toContain("Try again shortly");
  });

  it("an unreachable server keeps its own message, in red", () => {
    const err = new ApiError(0, "Can't reach the server.");
    expect(problemFrom(err, "x")).toEqual({ message: "Can't reach the server.", tone: "error" });
  });

  it("uses the fallback for errors that aren't ApiErrors", () => {
    expect(problemFrom(new Error("boom"), "Something broke.")).toEqual({
      message: "Something broke.",
      tone: "error",
    });
  });
});
