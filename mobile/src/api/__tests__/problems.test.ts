import { ApiError } from "../client";
import { formatWait, problemFrom } from "../problems";

// problems.ts imports ApiError from the client, which imports AsyncStorage.
jest.mock("@react-native-async-storage/async-storage", () =>
  require("@react-native-async-storage/async-storage/jest/async-storage-mock"),
);

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

describe("problemFrom (docs/auth-api.md §3, by `code`)", () => {
  it("401 invalid_credentials -> 'Wrong worker ID or PIN.'", () => {
    const err = new ApiError(401, "whatever the server wrote", "invalid_credentials");
    expect(problemFrom(err, "x")).toEqual({ message: "Wrong worker ID or PIN.", tone: "error" });
  });

  it("423 account_locked shows when to retry", () => {
    const err = new ApiError(423, "locked", "account_locked", 840);
    expect(problemFrom(err, "x").message).toContain("Try again in 14 minutes");
  });

  it("423 account_locked without retry_after_s still points to a supervisor", () => {
    const err = new ApiError(423, "locked", "account_locked");
    expect(problemFrom(err, "x").message).toContain("supervisor");
  });

  it("429 too_many_attempts says how long to wait", () => {
    const err = new ApiError(429, "slow down", "too_many_attempts", 30);
    expect(problemFrom(err, "x").message).toBe("Too many attempts. Wait 30 seconds and try again.");
  });

  it("403 account_disabled points to a supervisor", () => {
    const err = new ApiError(403, "off", "account_disabled");
    expect(problemFrom(err, "x").message).toContain("Ask your supervisor");
  });

  it("422 weak_pin shows the server's own sentence", () => {
    const err = new ApiError(422, "Choose a PIN that is not 111111.", "weak_pin");
    expect(problemFrom(err, "x")).toEqual({
      message: "Choose a PIN that is not 111111.",
      tone: "error",
    });
  });

  it("503 auth_unavailable is an amber 'try again shortly' notice", () => {
    const err = new ApiError(503, "no database", "auth_unavailable");
    expect(problemFrom(err, "x")).toEqual({
      message: "Sign-in is unavailable right now. Try again shortly.",
      tone: "notice",
    });
  });

  it("any other 503 (e.g. job matrix unavailable) is also an amber notice", () => {
    const err = new ApiError(503, "Job requirements are temporarily unavailable.");
    const problem = problemFrom(err, "x");
    expect(problem.tone).toBe("notice");
    expect(problem.message).toContain("Try again shortly");
  });

  it("network failures keep their own message in red", () => {
    const err = new ApiError(0, "Can't reach server — check the URL in Settings.");
    expect(problemFrom(err, "x")).toEqual({
      message: "Can't reach server — check the URL in Settings.",
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
