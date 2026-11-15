import { afterEach, describe, expect, it } from "vitest";

import { startOfDay, startOfNextDay } from "./dates";

const ORIGINAL_TZ = process.env.TZ;
afterEach(() => {
  if (ORIGINAL_TZ === undefined) delete process.env.TZ;
  else process.env.TZ = ORIGINAL_TZ;
});

/** Run in `zone`, and fail loudly if the timezone didn't really change: a test that silently ran in UTC would pass without the fix. */
function inZone(zone: string, expectedOffsetMinutes: number) {
  process.env.TZ = zone;
  expect(new Date(2026, 10, 2).getTimezoneOffset()).toBe(expectedOffsetMinutes); // minutes *behind* UTC, so India is negative
}

describe("calendar days in the person's own timezone", () => {
  it("in India a day runs from 18:30 UTC the evening before to 18:30 UTC", () => {
    inZone("Asia/Kolkata", -330);

    expect(startOfDay("2026-11-02")).toBe("2026-11-01T18:30:00Z");
    expect(startOfNextDay("2026-11-02")).toBe("2026-11-02T18:30:00Z");
  });

  it("puts 02:04 on 2 November in India (20:34 UTC on 1 November) inside 2 November, not 1 November", () => {
    inZone("Asia/Kolkata", -330);
    const entry = "2026-11-01T20:34:00Z";

    const inDay = (day: string) => entry >= startOfDay(day)! && entry < startOfNextDay(day)!;

    expect(inDay("2026-11-02")).toBe(true);
    expect(inDay("2026-11-01")).toBe(false);
  });

  it("is the same instant on both ends of a UTC day only when the timezone is UTC", () => {
    inZone("UTC", 0);

    expect(startOfDay("2026-11-02")).toBe("2026-11-02T00:00:00Z");
    expect(startOfNextDay("2026-11-02")).toBe("2026-11-03T00:00:00Z");
  });

  it("west of UTC the day starts later in UTC terms", () => {
    inZone("America/Chicago", 360); // November: UTC-6

    expect(startOfDay("2026-11-02")).toBe("2026-11-02T06:00:00Z");
    expect(startOfNextDay("2026-11-02")).toBe("2026-11-03T06:00:00Z");
  });

  it("a day isn't always 24 hours: the clocks go back (25 hours) and forward (23 hours)", () => {
    process.env.TZ = "America/New_York";
    expect(new Date(2026, 10, 1).getTimezoneOffset()).toBe(240); // 1 November 2026 starts on summer time (UTC-4)

    expect(startOfDay("2026-11-01")).toBe("2026-11-01T04:00:00Z");
    expect(startOfNextDay("2026-11-01")).toBe("2026-11-02T05:00:00Z"); // winter time by then (UTC-5): 25 hours later
    expect(startOfDay("2026-03-08")).toBe("2026-03-08T05:00:00Z");
    expect(startOfNextDay("2026-03-08")).toBe("2026-03-09T04:00:00Z"); // summer time: only 23 hours later
  });

  it("rolls over the end of a month and a year", () => {
    inZone("Asia/Kolkata", -330);

    expect(startOfNextDay("2026-11-30")).toBe("2026-11-30T18:30:00Z");
    expect(startOfNextDay("2026-12-31")).toBe("2026-12-31T18:30:00Z");
  });

  it("is undefined when no day is picked", () => {
    expect(startOfDay("")).toBeUndefined();
    expect(startOfNextDay("")).toBeUndefined();
  });
});
