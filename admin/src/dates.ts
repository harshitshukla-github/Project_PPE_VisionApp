/**
 * Calendar days, as the person sees them.
 *
 * The audit table shows times in the viewer's own timezone, so a day picked in
 * a date input has to mean that day *there*: from midnight to midnight where
 * the person is, not in UTC. In India, 02:04 on 2 November is 20:34 UTC on
 * 1 November; reading "2 November" as a UTC day would hide it. The API wants
 * instants in UTC with a `Z`, so the local midnights are converted.
 */

/** Local midnight of `day` ("2026-11-02"), plus `plusDays` calendar days, as UTC with a Z. */
function localMidnight(day: string, plusDays: number): string {
  const [year, month, date] = day.split("-").map(Number);
  // `new Date(y, m, d)` is local time, and `d + 1` rolls over month ends and daylight-saving
  // changes correctly: a day isn't always 24 hours.
  return new Date(year, month - 1, date + plusDays).toISOString().replace(".000Z", "Z");
}

/** The start of the picked day, for the API's inclusive `from`. Undefined when no day is picked. */
export const startOfDay = (day: string): string | undefined => (day ? localMidnight(day, 0) : undefined);

/**
 * The start of the day after the picked one, for the API's exclusive `to`:
 * picking 2 November as the last day means everything before 3 November begins.
 */
export const startOfNextDay = (day: string): string | undefined => (day ? localMidnight(day, 1) : undefined);
