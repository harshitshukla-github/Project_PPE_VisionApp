/**
 * Calendar days, as the person sees them.
 *
 * The audit table shows times in the viewer's own timezone, so a day picked in
 * a date input has to mean that day *there*: from midnight to midnight where
 * the person is, not in UTC. In India, 02:04 on 2 November is 20:34 UTC on
 * 1 November; reading "2 November" as a UTC day would hide it. The API wants
 * instants in UTC with a `Z`, so the local midnights are converted.
 */

const DAY = /^(\d{4})-(\d{2})-(\d{2})$/;

// A date box hands over whatever is typed so far: "2026" typed as far as "26" arrives as
// "0026-11-02", and `new Date(26, …)` would read that as 1926. Its last allowed day,
// 275760-09-13, is past what a Date can hold, and the day after it throws. Neither is a day
// anyone means, so only these years count.
const FIRST_YEAR = 1900;
const LAST_YEAR = 9999;

/** Whether `day` is a whole date in a year someone could mean ("2026-11-02"); false for "" too. */
export function isUsableDay(day: string): boolean {
  const match = DAY.exec(day);
  if (!match) return false;
  const year = Number(match[1]);
  return year >= FIRST_YEAR && year <= LAST_YEAR;
}

/** Local midnight of `day` ("2026-11-02"), plus `plusDays` calendar days, as UTC with a Z. */
function localMidnight(day: string, plusDays: number): string | undefined {
  if (!isUsableDay(day)) return undefined;
  const [year, month, date] = day.split("-").map(Number);
  // `new Date(y, m, d)` is local time, and `d + 1` rolls over month ends and daylight-saving
  // changes correctly: a day isn't always 24 hours.
  return new Date(year, month - 1, date + plusDays).toISOString().replace(".000Z", "Z");
}

/** The start of the picked day, for the API's inclusive `from`. Undefined when no usable day is picked. */
export const startOfDay = (day: string): string | undefined => localMidnight(day, 0);

/**
 * Local midnight at the start of the `n`th working day (Monday to Friday)
 * before today. With 3 it is the window `auth-api.md` §5 counts checks without
 * a login over: on a Wednesday, from the Friday before.
 */
export function workingDaysBack(n: number, today: Date = new Date()): Date {
  const day = new Date(today.getFullYear(), today.getMonth(), today.getDate());
  for (let counted = 0; counted < n; ) {
    day.setDate(day.getDate() - 1);
    if (day.getDay() !== 0 && day.getDay() !== 6) counted++;
  }
  return day;
}

/**
 * The start of the day after the picked one, for the API's exclusive `to`:
 * picking 2 November as the last day means everything before 3 November begins.
 */
export const startOfNextDay = (day: string): string | undefined => localMidnight(day, 1);
