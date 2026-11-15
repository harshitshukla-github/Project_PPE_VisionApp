import { render, screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterAll, beforeAll, beforeEach, describe, expect, it } from "vitest";

import { updateWorker } from "../api/client";
import type { MockBackend } from "../mock/backend";
import { startMockBackend } from "../testing/harness";
import { AuditPage } from "./AuditPage";

let backend: MockBackend;
beforeEach(() => {
  backend = startMockBackend();
});

// The date filter means the person's own days, so these tests run in India (UTC+5:30), where a
// day starts at 18:30 UTC the evening before. The same tests would pass in UTC with or without
// the fix, so the timezone is checked rather than assumed.
const ORIGINAL_TZ = process.env.TZ;
beforeAll(() => {
  process.env.TZ = "Asia/Kolkata";
  expect(new Date(2026, 10, 2).getTimezoneOffset()).toBe(-330);
});
afterAll(() => {
  if (ORIGINAL_TZ === undefined) delete process.env.TZ;
  else process.env.TZ = ORIGINAL_TZ;
});

const range = () => screen.getByTestId("range").textContent;
/** The "Action" column of each log row, newest first. */
const actions = () => screen.getAllByRole("row").slice(1).filter((r) => within(r).queryByRole("button")).map((r) => within(r).getAllByRole("cell")[2].textContent);

async function open() {
  render(<AuditPage />);
  await screen.findByText("worker.create");
  return userEvent.setup();
}

describe("AuditPage", () => {
  it("lists the log newest first, with a total", async () => {
    await open();

    expect(actions()).toEqual(["worker.reset_pin", "settings.update", "job.requirements", "worker.create"]);
    expect(range()).toBe("1–4 of 4");
  });

  it("shows a change made a moment ago at the top, with who made it", async () => {
    await updateWorker("W-0001", { is_active: false }); // goes through the mock, as the admin
    await open();

    expect(actions()[0]).toBe("worker.update");
    const top = screen.getAllByRole("row")[1];
    expect(within(top).getByText("admin@example.com")).toBeInTheDocument();
    expect(within(top).getByText("worker:W-0001")).toBeInTheDocument();
  });

  it("shows what was there before and after when a row is opened", async () => {
    const user = await open();

    await user.click(screen.getByRole("button", { name: "Show details of settings.update on setting:gemini_min_confidence" }));

    const details = screen.getByTestId("details-302");
    expect(within(details).getByRole("heading", { name: "Before" })).toBeInTheDocument();
    expect(details).toHaveTextContent('"value": 0.8');
    await user.click(screen.getByRole("button", { name: "Hide details of settings.update on setting:gemini_min_confidence" }));
    expect(screen.queryByTestId("details-302")).toBeNull();
  });

  it("says 'Nothing recorded' for a reset, where there is nothing to compare", async () => {
    const user = await open();

    await user.click(screen.getByRole("button", { name: "Show details of worker.reset_pin on worker:W-0004" }));

    expect(within(screen.getByTestId("details-303")).getAllByText("Nothing recorded.")).toHaveLength(2);
  });

  it("filters by action, by who (including the command line) and by site, then clears", async () => {
    const user = await open();

    await user.selectOptions(screen.getByLabelText("Action"), "job.requirements");
    await user.click(screen.getByRole("button", { name: "Apply" }));
    await waitFor(() => expect(actions()).toEqual(["job.requirements"]));

    await user.click(screen.getByRole("button", { name: "Clear" }));
    await user.type(screen.getByLabelText("Who"), "cli");
    await user.click(screen.getByRole("button", { name: "Apply" }));
    await waitFor(() => expect(actions()).toEqual(["worker.create"]));

    await user.click(screen.getByRole("button", { name: "Clear" }));
    await user.selectOptions(screen.getByLabelText("Site"), "pune_tower_b");
    await user.click(screen.getByRole("button", { name: "Apply" }));
    await waitFor(() => expect(range()).toBe("1–2 of 2"));
  });

  it("filters by date: 'From' starts that day, 'To' includes that whole day", async () => {
    const user = await open();

    await user.type(screen.getByLabelText("From"), "2026-10-24");
    await user.click(screen.getByRole("button", { name: "Apply" }));
    await waitFor(() => expect(actions()).toEqual(["worker.reset_pin"]));

    await user.click(screen.getByRole("button", { name: "Clear" }));
    await user.type(screen.getByLabelText("To"), "2026-10-19"); // the first entry is at 12:00 that day
    await user.click(screen.getByRole("button", { name: "Apply" }));
    await waitFor(() => expect(actions()).toEqual(["worker.create"]));
  });

  it("reads a picked day as the person's own day: 02:04 on 2 November in India is in 2 November", async () => {
    // 20:34 UTC on 1 November is 02:04 on 2 November in India, and the table shows it as 2 November.
    backend.state.audit.push({
      id: 500, at: "2026-11-01T20:34:00Z", actor: "cli", action: "worker.create", target: "worker:LATE-1",
      site: null, before: null, after: null, ip: null,
    });
    const user = await open();

    await user.type(screen.getByLabelText("From"), "2026-11-02");
    await user.type(screen.getByLabelText("To"), "2026-11-02");
    await user.click(screen.getByRole("button", { name: "Apply" }));
    await waitFor(() => expect(screen.getByText("worker:LATE-1")).toBeInTheDocument()); // found

    await user.clear(screen.getByLabelText("From"));
    await user.clear(screen.getByLabelText("To"));
    await user.type(screen.getByLabelText("From"), "2026-11-01");
    await user.type(screen.getByLabelText("To"), "2026-11-01");
    await user.click(screen.getByRole("button", { name: "Apply" }));
    await screen.findByText("Nothing in the log matches."); // not found under 1 November
    expect(screen.queryByText("worker:LATE-1")).toBeNull();
  });

  it("says when nothing matches", async () => {
    const user = await open();
    await user.type(screen.getByLabelText("Target"), "job:nothing");
    await user.click(screen.getByRole("button", { name: "Apply" }));

    expect(await screen.findByText("Nothing in the log matches.")).toBeInTheDocument();
  });

  it("pages 25 at a time", async () => {
    for (let i = 0; i < 30; i++) {
      backend.state.audit.push({
        id: 400 + i, at: "2026-10-25T10:00:00Z", actor: "cli", action: "worker.create", target: `worker:X-${i}`,
        site: null, before: null, after: null, ip: null,
      });
    }
    const user = await open();
    expect(range()).toBe("1–25 of 34");

    await user.click(screen.getByRole("button", { name: "Next" }));
    await waitFor(() => expect(range()).toBe("26–34 of 34"));
    expect(screen.getByRole("button", { name: "Next" })).toBeDisabled();
  });
});
