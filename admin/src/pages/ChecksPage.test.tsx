import { render, screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, beforeEach, describe, expect, it } from "vitest";

import { setAdminTransport, type DashboardUser } from "../api/client";
import type { MockBackend } from "../mock/backend";
import { ADMIN, SUPERVISOR_B, startMockBackend } from "../testing/harness";
import { ChecksPage } from "./ChecksPage";

let backend: MockBackend;

beforeEach(() => {
  backend = startMockBackend();
  window.location.hash = "";
});
afterEach(() => {
  window.location.hash = "";
});

/** The page and the stand-in as the same user, so they agree on who is signed in. */
async function open(user: DashboardUser = ADMIN) {
  backend = startMockBackend(user);
  render(<ChecksPage user={user} />);
  await waitFor(() => expect(screen.getByTestId("range")).toBeInTheDocument());
  return userEvent.setup();
}

const range = () => screen.getByTestId("range").textContent;
/** The data rows (the header row has no row header). */
const rows = () => screen.getAllByRole("row").filter((r) => within(r).queryByRole("rowheader"));
const ids = () => rows().map((r) => within(r).getByRole("rowheader").textContent);

describe("ChecksPage as a supervisor", () => {
  it("opens on the checks waiting for review at their own site: needs review, not reviewed", async () => {
    await open(SUPERVISOR_B);

    const waiting = backend.state.checks.filter((c) => c.site === "pune_tower_b" && c.verdict === "NEEDS_REVIEW" && !c.review);
    expect(waiting.length).toBeGreaterThan(0);
    expect(screen.getByText(/Showing the ones waiting for review/)).toBeInTheDocument();
    expect(ids()).toEqual(waiting.sort((a, b) => (a.at < b.at ? 1 : -1)).map((c) => `#${c.id}`));
    for (const row of rows()) {
      expect(row).toHaveTextContent("Needs review");
      expect(row).toHaveTextContent("Waiting");
    }
    expect(screen.queryByRole("columnheader", { name: "Site" })).toBeNull();
    expect(screen.queryByLabelText("Filter by site")).toBeNull();
  });

  it("'Show all' lists every check at their site, and never one from another site or without a site", async () => {
    const user = await open(SUPERVISOR_B);

    await user.click(screen.getByRole("button", { name: "Show all" }));

    const own = backend.state.checks.filter((c) => c.site === "pune_tower_b").length;
    await waitFor(() => expect(range()).toBe(`1–${Math.min(own, 25)} of ${own}`));
    expect(screen.queryByText("Unknown worker")).toBeNull();
  });
});

describe("ChecksPage as an admin", () => {
  it("opens on every check, newest first, 25 at a time, with the site and checks without a login marked", async () => {
    await open();

    expect(range()).toBe(`1–25 of ${backend.state.checks.length}`);
    expect(ids()[0]).toBe("#1000"); // the newest
    expect(screen.getByRole("columnheader", { name: "Site" })).toBeInTheDocument();
    const noLogin = rows().find((r) => within(r).queryByText("No login"))!;
    expect(within(noLogin).getByText("Unknown worker")).toBeInTheDocument();
  });

  it("filters: a verdict, then the worker from the filter box", async () => {
    const user = await open();

    await user.selectOptions(screen.getByLabelText("Verdict"), "FAIL");
    await user.click(screen.getByRole("button", { name: "Apply" }));
    const fails = backend.state.checks.filter((c) => c.verdict === "FAIL");
    await waitFor(() => expect(range()).toBe(`1–${fails.length} of ${fails.length}`));
    for (const row of rows()) expect(row).toHaveTextContent("Fail");

    await user.selectOptions(screen.getByLabelText("Verdict"), "");
    await user.type(screen.getByLabelText("Worker"), "w-0043");
    await user.click(screen.getByRole("button", { name: "Apply" }));
    const theirs = backend.state.checks.filter((c) => c.worker_id === "W-0043");
    await waitFor(() => expect(range()).toBe(`1–${theirs.length} of ${theirs.length}`));
  });

  it("links each check to its own page", async () => {
    await open();

    expect(within(rows()[0]).getByRole("link", { name: "#1000" })).toHaveAttribute("href", "#/checks/1000");
  });
});

it("opens on one worker's checks when the address names them (the Workers page's link)", async () => {
  window.location.hash = "#/checks?worker=W-0043";

  await open(SUPERVISOR_B);

  const theirs = backend.state.checks.filter((c) => c.worker_id === "W-0043");
  expect(theirs.length).toBeGreaterThan(0);
  expect(range()).toBe(`1–${theirs.length} of ${theirs.length}`);
  expect(screen.getByLabelText("Worker")).toHaveValue("W-0043");
  expect(screen.queryByText(/Showing the ones waiting for review/)).toBeNull(); // not the supervisor's default
});

it("says so plainly when the server doesn't have the checks endpoints yet, instead of 'Not Found'", async () => {
  setAdminTransport(async (url, init) =>
    url.startsWith("/api/v1/admin/checks")
      ? new Response(JSON.stringify({ detail: "Not Found" }), { status: 404 }) // FastAPI's own, no `code`
      : backend.fetch(url, init),
  );
  render(<ChecksPage user={ADMIN} />);

  expect(await screen.findByTestId("problem-notice")).toHaveTextContent("This server doesn't have this part of the dashboard yet");
  expect(screen.queryByText("Not Found")).toBeNull();
});

it("says a half-typed date isn't a day yet, and doesn't apply it", async () => {
  const user = await open();

  await user.type(screen.getByLabelText("From"), "0026-11-02");
  await user.click(screen.getByRole("button", { name: "Apply" }));

  expect(await screen.findByText("Finish typing the date, such as 2026-11-02.")).toBeInTheDocument();
  expect(range()).toBe(`1–25 of ${backend.state.checks.length}`);
});
