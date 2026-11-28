import { render, screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { beforeEach, describe, expect, it } from "vitest";

import type { DashboardUser } from "../api/client";
import type { MockBackend } from "../mock/backend";
import { ADMIN, SUPERVISOR_B, startMockBackend } from "../testing/harness";
import { WorkersPage } from "./WorkersPage";

let backend: MockBackend;
const NOW = () => new Date("2026-10-25T12:00:00Z"); // the mock's clock too, so "locked until" is still ahead

beforeEach(() => {
  backend = startMockBackend();
});

async function open(user: DashboardUser = ADMIN) {
  render(<WorkersPage user={user} now={NOW} />);
  await screen.findByText("W-0001");
  return userEvent.setup();
}

const range = () => screen.getByTestId("range").textContent;
const ids = () => screen.getAllByRole("rowheader").map((h) => h.textContent);
const worker = (id: string) => backend.state.workers.find((w) => w.worker_id === id)!;

describe("WorkersPage as an admin", () => {
  it("pages 25 at a time, with '1–25 of 60', and goes back", async () => {
    const user = await open();
    expect(range()).toBe("1–25 of 60");
    expect(ids()).toHaveLength(25);
    expect(screen.getByRole("button", { name: "Previous" })).toBeDisabled();

    await user.click(screen.getByRole("button", { name: "Next" }));
    await screen.findByText("W-0026");
    expect(range()).toBe("26–50 of 60");

    await user.click(screen.getByRole("button", { name: "Next" }));
    await screen.findByText("W-0051");
    expect(range()).toBe("51–60 of 60");
    expect(screen.getByRole("button", { name: "Next" })).toBeDisabled();

    await user.click(screen.getByRole("button", { name: "Previous" }));
    await screen.findByText("W-0026");
    expect(range()).toBe("26–50 of 60");
  });

  it("searches by ID or name, then clears", async () => {
    const user = await open();

    await user.type(screen.getByLabelText("Search"), "w-0007");
    await user.click(screen.getByRole("button", { name: "Apply" }));
    await waitFor(() => expect(ids()).toEqual(["W-0007"]));
    expect(range()).toBe("1–1 of 1");

    await user.click(screen.getByRole("button", { name: "Clear" }));
    await waitFor(() => expect(range()).toBe("1–25 of 60"));
    expect(screen.getByLabelText("Search")).toHaveValue("");
  });

  it("filters by site and by status", async () => {
    const user = await open();

    await user.selectOptions(screen.getByLabelText("Filter by site"), "pune_tower_c");
    await user.click(screen.getByRole("button", { name: "Apply" }));
    await waitFor(() => expect(range()).toBe("1–20 of 20"));

    await user.selectOptions(screen.getByLabelText("Filter by site"), "");
    await user.selectOptions(screen.getByLabelText("Status"), "false");
    await user.click(screen.getByRole("button", { name: "Apply" }));
    await waitFor(() => expect(range()).toBe("1–8 of 8"));
  });

  it("lists the workers locked out right now, and marks them", async () => {
    const user = await open();
    expect(within(screen.getByRole("row", { name: /W-0004/ })).getByText(/Locked until/)).toBeInTheDocument();

    await user.click(screen.getByLabelText("Locked out now"));
    await user.click(screen.getByRole("button", { name: "Apply" }));

    await waitFor(() => expect(ids()).toEqual(["W-0004", "W-0009"]));
  });

  it("says when nothing matches", async () => {
    const user = await open();
    await user.type(screen.getByLabelText("Search"), "nobody-by-this-name");
    await user.click(screen.getByRole("button", { name: "Apply" }));

    expect(await screen.findByText("No workers match.")).toBeInTheDocument();
    expect(range()).toBe("Nothing to show");
  });

  describe("adding a worker", () => {
    it("needs a site, then shows the one-time PIN once and lists the worker", async () => {
      const user = await open();
      await user.type(screen.getByLabelText(/^Worker ID/), "W-0100");
      await user.type(screen.getByLabelText("Name", { selector: "input[placeholder='Ravi Kumar']" }), "New Worker");
      expect(screen.getByRole("button", { name: "Add worker" })).toBeDisabled(); // no site yet

      await user.selectOptions(screen.getByLabelText("Site"), "pune_tower_b");
      await user.click(screen.getByRole("button", { name: "Add worker" }));

      const panel = await screen.findByTestId("once-secret");
      expect(within(panel).getByTestId("secret-value").textContent).toMatch(/^\d{6}$/);
      expect(panel).toHaveTextContent("shown once");
      expect(worker("W-0100")).toMatchObject({ site: "pune_tower_b", must_change_pin: true });

      await user.click(within(panel).getByRole("button", { name: "I've handed it over" }));
      expect(screen.queryByTestId("once-secret")).toBeNull(); // gone, and nothing on the page still has it
    });

    it("puts 'ID is taken' next to the ID, without saying where", async () => {
      const user = await open();
      await user.type(screen.getByLabelText(/^Worker ID/), "W-0003");
      await user.type(screen.getByLabelText("Name", { selector: "input[placeholder='Ravi Kumar']" }), "Duplicate");
      await user.selectOptions(screen.getByLabelText("Site"), "pune_tower_b");

      await user.click(screen.getByRole("button", { name: "Add worker" }));

      expect(await screen.findByText(/The worker ID W-0003 is taken/)).toBeInTheDocument();
      expect(screen.getByLabelText(/^Worker ID/)).toHaveAttribute("aria-invalid", "true");
    });

    it("explains a badly formed ID by the ID field", async () => {
      const user = await open();
      await user.type(screen.getByLabelText(/^Worker ID/), "-bad id");
      await user.type(screen.getByLabelText("Name", { selector: "input[placeholder='Ravi Kumar']" }), "x");
      await user.selectOptions(screen.getByLabelText("Site"), "pune_tower_b");

      await user.click(screen.getByRole("button", { name: "Add worker" }));

      expect(await screen.findByText(/A worker ID is 2 to 32 letters, digits and hyphens/)).toBeInTheDocument();
    });
  });

  it("resets a PIN after asking: shows a new one once and clears the lock (§12, 7)", async () => {
    const user = await open();
    const name = worker("W-0004").display_name;

    await user.click(screen.getByRole("button", { name: "Reset PIN for W-0004" }));

    const dialog = screen.getByRole("dialog", { name: `Reset ${name}'s PIN?` });
    expect(dialog).toHaveTextContent(
      `Their current PIN stops working and they're signed out of the app. You'll get a one-time PIN to hand over to ${name} (W-0004).`,
    );
    expect(worker("W-0004").locked_until).not.toBeNull(); // nothing happens before "yes"
    await user.click(within(dialog).getByRole("button", { name: "Reset PIN" }));

    const panel = await screen.findByTestId("once-secret");
    expect(panel).toHaveTextContent("W-0004");
    expect(within(panel).getByTestId("secret-value").textContent).toMatch(/^\d{6}$/);
    expect(worker("W-0004").locked_until).toBeNull();
    await waitFor(() => expect(within(screen.getByRole("row", { name: /W-0004/ })).queryByText(/Locked until/)).toBeNull());
  });

  it("deactivates a worker after asking; reactivating doesn't ask (§12, 7)", async () => {
    const user = await open();

    await user.click(screen.getByRole("button", { name: "Deactivate W-0001" }));
    const dialog = screen.getByRole("dialog", { name: "Deactivate Asha Verma?" });
    expect(dialog).toHaveTextContent(
      "Asha Verma (W-0001) is signed out of the app at once and can't sign in again until they're reactivated. Their checks stay.",
    );
    expect(worker("W-0001").is_active).toBe(true); // nothing happens before "yes"
    await user.click(within(dialog).getByRole("button", { name: "Deactivate" }));
    expect(await screen.findByRole("button", { name: "Reactivate W-0001" })).toBeInTheDocument();
    expect(worker("W-0001").is_active).toBe(false);

    await user.click(screen.getByRole("button", { name: "Reactivate W-0001" }));
    expect(screen.queryByRole("dialog")).toBeNull();
    expect(await screen.findByRole("button", { name: "Deactivate W-0001" })).toBeInTheDocument();
    expect(worker("W-0001").is_active).toBe(true);
  });

  it("cancelling a PIN reset or a deactivation changes nothing", async () => {
    const user = await open();
    const lockedUntil = worker("W-0004").locked_until;

    await user.click(screen.getByRole("button", { name: "Reset PIN for W-0004" }));
    await user.click(within(screen.getByRole("dialog")).getByRole("button", { name: "Cancel" }));
    await user.click(screen.getByRole("button", { name: "Deactivate W-0001" }));
    await user.click(within(screen.getByRole("dialog")).getByRole("button", { name: "Cancel" }));

    expect(screen.queryByRole("dialog")).toBeNull();
    expect(screen.queryByTestId("once-secret")).toBeNull();
    expect(worker("W-0004").locked_until).toBe(lockedUntil);
    expect(worker("W-0001").is_active).toBe(true);
  });

  it("renames a worker and moves them to another site", async () => {
    const user = await open();

    await user.click(screen.getByRole("button", { name: "Edit W-0001" }));
    const name = screen.getByLabelText("Name for W-0001");
    await user.clear(name);
    await user.type(name, "Asha V. Verma");
    await user.selectOptions(screen.getByLabelText("Site for W-0001"), "pune_tower_c");
    await user.click(screen.getByRole("button", { name: "Save" }));

    expect(await screen.findByText("Asha V. Verma")).toBeInTheDocument();
    expect(worker("W-0001")).toMatchObject({ display_name: "Asha V. Verma", site: "pune_tower_c" });
  });
});

describe("WorkersPage as a supervisor", () => {
  beforeEach(() => {
    backend = startMockBackend(SUPERVISOR_B);
  });

  it("shows only their own site: no site filter or column, and only that site's workers", async () => {
    await open(SUPERVISOR_B);

    expect(screen.getByText(/Workers at Pune Tower B/)).toBeInTheDocument();
    expect(screen.queryByLabelText("Filter by site")).toBeNull();
    expect(screen.queryByRole("columnheader", { name: "Site" })).toBeNull();
    expect(range()).toBe("1–25 of 40");
    expect(ids()).not.toContain("W-0003"); // W-0003 is at Tower C
  });

  it("adds a worker to their own site without choosing one", async () => {
    const user = await open(SUPERVISOR_B);
    expect(screen.getByText("Site: Pune Tower B")).toBeInTheDocument();

    await user.type(screen.getByLabelText(/^Worker ID/), "W-0100");
    await user.type(screen.getByLabelText("Name", { selector: "input[placeholder='Ravi Kumar']" }), "New Worker");
    await user.click(screen.getByRole("button", { name: "Add worker" }));

    expect(await screen.findByTestId("once-secret")).toBeInTheDocument();
    expect(worker("W-0100").site).toBe("pune_tower_b");
  });

  it("can reset a PIN for a worker at their site", async () => {
    const user = await open(SUPERVISOR_B);

    await user.click(screen.getByRole("button", { name: "Reset PIN for W-0004" }));
    await user.click(within(screen.getByRole("dialog")).getByRole("button", { name: "Reset PIN" }));

    expect(await screen.findByTestId("once-secret")).toBeInTheDocument();
  });
});

describe("WorkersPage and the worker's checks (docs/checks-api.md)", () => {
  /** How many of this worker's photos the stand-in still keeps. */
  const kept = (id: string) => backend.state.checks.filter((c) => c.worker_id === id && c.photo.state === "kept").length;

  it("links each worker to their checks", async () => {
    await open();

    expect(screen.getByRole("link", { name: "Checks of W-0001" })).toHaveAttribute("href", "#/checks?worker=W-0001");
  });

  /** "the 1 stored photo", "all 3 stored photos". */
  const wording = (count: number) => (count === 1 ? "the 1 stored photo" : `all ${count} stored photos`);

  it("an admin deletes a worker's photos after a confirmation that says how many; their checks stay", async () => {
    const user = await open();
    const id = ids().find((w) => w && kept(w) > 0)!;
    const count = kept(id);
    const checks = backend.state.checks.filter((c) => c.worker_id === id).length;

    await user.click(screen.getByRole("button", { name: `Delete photos of ${id}` }));
    const dialog = await screen.findByRole("dialog");
    expect(dialog).toHaveTextContent(`This deletes ${wording(count)} of`);
    expect(dialog).toHaveTextContent("It can't be undone.");
    await user.click(within(dialog).getByRole("button", { name: `Delete ${count} photo${count === 1 ? "" : "s"}` }));

    expect(await screen.findByRole("status")).toHaveTextContent(`Deleted ${count} photo${count === 1 ? "" : "s"} of`);
    expect(kept(id)).toBe(0);
    expect(backend.state.checks.filter((c) => c.worker_id === id)).toHaveLength(checks);
  });

  it("says 'the 1 stored photo' for one, not 'all 1 stored photo'", async () => {
    const user = await open();
    const id = ids().find((w) => w && kept(w) > 0)!;
    // Leave this worker exactly one stored photo.
    backend.state.checks
      .filter((c) => c.worker_id === id && c.photo.state === "kept")
      .slice(1)
      .forEach((c) => (c.photo = { ...c.photo, state: "deleted" }));

    await user.click(screen.getByRole("button", { name: `Delete photos of ${id}` }));

    const dialog = await screen.findByRole("dialog");
    expect(dialog).toHaveTextContent("This deletes the 1 stored photo of");
    expect(dialog).not.toHaveTextContent("all 1");
    expect(within(dialog).getByRole("button", { name: "Delete 1 photo" })).toBeInTheDocument();
  });

  it("a 500 part way says the photos already deleted stay deleted, and asking again deletes the rest", async () => {
    const user = await open();
    const id = ids().find((w) => w && kept(w) > 0)!;
    // The sample data gives each worker one check; give this one a second stored photo.
    const mine = backend.state.checks.find((c) => c.worker_id === id)!;
    const other = backend.state.checks.find((c) => c.photo.state === "kept" && c.worker_id !== id)!;
    Object.assign(other, { worker_id: id, worker_name: mine.worker_name, site: mine.site });
    const count = kept(id);
    expect(count).toBe(2);
    backend.state.photoFileFailsAfter = 1; // a disk error on the second file (checks-api §4.4)

    await user.click(screen.getByRole("button", { name: `Delete photos of ${id}` }));
    await user.click(within(await screen.findByRole("dialog")).getByRole("button", { name: /^Delete \d+ photos?$/ }));

    const problem = await screen.findByTestId("problem-error");
    expect(problem).toHaveTextContent("The ones already deleted stay deleted");
    expect(problem).toHaveTextContent("Delete photos again to delete the rest");
    expect(problem).not.toHaveTextContent("Internal Server Error");
    expect(kept(id)).toBe(count - 1);
    expect(screen.queryByRole("dialog")).toBeNull();

    // Asking again: the confirmation counts what is left, and deletes it.
    await user.click(screen.getByRole("button", { name: `Delete photos of ${id}` }));
    expect(await screen.findByRole("dialog")).toHaveTextContent(`This deletes ${wording(count - 1)} of`);
    await user.click(within(screen.getByRole("dialog")).getByRole("button", { name: /^Delete \d+ photos?$/ }));
    expect(await screen.findByRole("status")).toHaveTextContent(`Deleted ${count - 1} photo`);
    expect(kept(id)).toBe(0);
  });

  it("cancelling the confirmation deletes nothing", async () => {
    const user = await open();
    const id = ids().find((w) => w && kept(w) > 0)!;
    const count = kept(id);

    await user.click(screen.getByRole("button", { name: `Delete photos of ${id}` }));
    await user.click(within(await screen.findByRole("dialog")).getByRole("button", { name: "Cancel" }));

    expect(screen.queryByRole("dialog")).toBeNull();
    expect(kept(id)).toBe(count);
  });

  it("a worker with no stored photos gets a plain note, not a confirmation", async () => {
    const user = await open();
    const id = ids().find((w) => w && kept(w) === 0)!;

    await user.click(screen.getByRole("button", { name: `Delete photos of ${id}` }));

    expect(await screen.findByRole("status")).toHaveTextContent("has no stored photos");
    expect(screen.queryByRole("dialog")).toBeNull();
  });

  it("a supervisor can see a worker's checks, but not delete their photos", async () => {
    backend = startMockBackend(SUPERVISOR_B);
    await open(SUPERVISOR_B);

    expect(screen.getAllByRole("link", { name: /^Checks of / }).length).toBeGreaterThan(0);
    expect(screen.queryByRole("button", { name: /^Delete photos of / })).toBeNull();
  });
});
