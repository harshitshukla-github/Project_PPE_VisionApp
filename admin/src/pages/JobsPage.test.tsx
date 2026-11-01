import { render, screen, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { beforeEach, describe, expect, it } from "vitest";

import type { MockBackend } from "../mock/backend";
import { startMockBackend } from "../testing/harness";
import { JobsPage } from "./JobsPage";

let backend: MockBackend;
beforeEach(() => {
  backend = startMockBackend();
});

const order = () => [...backend.state.jobs].sort((a, b) => a.position - b.position).map((j) => j.key);

async function open() {
  render(<JobsPage />);
  await screen.findByText("Welder");
  return userEvent.setup();
}

describe("JobsPage", () => {
  it("lists every job in the order the app shows them, with its key and status", async () => {
    await open();

    const rows = screen.getAllByRole("row").slice(1);
    expect(rows.map((r) => within(r).getAllByRole("cell")[2].textContent)).toEqual(order());
    expect(within(rows[0]).getByText("Active")).toBeInTheDocument();
  });

  it("moves a job up; the order is saved and the job versions stay as they were", async () => {
    const user = await open();
    const versions = backend.state.jobs.map((j) => j.version);

    await user.click(screen.getByRole("button", { name: "Move Welder up" }));

    await screen.findByText("Welder");
    expect(order().slice(0, 2)).toEqual(["welder", "general_labor"]);
    expect(backend.state.jobs.map((j) => j.version)).toEqual(versions);
    expect(within(screen.getAllByRole("row")[1]).getByText("Welder")).toBeInTheDocument();
  });

  it("can't move the first job up or the last one down", async () => {
    await open();

    expect(screen.getByRole("button", { name: "Move General labor up" })).toBeDisabled();
    expect(screen.getByRole("button", { name: "Move Painter down" })).toBeDisabled();
  });

  it("deactivates and reactivates a job (it stays in the list)", async () => {
    const user = await open();

    await user.click(screen.getByRole("button", { name: "Deactivate Painter" }));
    expect(await screen.findByRole("button", { name: "Reactivate Painter" })).toBeInTheDocument();
    expect(backend.state.jobs.find((j) => j.key === "painter")).toMatchObject({ is_active: false, version: 3 });

    await user.click(screen.getByRole("button", { name: "Reactivate Painter" }));
    expect(await screen.findByRole("button", { name: "Deactivate Painter" })).toBeInTheDocument();
  });

  it("explains why the last active job can't be deactivated", async () => {
    const user = await open();
    // Meanwhile another admin deactivates every other job; the server then refuses to deactivate this one.
    for (const job of backend.state.jobs.slice(1)) job.is_active = false;
    await user.click(screen.getByRole("button", { name: "Deactivate General labor" }));

    expect(await screen.findByText("That would leave the app with no jobs to choose from.")).toBeInTheDocument();
  });

  it("renames a job", async () => {
    const user = await open();

    await user.click(within(screen.getByRole("row", { name: /Painter/ })).getByRole("button", { name: "Rename" }));
    const input = screen.getByLabelText("New name for Painter");
    await user.clear(input);
    await user.type(input, "Painter and decorator");
    await user.click(screen.getByRole("button", { name: "Save name" }));

    expect(await screen.findByText("Painter and decorator")).toBeInTheDocument();
    expect(backend.state.jobs.find((j) => j.key === "painter")!.display_name).toBe("Painter and decorator");
  });

  it("shows their version when someone else changed the job first (409), instead of overwriting it", async () => {
    const user = await open();
    const painter = backend.state.jobs.find((j) => j.key === "painter")!;
    painter.display_name = "Painter (renamed elsewhere)";
    painter.version += 1;

    await user.click(screen.getByRole("button", { name: "Deactivate Painter" }));

    expect(await screen.findByText(/Someone changed Painter \(renamed elsewhere\) while you were working/)).toBeInTheDocument();
    expect(painter.is_active).toBe(true); // not overwritten
    expect(screen.getByText("Painter (renamed elsewhere)")).toBeInTheDocument();
  });

  describe("adding a job", () => {
    it("adds it last, with the required equipment ticked", async () => {
      const user = await open();
      expect(screen.getByRole("button", { name: "Add job" })).toBeDisabled();

      await user.type(screen.getByLabelText(/^Key/), "scaffolder");
      await user.type(screen.getByLabelText("Name"), "Scaffolder");
      await user.click(screen.getByRole("checkbox", { name: "Hardhat" }));
      await user.click(screen.getByRole("checkbox", { name: "Boots" }));
      await user.click(screen.getByRole("button", { name: "Add job" }));

      expect(await screen.findByText("Scaffolder")).toBeInTheDocument();
      expect(order().at(-1)).toBe("scaffolder");
      expect(backend.state.jobs.at(-1)!.requirements).toEqual([
        { item: "hardhat", mode: "required" },
        { item: "boots", mode: "required" },
      ]);
      expect(screen.getByLabelText(/^Key/)).toHaveValue(""); // ready for the next one
    });

    it("puts the server's sentence next to the key it is about", async () => {
      const user = await open();
      await user.type(screen.getByLabelText(/^Key/), "Bad Key");
      await user.type(screen.getByLabelText("Name"), "Whatever");
      await user.click(screen.getByRole("checkbox", { name: "Hardhat" }));

      await user.click(screen.getByRole("button", { name: "Add job" }));

      expect(await screen.findByText(/A job key is lowercase letters/)).toBeInTheDocument();
      expect(screen.getByLabelText(/^Key/)).toHaveAttribute("aria-invalid", "true");
    });

    it("refuses a job with nothing required, saying so by the equipment list", async () => {
      const user = await open();
      await user.type(screen.getByLabelText(/^Key/), "empty_job");
      await user.type(screen.getByLabelText("Name"), "Empty");

      await user.click(screen.getByRole("button", { name: "Add job" }));

      expect(await screen.findByText("A job needs at least one required item.")).toBeInTheDocument();
      expect(backend.state.jobs.some((j) => j.key === "empty_job")).toBe(false);
    });

    it("warns when a required item can only be judged by Gemini", async () => {
      const user = await open();
      await user.type(screen.getByLabelText(/^Key/), "welder_two");
      await user.type(screen.getByLabelText("Name"), "Welder two");
      await user.click(screen.getByRole("checkbox", { name: "Hardhat" }));
      await user.click(screen.getByRole("checkbox", { name: "Face shield" }));

      await user.click(screen.getByRole("button", { name: "Add job" }));

      expect(await screen.findByText(/Face shield can only be judged by Gemini/)).toBeInTheDocument();
    });
  });
});
