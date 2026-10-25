import { render, screen, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { beforeEach, describe, expect, it } from "vitest";

import { ADMIN, SUPERVISOR_B, startMockBackend } from "../testing/harness";
import type { MockBackend } from "../mock/backend";
import { MatrixPage } from "./MatrixPage";

let backend: MockBackend;
beforeEach(() => {
  backend = startMockBackend();
});

const welder = () => backend.state.jobs.find((j) => j.key === "welder")!;

async function open(user = ADMIN) {
  render(<MatrixPage user={user} />);
  await screen.findByRole("columnheader", { name: /Hardhat/ });
  return userEvent.setup();
}

describe("MatrixPage", () => {
  it("shows jobs down the side and items across, with each item's AP50 or 'Gemini only'", async () => {
    await open();

    expect(screen.getByRole("columnheader", { name: /Hardhat.*AP50 0\.91/ })).toBeInTheDocument();
    expect(screen.getByRole("columnheader", { name: /Face shield.*Gemini only/ })).toBeInTheDocument();
    expect(screen.getByLabelText("Welder: Hardhat")).toHaveValue("required");
    expect(screen.getByLabelText("Welder: Face shield")).toHaveValue("pending"); // shown as "Manual"
    expect(screen.getByLabelText("Painter: Boots")).toHaveValue("off");
  });

  it("keeps Save off until a cell changes, then saves the row with its version and shows the new one", async () => {
    const user = await open();
    const save = screen.getByRole("button", { name: "Save Painter" });
    expect(save).toBeDisabled();

    await user.selectOptions(screen.getByLabelText("Painter: Boots"), "required");
    expect(save).toBeEnabled();
    await user.click(save);

    expect(await screen.findByText(/Saved\. It takes effect within 30 seconds/)).toBeInTheDocument();
    expect(backend.state.jobs.find((j) => j.key === "painter")!.requirements).toContainEqual({ item: "boots", mode: "required" });
    expect(within(screen.getByRole("row", { name: /Painter/ })).getByText("version 3")).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Save Painter" })).toBeDisabled();
  });

  it("discards an unsaved row", async () => {
    const user = await open();
    await user.selectOptions(screen.getByLabelText("Painter: Boots"), "required");

    await user.click(screen.getByRole("button", { name: "Discard Painter" }));

    expect(screen.getByLabelText("Painter: Boots")).toHaveValue("off");
  });

  it("badges a required item only Gemini can judge, and explains it after saving", async () => {
    const user = await open();
    await user.selectOptions(screen.getByLabelText("Painter: Face shield"), "required");

    await user.click(screen.getByRole("button", { name: "Save Painter" }));

    expect(await screen.findByText(/can only be judged by Gemini/)).toBeInTheDocument();
    const cell = screen.getByLabelText("Painter: Face shield").closest("td")!;
    expect(within(cell).getByText("Gemini only")).toBeInTheDocument();
  });

  it("shows the server's reason when a row would have no required item", async () => {
    const user = await open();
    for (const item of ["Safety vest", "Mask", "Gloves"]) {
      await user.selectOptions(screen.getByLabelText(`Painter: ${item}`), "off");
    }

    await user.click(screen.getByRole("button", { name: "Save Painter" }));

    expect(await screen.findByText("A job needs at least one required item.")).toBeInTheDocument();
    expect(backend.state.jobs.find((j) => j.key === "painter")!.version).toBe(2); // nothing saved
    expect(screen.getByRole("button", { name: "Save Painter" })).toBeEnabled(); // the edits are still there
  });

  describe("when someone else saved the job first (409 version_conflict)", () => {
    async function conflictOnWelder() {
      const user = await open();
      await user.selectOptions(screen.getByLabelText("Welder: Gloves"), "off"); // my edit
      // ...meanwhile another admin makes Gloves "Manual" and renames the job.
      welder().requirements = welder().requirements.map((r) => (r.item === "gloves" ? { ...r, mode: "pending" as const } : r));
      welder().display_name = "Welder (hot work)";
      welder().version += 1;
      await user.click(screen.getByRole("button", { name: "Save Welder" }));
      return user;
    }

    it("shows what changed, using the job from the 409, and saves nothing", async () => {
      await conflictOnWelder();

      const panel = await screen.findByTestId("conflict-welder");
      expect(panel).toHaveTextContent('It was renamed to "Welder (hot work)"');
      expect(panel).toHaveTextContent("Gloves: was Required, now Manual (you set Off)");
      expect(welder().requirements.find((r) => r.item === "gloves")!.mode).toBe("pending");
    });

    it("'Use their version' drops my edits and shows theirs", async () => {
      const user = await conflictOnWelder();

      await user.click(await screen.findByRole("button", { name: "Use their version" }));

      expect(screen.getByLabelText("Welder (hot work): Gloves")).toHaveValue("pending");
      expect(screen.getByRole("button", { name: "Save Welder (hot work)" })).toBeDisabled();
      expect(screen.queryByTestId("conflict-welder")).toBeNull();
    });

    it("'Keep my edits on top' takes their version and keeps mine, ready to save again", async () => {
      const user = await conflictOnWelder();

      await user.click(await screen.findByRole("button", { name: "Keep my edits on top" }));
      expect(screen.getByLabelText("Welder (hot work): Gloves")).toHaveValue("off");
      await user.click(screen.getByRole("button", { name: "Save Welder (hot work)" }));

      expect(await screen.findByText(/Saved\./)).toBeInTheDocument();
      expect(welder().requirements.some((r) => r.item === "gloves")).toBe(false);
      expect(welder().display_name).toBe("Welder (hot work)"); // their rename survives
    });
  });

  it("is read-only for a supervisor: no selects, no Save", async () => {
    backend = startMockBackend(SUPERVISOR_B);
    await open(SUPERVISOR_B);

    expect(screen.getByText(/Only an admin can change it/)).toBeInTheDocument();
    expect(screen.queryByRole("combobox")).toBeNull();
    expect(screen.queryByRole("button", { name: /^Save/ })).toBeNull();
    expect(within(screen.getByRole("row", { name: /Welder/ })).getAllByText("Required").length).toBeGreaterThan(0);
  });

  it("marks a deactivated job", async () => {
    backend.state.jobs.find((j) => j.key === "painter")!.is_active = false;
    await open();

    expect(within(screen.getByRole("row", { name: /Painter/ })).getByText("inactive")).toBeInTheDocument();
  });
});
