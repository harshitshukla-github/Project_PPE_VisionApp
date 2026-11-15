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

  /** The item names across the top, in the order drawn. The first column is "Job" and the last is "Row" (the Save buttons). */
  const columnNames = () =>
    screen
      .getAllByRole("columnheader")
      .slice(1, -1)
      .map((h) => (h.textContent ?? "").replace(/(AP50.*|No AP50|Gemini only)$/, ""));

  it("draws the columns in the order the API gives them, whatever that is", async () => {
    // Reverse what the API will return. A page that sorted the items itself, alphabetically
    // or head to toe, would ignore this; one that draws them as given follows it.
    backend.state.items.reverse();
    await open();

    expect(columnNames()).toEqual([
      "Boots", "Gloves", "Harness", "Safety vest", "Mask", "Ear protection", "Face shield", "Goggles", "Hardhat",
    ]);
  });

  it("draws them head to toe when the API does (the stand-in's own order)", async () => {
    await open();

    expect(columnNames()).toEqual([
      "Hardhat", "Goggles", "Face shield", "Ear protection", "Mask", "Safety vest", "Harness", "Gloves", "Boots",
    ]);
  });

  describe("an item with no AP50 figure", () => {
    const header = (name: string) => screen.getByRole("columnheader", { name: new RegExp(`^${name}`) });

    it("is 'Gemini only' only if the detector can't see it; a detectable item with no figure says 'No AP50'", async () => {
      // A model with no metrics file: the backend then sends detector_ap50: null for every item
      // (docs/admin-api.md §4), detectable or not.
      for (const item of backend.state.items) item.detector_ap50 = null;
      await open();

      expect(header("Hardhat")).toHaveTextContent("No AP50");
      expect(header("Hardhat")).not.toHaveTextContent("Gemini only");
      expect(header("Boots")).not.toHaveTextContent("Gemini only");
      for (const name of ["Face shield", "Ear protection", "Harness"]) {
        expect(header(name)).toHaveTextContent("Gemini only");
      }
      // Exactly the three the detector can't see, not all nine.
      expect(screen.getAllByText("Gemini only")).toHaveLength(3);
    });

    it("shows the figure when there is one", async () => {
      backend.state.items.find((i) => i.key === "hardhat")!.detector_ap50 = 0.9;
      await open();

      expect(header("Hardhat")).toHaveTextContent("AP50 0.90");
    });
  });

  it("marks Ear protection 'Gemini only', like Face shield and Harness", async () => {
    await open();

    for (const name of ["Face shield", "Ear protection", "Harness"]) {
      expect(screen.getByRole("columnheader", { name: new RegExp(`${name}.*Gemini only`) })).toBeInTheDocument();
    }
    expect(screen.getByRole("columnheader", { name: /Mask.*AP50/ })).toBeInTheDocument();
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
    for (const item of ["Hardhat", "Safety vest", "Mask", "Goggles"]) { // everything Painter requires
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

  describe("'Keep my edits on top' re-applies only the cells I changed", () => {
    const grade = (key: string) => welder().requirements.find((r) => r.item === key)?.mode ?? "off";

    /** I change `mine`; meanwhile another admin changes `theirs`; I save and meet the conflict. */
    async function conflictWith(mine: Array<[string, string]>, theirs: (job: ReturnType<typeof welder>) => void) {
      const user = await open();
      for (const [item, mode] of mine) await user.selectOptions(screen.getByLabelText(`Welder: ${item}`), mode);
      theirs(welder());
      welder().version += 1;
      await user.click(screen.getByRole("button", { name: "Save Welder" }));
      await screen.findByTestId("conflict-welder");
      return user;
    }
    const removeItem = (key: string) => (job: ReturnType<typeof welder>) => {
      job.requirements = job.requirements.filter((r) => r.item !== key);
    };

    it("keeps their changes to cells I didn't touch, and mine to the ones I did", async () => {
      // I only turn Mask on. They turn Goggles and Boots off (two cells I never touched).
      const user = await conflictWith([["Mask", "required"]], (job) => {
        removeItem("goggles")(job);
        removeItem("boots")(job);
      });

      await user.click(screen.getByRole("button", { name: "Keep my edits on top" }));

      expect(screen.getByLabelText("Welder: Goggles")).toHaveValue("off"); // theirs
      expect(screen.getByLabelText("Welder: Boots")).toHaveValue("off"); // theirs
      expect(screen.getByLabelText("Welder: Mask")).toHaveValue("required"); // mine
      await user.click(screen.getByRole("button", { name: "Save Welder" }));
      await screen.findByText(/Saved\./);
      expect(grade("mask")).toBe("required");
      expect(grade("goggles")).toBe("off"); // their change survived the save
      expect(grade("boots")).toBe("off");
    });

    it("lets my value win on a cell we both changed", async () => {
      const user = await conflictWith([["Gloves", "off"]], (job) => {
        job.requirements = job.requirements.map((r) => (r.item === "gloves" ? { ...r, mode: "pending" as const } : r));
      });

      await user.click(screen.getByRole("button", { name: "Keep my edits on top" }));
      await user.click(screen.getByRole("button", { name: "Save Welder" }));

      await screen.findByText(/Saved\./);
      expect(grade("gloves")).toBe("off");
    });

    it("both together: my cell, their other cell, and a cell we both changed", async () => {
      const user = await conflictWith(
        [["Mask", "required"], ["Gloves", "off"]],
        (job) => {
          removeItem("goggles")(job);
          job.requirements = job.requirements.map((r) => (r.item === "gloves" ? { ...r, mode: "pending" as const } : r));
        },
      );

      await user.click(screen.getByRole("button", { name: "Keep my edits on top" }));
      await user.click(screen.getByRole("button", { name: "Save Welder" }));

      await screen.findByText(/Saved\./);
      expect([grade("mask"), grade("goggles"), grade("gloves")]).toEqual(["required", "off", "off"]);
    });

    it("has nothing left to save when their version already contains my change", async () => {
      const user = await conflictWith([["Gloves", "off"]], removeItem("gloves")); // they made the same change

      await user.click(screen.getByRole("button", { name: "Keep my edits on top" }));

      expect(screen.getByLabelText("Welder: Gloves")).toHaveValue("off");
      expect(screen.getByRole("button", { name: "Save Welder" })).toBeDisabled();
      expect(screen.getByText(/already has your changes/)).toBeInTheDocument();
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
