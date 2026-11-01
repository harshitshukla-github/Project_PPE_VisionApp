import { render, screen, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { beforeEach, describe, expect, it } from "vitest";

import type { MockBackend } from "../mock/backend";
import { startMockBackend } from "../testing/harness";
import { SitesPage } from "./SitesPage";

let backend: MockBackend;
beforeEach(() => {
  backend = startMockBackend();
});

async function open() {
  render(<SitesPage />);
  await screen.findByText("Pune Tower B");
  return userEvent.setup();
}

describe("SitesPage", () => {
  it("lists every site, active or not, with its active worker count", async () => {
    await open();

    const tower = screen.getByRole("row", { name: /Pune Tower B/ });
    expect(within(tower).getByText("pune_tower_b")).toBeInTheDocument();
    expect(within(tower).getByText("Active")).toBeInTheDocument();
    expect(within(screen.getByRole("row", { name: /Old Depot/ })).getByText("Inactive")).toBeInTheDocument();
  });

  it("won't deactivate a site that still has active workers or supervisors, and says why", async () => {
    const user = await open();

    await user.click(screen.getByRole("button", { name: "Deactivate Pune Tower C" }));

    expect(await screen.findByText(/still has active workers or supervisors/)).toBeInTheDocument();
    expect(backend.state.sites.find((s) => s.key === "pune_tower_c")!.is_active).toBe(true);
  });

  it("reactivates a deactivated site", async () => {
    const user = await open();

    await user.click(screen.getByRole("button", { name: "Reactivate Old Depot" }));

    expect(await screen.findByRole("button", { name: "Deactivate Old Depot" })).toBeInTheDocument();
  });

  it("renames a site", async () => {
    const user = await open();

    await user.click(within(screen.getByRole("row", { name: /Pune Tower C/ })).getByRole("button", { name: "Rename" }));
    const input = screen.getByLabelText("New name for Pune Tower C");
    await user.clear(input);
    await user.type(input, "Pune Tower C (east)");
    await user.click(screen.getByRole("button", { name: "Save name" }));

    expect(await screen.findByText("Pune Tower C (east)")).toBeInTheDocument();
  });

  it("adds a site", async () => {
    const user = await open();

    await user.type(screen.getByLabelText(/^Key/), "pune_tower_d");
    await user.type(screen.getByLabelText("Name"), "Pune Tower D");
    await user.click(screen.getByRole("button", { name: "Add site" }));

    expect(await screen.findByText("Pune Tower D")).toBeInTheDocument();
    expect(backend.state.sites.some((s) => s.key === "pune_tower_d")).toBe(true);
  });

  it("says a key is taken, next to the key", async () => {
    const user = await open();

    await user.type(screen.getByLabelText(/^Key/), "pune_tower_b");
    await user.type(screen.getByLabelText("Name"), "Another");
    await user.click(screen.getByRole("button", { name: "Add site" }));

    expect(await screen.findByText(/The site key "pune_tower_b" is taken/)).toBeInTheDocument();
  });
});
