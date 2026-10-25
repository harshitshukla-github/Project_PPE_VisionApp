import { render, screen, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { beforeEach, describe, expect, it } from "vitest";

import type { MockBackend } from "../mock/backend";
import { ADMIN, startMockBackend } from "../testing/harness";
import { UsersPage } from "./UsersPage";

let backend: MockBackend;
beforeEach(() => {
  backend = startMockBackend();
});

const byEmail = (email: string) => backend.state.users.find((u) => u.email === email)!;
/** The table row for one user, found by their exact email (one email can contain another). */
const row = (email: string) => screen.getByText(email, { selector: "small" }).closest("tr")!;

async function open() {
  render(<UsersPage user={ADMIN} />);
  await screen.findByText("Site Supervisor B");
  return userEvent.setup();
}

describe("UsersPage", () => {
  it("lists every user with role, site and status, and marks you", async () => {
    await open();

    expect(within(row(ADMIN.email)).getByText("You")).toBeInTheDocument();
    const supervisor = row("supervisor.b@example.com");
    expect(within(supervisor).getByText("Supervisor")).toBeInTheDocument();
    expect(within(supervisor).getByText("Pune Tower B")).toBeInTheDocument();
    expect(within(supervisor).getByText("Password not changed yet")).toBeInTheDocument();
    expect(within(row("second.admin@example.com")).getByText("All sites")).toBeInTheDocument();
  });

  it("won't let you act on your own account, and says why", async () => {
    await open();

    for (const name of [`Reset password for ${ADMIN.email}`, `Edit ${ADMIN.email}`, `Deactivate ${ADMIN.email}`]) {
      expect(screen.getByRole("button", { name })).toBeDisabled();
    }
    expect(screen.getByRole("button", { name: `Deactivate ${ADMIN.email}` })).toHaveAttribute("title", "You can't deactivate yourself.");
  });

  describe("adding a user", () => {
    it("a supervisor needs a site; the temporary password is shown once", async () => {
      const user = await open();
      await user.type(screen.getByLabelText("Email"), "new.supervisor@example.com");
      await user.type(screen.getByLabelText("Name"), "New Supervisor");
      expect(screen.getByRole("button", { name: "Add user" })).toBeDisabled(); // no site yet

      await user.selectOptions(screen.getByLabelText("Site"), "pune_tower_c");
      await user.click(screen.getByRole("button", { name: "Add user" }));

      const panel = await screen.findByTestId("once-secret");
      expect(within(panel).getByTestId("secret-value").textContent).toMatch(/^\w+-\w+-\d\d-\w+$/);
      expect(panel).toHaveTextContent("replace it when they first sign in");
      expect(byEmail("new.supervisor@example.com")).toMatchObject({ role: "supervisor", site: "pune_tower_c", must_change_password: true });
      expect(within(row("new.supervisor@example.com")).getByText("Pune Tower C")).toBeInTheDocument();

      await user.click(within(panel).getByRole("button", { name: "I've handed it over" }));
      expect(screen.queryByTestId("once-secret")).toBeNull();
    });

    it("an admin has no site, so the site choice goes away", async () => {
      const user = await open();
      await user.selectOptions(screen.getByLabelText("Role"), "admin");
      expect(screen.queryByLabelText("Site")).toBeNull();

      await user.type(screen.getByLabelText("Email"), "new.admin@example.com");
      await user.type(screen.getByLabelText("Name"), "New Admin");
      await user.click(screen.getByRole("button", { name: "Add user" }));

      await screen.findByTestId("once-secret");
      expect(byEmail("new.admin@example.com")).toMatchObject({ role: "admin", site: null });
    });

    it("says an email is taken, next to the email", async () => {
      const user = await open();
      await user.type(screen.getByLabelText("Email"), "supervisor.b@example.com");
      await user.type(screen.getByLabelText("Name"), "Duplicate");
      await user.selectOptions(screen.getByLabelText("Site"), "pune_tower_b");

      await user.click(screen.getByRole("button", { name: "Add user" }));

      expect(await screen.findByText(/already has an account/)).toBeInTheDocument();
      expect(screen.getByLabelText("Email")).toHaveAttribute("aria-invalid", "true");
    });
  });

  it("resets another user's password and shows the temporary one once", async () => {
    const user = await open();

    await user.click(screen.getByRole("button", { name: "Reset password for supervisor.c@example.com" }));

    const panel = await screen.findByTestId("once-secret");
    expect(panel).toHaveTextContent("Site Supervisor C");
    expect(byEmail("supervisor.c@example.com").must_change_password).toBe(true);
  });

  it("deactivates and reactivates another user", async () => {
    const user = await open();

    await user.click(screen.getByRole("button", { name: "Deactivate supervisor.c@example.com" }));
    expect(await screen.findByRole("button", { name: "Reactivate supervisor.c@example.com" })).toBeInTheDocument();
    expect(byEmail("supervisor.c@example.com").is_active).toBe(false);

    await user.click(screen.getByRole("button", { name: "Reactivate supervisor.c@example.com" }));
    expect(await screen.findByRole("button", { name: "Deactivate supervisor.c@example.com" })).toBeInTheDocument();
  });

  it("explains 'no active admin would remain' when the server refuses", async () => {
    const user = await open();
    byEmail(ADMIN.email).is_active = false; // you were deactivated a moment ago; your session is still open

    await user.click(screen.getByRole("button", { name: "Deactivate second.admin@example.com" }));

    expect(await screen.findByText("That would leave no active admin.")).toBeInTheDocument();
    expect(byEmail("second.admin@example.com").is_active).toBe(true);
  });

  it("edits a supervisor's name and moves them to another site", async () => {
    const user = await open();

    await user.click(screen.getByRole("button", { name: "Edit supervisor.b@example.com" }));
    const name = screen.getByLabelText("Name for supervisor.b@example.com");
    await user.clear(name);
    await user.type(name, "Senior Supervisor B");
    await user.selectOptions(screen.getByLabelText("Site for supervisor.b@example.com"), "pune_tower_c");
    await user.click(screen.getByRole("button", { name: "Save" }));

    expect(await screen.findByText("Senior Supervisor B")).toBeInTheDocument();
    expect(byEmail("supervisor.b@example.com")).toMatchObject({ display_name: "Senior Supervisor B", site: "pune_tower_c" });
  });

  it("making someone an admin removes their site", async () => {
    const user = await open();

    await user.click(screen.getByRole("button", { name: "Edit supervisor.c@example.com" }));
    await user.selectOptions(screen.getByLabelText("Role for supervisor.c@example.com"), "admin");
    expect(screen.queryByLabelText("Site for supervisor.c@example.com")).toBeNull();
    await user.click(screen.getByRole("button", { name: "Save" }));

    await screen.findAllByText("All sites");
    expect(byEmail("supervisor.c@example.com")).toMatchObject({ role: "admin", site: null });
  });
});
