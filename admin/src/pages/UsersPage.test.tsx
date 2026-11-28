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

  it("won't let you reset or deactivate your own account, and says why; Edit stays open (§12, 8)", async () => {
    await open();

    for (const name of [`Reset password for ${ADMIN.email}`, `Deactivate ${ADMIN.email}`]) {
      expect(screen.getByRole("button", { name })).toBeDisabled();
    }
    expect(screen.getByRole("button", { name: `Deactivate ${ADMIN.email}` })).toHaveAttribute("title", "You can't deactivate yourself.");
    expect(screen.getByRole("button", { name: `Edit ${ADMIN.email}` })).toBeEnabled();
  });

  it("edits your own name: only the name, and the header is told (§12, 8)", async () => {
    const renamed: string[] = [];
    render(<UsersPage user={ADMIN} onOwnNameChange={(name) => renamed.push(name)} />);
    await screen.findByText("Site Supervisor B");
    const user = userEvent.setup();

    await user.click(screen.getByRole("button", { name: `Edit ${ADMIN.email}` }));
    expect(screen.queryByLabelText(`Role for ${ADMIN.email}`)).toBeNull();
    expect(screen.queryByLabelText(`Site for ${ADMIN.email}`)).toBeNull();
    expect(screen.getByText("You can't change your own role or site.")).toBeInTheDocument();

    const name = screen.getByLabelText(`Name for ${ADMIN.email}`);
    await user.clear(name);
    await user.type(name, "Harshit Shukla");
    await user.click(screen.getByRole("button", { name: "Save" }));

    expect(await within(row(ADMIN.email)).findByText(/Harshit Shukla/)).toBeInTheDocument();
    expect(byEmail(ADMIN.email)).toMatchObject({ display_name: "Harshit Shukla", role: "admin", site: null });
    expect(renamed).toEqual(["Harshit Shukla"]);
  });

  it("someone else's name change isn't passed to the header", async () => {
    const renamed: string[] = [];
    render(<UsersPage user={ADMIN} onOwnNameChange={(name) => renamed.push(name)} />);
    await screen.findByText("Site Supervisor B");
    const user = userEvent.setup();

    await user.click(screen.getByRole("button", { name: "Edit supervisor.b@example.com" }));
    await user.type(screen.getByLabelText("Name for supervisor.b@example.com"), " Jr");
    await user.click(screen.getByRole("button", { name: "Save" }));

    expect(await screen.findByText("Site Supervisor B Jr")).toBeInTheDocument();
    expect(renamed).toEqual([]);
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

  it("asks before resetting another user's password, then shows the temporary one once (§12, 7)", async () => {
    const user = await open();
    const auditBefore = backend.state.audit.length;

    await user.click(screen.getByRole("button", { name: "Reset password for supervisor.c@example.com" }));

    const dialog = screen.getByRole("dialog", { name: "Reset Site Supervisor C's password?" });
    expect(dialog).toHaveTextContent("Their current password stops working and they're signed out of the dashboard.");
    expect(dialog).toHaveTextContent("Site Supervisor C (supervisor.c@example.com)");
    expect(backend.state.audit.length).toBe(auditBefore); // nothing happens before "yes"

    await user.click(within(dialog).getByRole("button", { name: "Reset password" }));

    const panel = await screen.findByTestId("once-secret");
    expect(panel).toHaveTextContent("Site Supervisor C");
    expect(byEmail("supervisor.c@example.com").must_change_password).toBe(true);
    expect(screen.queryByRole("dialog")).toBeNull();
  });

  it("cancelling a reset or a deactivation changes nothing", async () => {
    const user = await open();
    const auditBefore = backend.state.audit.length;

    await user.click(screen.getByRole("button", { name: "Reset password for supervisor.c@example.com" }));
    await user.click(within(screen.getByRole("dialog")).getByRole("button", { name: "Cancel" }));
    await user.click(screen.getByRole("button", { name: "Deactivate supervisor.c@example.com" }));
    await user.click(within(screen.getByRole("dialog")).getByRole("button", { name: "Cancel" }));

    expect(screen.queryByRole("dialog")).toBeNull();
    expect(screen.queryByTestId("once-secret")).toBeNull();
    expect(byEmail("supervisor.c@example.com").is_active).toBe(true);
    expect(backend.state.audit.length).toBe(auditBefore);
  });

  it("deactivates another user after asking; reactivating doesn't ask (§12, 7)", async () => {
    const user = await open();

    await user.click(screen.getByRole("button", { name: "Deactivate supervisor.c@example.com" }));
    const dialog = screen.getByRole("dialog", { name: "Deactivate Site Supervisor C?" });
    expect(dialog).toHaveTextContent(
      "Site Supervisor C (supervisor.c@example.com) is signed out of the dashboard at once and can't sign in again until an admin reactivates them.",
    );
    await user.click(within(dialog).getByRole("button", { name: "Deactivate" }));
    expect(await screen.findByRole("button", { name: "Reactivate supervisor.c@example.com" })).toBeInTheDocument();
    expect(byEmail("supervisor.c@example.com").is_active).toBe(false);

    await user.click(screen.getByRole("button", { name: "Reactivate supervisor.c@example.com" }));
    expect(screen.queryByRole("dialog")).toBeNull();
    expect(await screen.findByRole("button", { name: "Deactivate supervisor.c@example.com" })).toBeInTheDocument();
    expect(byEmail("supervisor.c@example.com").is_active).toBe(true);
  });

  it("explains 'no active admin would remain' when the server refuses", async () => {
    const user = await open();
    byEmail(ADMIN.email).is_active = false; // you were deactivated a moment ago; your session is still open

    await user.click(screen.getByRole("button", { name: "Deactivate second.admin@example.com" }));
    await user.click(within(screen.getByRole("dialog")).getByRole("button", { name: "Deactivate" }));

    expect(await screen.findByText("That would leave no active admin.")).toBeInTheDocument();
    expect(byEmail("second.admin@example.com").is_active).toBe(true);
    expect(screen.queryByRole("dialog")).toBeNull();
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
