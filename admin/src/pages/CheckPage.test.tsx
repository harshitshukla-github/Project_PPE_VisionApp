import { render, screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it } from "vitest";

import type { DashboardUser } from "../api/client";
import type { MockBackend } from "../mock/backend";
import { ADMIN, SUPERVISOR_B, startMockBackend } from "../testing/harness";
import { CheckPage } from "./CheckPage";

let backend: MockBackend;

/** The page and the stand-in as the same user. Waits for the check to load. */
async function open(id: string, user: DashboardUser = SUPERVISOR_B) {
  backend = startMockBackend(user);
  render(<CheckPage user={user} id={id} />);
  await screen.findByRole("heading", { name: "Review" });
  return userEvent.setup();
}

const check = (id: number) => backend.state.checks.find((c) => c.id === id)!;

// Seeded (src/mock/seed.ts): 1001 needs review, nobody has reviewed it, at Tower B, photo kept.
// 1006 needs review and was reviewed. 1004 is a FAIL with a photo. 1022 was sent without a
// login (no photo). 1040's photo was deleted.

describe("what the check shows", () => {
  it("the photo, with a box for each thing the detector found, and the same list in words", async () => {
    await open("1001");

    const photo = await screen.findByRole("img", { name: "The photo from check #1001" });
    expect(photo.getAttribute("src")).toMatch(/^data:image\/svg\+xml/); // the stand-in's drawn photo
    expect(screen.getAllByTestId("box")).toHaveLength(check(1001).detections.length);
    const found = screen.getByRole("list", { name: "What the detector found" });
    expect(within(found).getAllByRole("listitem")).toHaveLength(check(1001).detections.length);
  });

  it("what the job required then, which item wasn't confirmed, and Gemini's finding on it", async () => {
    await open("1001");

    const c = check(1001);
    const missing = c.missing_equipment[0];
    const missingName = backend.state.items.find((i) => i.key === missing)!.display_name;
    expect(screen.getByText("What the job required then")).toBeInTheDocument();
    const requirements = screen.getAllByRole("listitem").filter((li) => li.closest(".requirements"));
    expect(requirements.map((li) => li.textContent)).toContainEqual(expect.stringMatching(new RegExp(`^${missingName}.*Not confirmed`)));
    expect(requirements.filter((li) => li.textContent?.includes("Found"))).toHaveLength(c.required.length - 1);

    const findings = screen.getByRole("table");
    expect(within(findings).getByRole("rowheader", { name: missingName })).toBeInTheDocument();
    expect(within(findings).getByText("Uncertain")).toBeInTheDocument();
  });

  it("the three settings the verdict was decided with, as they were then", async () => {
    await open("1001");

    const decided = screen.getByTestId("decided-with");
    expect(decided).toHaveTextContent("Detector threshold0.45");
    expect(decided).toHaveTextContent("Gemini second opinionOn");
    expect(decided).toHaveTextContent(`Gemini minimum confidence${check(1001).gemini_min_confidence}`);
  });

  it("why there's no photo: deleted, or never kept because the check had no login", async () => {
    await open("1040");
    expect(screen.getByTestId("no-photo")).toHaveTextContent("The photo has been deleted");
    expect(screen.queryByRole("img")).toBeNull();
  });

  it("a check without a login says its photo is never kept, and that the worker ID wasn't checked", async () => {
    await open("1022", ADMIN); // no site, so only an admin sees it
    expect(screen.getByTestId("no-photo")).toHaveTextContent("sent without a login, so its photo is never kept");
    expect(screen.getByText(/The worker ID was typed in an app without login/)).toBeInTheDocument();
  });

  it("another site's check is 'No such check' to a supervisor", async () => {
    backend = startMockBackend(SUPERVISOR_B);
    const elsewhere = backend.state.checks.find((c) => c.site === "pune_tower_c")!;
    render(<CheckPage user={SUPERVISOR_B} id={String(elsewhere.id)} />);

    expect(await screen.findByText("No such check.")).toBeInTheDocument();
  });

  it("an address that isn't a check number is 'No such check', without asking the server", async () => {
    backend = startMockBackend(SUPERVISOR_B);
    render(<CheckPage user={SUPERVISOR_B} id="abc" />);

    expect(await screen.findByText("No such check.")).toBeInTheDocument();
  });
});

describe("reviewing", () => {
  it("a supervisor records whether the worker was compliant, with a note; the verdict stays", async () => {
    const user = await open("1001");

    expect(screen.getByRole("button", { name: "Save review" })).toBeDisabled(); // nothing chosen yet
    await user.click(screen.getByRole("radio", { name: "Compliant" }));
    await user.type(screen.getByLabelText("Note (optional)"), "Gloves were in his pocket; he put them on.");
    await user.click(screen.getByRole("button", { name: "Save review" }));

    expect(await screen.findByText("Review saved.")).toBeInTheDocument();
    const review = screen.getByTestId("current-review");
    expect(review).toHaveTextContent("Compliant");
    expect(review).toHaveTextContent(`by ${SUPERVISOR_B.email}`);
    expect(review).toHaveTextContent("Gloves were in his pocket; he put them on.");
    expect(check(1001)).toMatchObject({ verdict: "NEEDS_REVIEW", review: { outcome: "compliant" } });
    expect(screen.queryByRole("button", { name: "Replace review" })).toBeNull(); // not for a supervisor
  });

  it("if someone else reviews it first, shows their review instead of overwriting it", async () => {
    const user = await open("1001");
    await user.click(screen.getByRole("radio", { name: "Not compliant" }));
    // Meanwhile, at another desk:
    check(1001).review = { outcome: "compliant", note: "Seen on site.", by: "supervisor.c@example.com", at: "2026-10-25T11:59:00Z" };

    await user.click(screen.getByRole("button", { name: "Save review" }));

    expect(await screen.findByText(/Someone reviewed this check while you had it open/)).toBeInTheDocument();
    expect(screen.getByTestId("current-review")).toHaveTextContent("by supervisor.c@example.com");
    expect(check(1001).review!.outcome).toBe("compliant"); // theirs stands
  });

  it("a FAIL can be reviewed too, and the page says what that's for", async () => {
    await open("1004");

    expect(screen.getByText(/If the worker was compliant after all, say so here/)).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Save review" })).toBeInTheDocument();
  });

  it("an admin can replace a review; the replaced one goes into the audit log", async () => {
    const user = await open("1006", ADMIN);
    const was = check(1006).review!;
    expect(screen.queryByRole("button", { name: "Save review" })).toBeNull(); // already reviewed

    await user.click(screen.getByRole("button", { name: "Replace review" }));
    await user.click(screen.getByRole("radio", { name: was.outcome === "compliant" ? "Not compliant" : "Compliant" }));
    await user.click(screen.getByRole("button", { name: "Replace review" }));

    await screen.findByText("Review saved.");
    expect(check(1006).review!.by).toBe(ADMIN.email);
    expect(backend.state.audit.at(-1)).toMatchObject({ action: "check.review", before: { outcome: was.outcome } });
  });

  it("a supervisor sees an existing review, with no way to replace it", async () => {
    await open("1006");

    expect(screen.getByTestId("current-review")).toBeInTheDocument();
    expect(screen.queryByRole("button", { name: "Replace review" })).toBeNull();
    expect(screen.queryByRole("button", { name: "Save review" })).toBeNull();
  });

  it("keeps the note to 500 characters, and counts them", async () => {
    const user = await open("1001");

    await user.type(screen.getByLabelText("Note (optional)"), "abc");

    expect(screen.getByLabelText("Note (optional)")).toHaveAttribute("maxLength", "500");
    await waitFor(() => expect(screen.getByText("3/500")).toBeInTheDocument());
  });
});
