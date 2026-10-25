import { render, screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import { App } from "./App";
import { resetClientState, setAdminTransport } from "./api/client";
import { AuthProvider } from "./auth/AuthContext";
import {
  ADMIN_SESSION,
  MUST_CHANGE_SESSION,
  fail,
  headersOf,
  mockApi,
  noContent,
  ok,
} from "./testing/mockApi";
import { startMockBackend } from "./testing/harness";

const NOT_SIGNED_IN = () => fail(401, { detail: "Sign in.", code: "not_authenticated" });

function renderApp() {
  return render(
    <AuthProvider>
      <App />
    </AuthProvider>,
  );
}

async function signInWith(email: string, password: string) {
  const user = userEvent.setup();
  await user.type(await screen.findByLabelText("Email"), email);
  await user.type(screen.getByLabelText("Password"), password);
  await user.click(screen.getByRole("button", { name: "Sign in" }));
  return user;
}

beforeEach(() => {
  resetClientState();
  // The pages behind sign-in ask for lists; these tests are about signing in, so give them empty ones.
  setAdminTransport(async () => ok([]));
  window.location.hash = "";
});
afterEach(() => {
  vi.unstubAllGlobals();
  window.location.hash = "";
});

describe("on page load (GET /auth/me)", () => {
  it("shows a spinner, then the login page when nobody is signed in", async () => {
    mockApi({ "GET /api/v1/auth/me": NOT_SIGNED_IN });

    renderApp();

    expect(screen.getByText("Loading…")).toBeInTheDocument();
    expect(await screen.findByLabelText("Email")).toBeInTheDocument();
    expect(screen.queryByTestId("problem-error")).toBeNull(); // not being signed in yet isn't an error
  });

  it("goes straight to the dashboard when the session is still good", async () => {
    mockApi({ "GET /api/v1/auth/me": () => ok(ADMIN_SESSION) });

    renderApp();

    expect(await screen.findByTestId("whoami")).toHaveTextContent("Test Admin · admin");
    expect(screen.queryByLabelText("Email")).toBeNull();
  });

  it("opens Change password first when the temporary password is still in use", async () => {
    mockApi({ "GET /api/v1/auth/me": () => ok(MUST_CHANGE_SESSION) });

    renderApp();

    expect(await screen.findByText("Choose a new password")).toBeInTheDocument();
    expect(screen.queryByTestId("whoami")).toBeNull();
  });

  it("says so when the server can't be reached", async () => {
    vi.stubGlobal("fetch", vi.fn().mockRejectedValue(new TypeError("Failed to fetch")));

    renderApp();

    expect(await screen.findByText("Can't reach the server.")).toBeInTheDocument();
  });
});

describe("login page", () => {
  it("keeps Sign in disabled until both fields are filled", async () => {
    mockApi({ "GET /api/v1/auth/me": NOT_SIGNED_IN });
    renderApp();
    const user = userEvent.setup();
    const button = await screen.findByRole("button", { name: "Sign in" });
    expect(button).toBeDisabled();

    await user.type(screen.getByLabelText("Email"), "admin@example.com");
    expect(button).toBeDisabled();

    await user.type(screen.getByLabelText("Password"), "x");
    expect(button).toBeEnabled();
  });

  it("signs in, sends the trimmed email, and shows the dashboard", async () => {
    const fetchMock = mockApi({
      "GET /api/v1/auth/me": NOT_SIGNED_IN,
      "POST /api/v1/auth/login": () => ok(ADMIN_SESSION),
    });
    renderApp();

    await signInWith("  admin@example.com ", "correct horse battery staple");

    expect(await screen.findByTestId("whoami")).toHaveTextContent("Test Admin · admin");
    const login = fetchMock.mock.calls.find(([url]) => url === "/api/v1/auth/login")!;
    expect(JSON.parse(login[1]?.body as string)).toEqual({
      email: "admin@example.com",
      password: "correct horse battery staple",
    });
  });

  it("401 invalid_credentials: red message, and the password field is cleared", async () => {
    mockApi({
      "GET /api/v1/auth/me": NOT_SIGNED_IN,
      "POST /api/v1/auth/login": () => fail(401, { detail: "Wrong.", code: "invalid_credentials" }),
    });
    renderApp();

    await signInWith("admin@example.com", "wrong password here");

    expect(await screen.findByText("Wrong email or password.")).toBeInTheDocument();
    expect(screen.getByTestId("problem-error")).toBeInTheDocument();
    expect(screen.getByLabelText("Password")).toHaveValue("");
  });

  it("423 account_locked: shows when to retry", async () => {
    mockApi({
      "GET /api/v1/auth/me": NOT_SIGNED_IN,
      "POST /api/v1/auth/login": () =>
        fail(423, { detail: "Locked.", code: "account_locked", retry_after_s: 840 }),
    });
    renderApp();

    await signInWith("admin@example.com", "wrong password here");

    expect(await screen.findByText(/Try again in 14 minutes/)).toBeInTheDocument();
  });

  it("503: amber 'try again shortly', not a red error", async () => {
    mockApi({
      "GET /api/v1/auth/me": NOT_SIGNED_IN,
      "POST /api/v1/auth/login": () => fail(503, { detail: "No database.", code: "auth_unavailable" }),
    });
    renderApp();

    await signInWith("admin@example.com", "correct horse battery staple");

    expect(await screen.findByTestId("problem-notice")).toHaveTextContent(/Try again shortly/);
    expect(screen.queryByTestId("problem-error")).toBeNull();
  });
});

describe("change password page", () => {
  const NEW = "a-long-new-passphrase";

  async function openChangePage() {
    mockApi({
      "GET /api/v1/auth/me": () => ok(MUST_CHANGE_SESSION),
      "POST /api/v1/auth/password": () => noContent(),
    });
    renderApp();
    await screen.findByText("Choose a new password");
    return userEvent.setup();
  }

  async function fill(user: ReturnType<typeof userEvent.setup>, current: string, next: string, again: string) {
    await user.type(screen.getByLabelText("Current password"), current);
    await user.type(screen.getByLabelText("New password"), next);
    await user.type(screen.getByLabelText("Type the new password again"), again);
  }

  it("keeps Save disabled until the new password is long enough and typed twice the same", async () => {
    const user = await openChangePage();
    const save = screen.getByRole("button", { name: "Save new password" });

    await fill(user, "temporary-pass", "short", "short");
    expect(save).toBeDisabled();
    expect(screen.getByText("Use at least 12 characters.")).toBeInTheDocument();

    await user.clear(screen.getByLabelText("New password"));
    await user.clear(screen.getByLabelText("Type the new password again"));
    await user.type(screen.getByLabelText("New password"), NEW);
    await user.type(screen.getByLabelText("Type the new password again"), NEW + "x");
    expect(save).toBeDisabled();
    expect(screen.getByText("The two passwords don't match.")).toBeInTheDocument();
  });

  it("changes the password with the CSRF token, then shows the dashboard", async () => {
    const fetchMock = mockApi({
      "GET /api/v1/auth/me": () => ok(MUST_CHANGE_SESSION),
      "POST /api/v1/auth/password": () => noContent(),
    });
    renderApp();
    await screen.findByText("Choose a new password");
    const user = userEvent.setup();

    await fill(user, "temporary-pass", NEW, NEW);
    await user.click(screen.getByRole("button", { name: "Save new password" }));

    expect(await screen.findByTestId("whoami")).toBeInTheDocument();
    const index = fetchMock.mock.calls.findIndex(([url]) => url === "/api/v1/auth/password");
    expect(headersOf(fetchMock, index)["X-CSRF-Token"]).toBe("csrf-1");
    expect(JSON.parse(fetchMock.mock.calls[index][1]?.body as string)).toEqual({
      current_password: "temporary-pass",
      new_password: NEW,
    });
  });

  it("422 weak_password: shows the server's reason and stays on the page", async () => {
    mockApi({
      "GET /api/v1/auth/me": () => ok(MUST_CHANGE_SESSION),
      "POST /api/v1/auth/password": () =>
        fail(422, { detail: "A password can't be your email address.", code: "weak_password" }),
    });
    renderApp();
    await screen.findByText("Choose a new password");
    const user = userEvent.setup();

    await fill(user, "temporary-pass", NEW, NEW);
    await user.click(screen.getByRole("button", { name: "Save new password" }));

    expect(await screen.findByText("A password can't be your email address.")).toBeInTheDocument();
    expect(screen.getByText("Choose a new password")).toBeInTheDocument();
  });

  it("a wrong current password says so plainly", async () => {
    mockApi({
      "GET /api/v1/auth/me": () => ok(MUST_CHANGE_SESSION),
      "POST /api/v1/auth/password": () => fail(401, { detail: "Wrong.", code: "invalid_credentials" }),
    });
    renderApp();
    await screen.findByText("Choose a new password");
    const user = userEvent.setup();

    await fill(user, "not-the-temp-one", NEW, NEW);
    await user.click(screen.getByRole("button", { name: "Save new password" }));

    expect(await screen.findByText("That isn't your current password.")).toBeInTheDocument();
  });

  it("an ended session (401 not_authenticated) sends the user back to the login page, saying why", async () => {
    mockApi({
      "GET /api/v1/auth/me": () => ok(MUST_CHANGE_SESSION),
      "POST /api/v1/auth/password": () => fail(401, { detail: "Sign in.", code: "not_authenticated" }),
    });
    renderApp();
    await screen.findByText("Choose a new password");
    const user = userEvent.setup();

    await fill(user, "temporary-pass", NEW, NEW);
    await user.click(screen.getByRole("button", { name: "Save new password" }));

    expect(await screen.findByLabelText("Email")).toBeInTheDocument();
    expect(screen.getByText("Your session ended. Sign in again.")).toBeInTheDocument();
  });
});

describe("signing out", () => {
  it("returns to the login page", async () => {
    mockApi({
      "GET /api/v1/auth/me": () => ok(ADMIN_SESSION),
      "POST /api/v1/auth/logout": () => noContent(),
    });
    renderApp();
    const user = userEvent.setup();

    await user.click(await screen.findByRole("button", { name: "Sign out" }));

    expect(await screen.findByLabelText("Email")).toBeInTheDocument();
  });

  it("signs out locally even when the server can't be reached", async () => {
    const fetchMock = mockApi({
      "GET /api/v1/auth/me": () => ok(ADMIN_SESSION),
      "POST /api/v1/auth/logout": () => ok({}),
    });
    renderApp();
    const user = userEvent.setup();
    const button = await screen.findByRole("button", { name: "Sign out" });
    fetchMock.mockRejectedValueOnce(new TypeError("Failed to fetch"));

    await user.click(button);

    await waitFor(() => expect(screen.getByLabelText("Email")).toBeInTheDocument());
  });
});

describe("the signed-in dashboard", () => {
  const SUPERVISOR_SESSION = {
    ...ADMIN_SESSION,
    user: {
      ...ADMIN_SESSION.user, id: 101, email: "supervisor.b@example.com", display_name: "Site Supervisor B",
      role: "supervisor" as const, site: "pune_tower_b",
    },
  };

  /** Signed in as `session`'s user, with the mock admin backend serving that same user. */
  function openAs(session: typeof ADMIN_SESSION, hash = "") {
    const backend = startMockBackend(session.user);
    mockApi({ "GET /api/v1/auth/me": () => ok(session) });
    window.location.hash = hash;
    renderApp();
    return backend;
  }

  it("gives an admin every page in the navigation", async () => {
    openAs(ADMIN_SESSION);

    const nav = await screen.findByRole("navigation", { name: "Pages" });
    expect(within(nav).getAllByRole("link").map((a) => a.textContent)).toEqual([
      "Matrix", "Jobs", "Workers", "Sites", "Users", "Settings", "Audit log",
    ]);
  });

  it("gives a supervisor only the matrix and the workers", async () => {
    openAs(SUPERVISOR_SESSION);

    const nav = await screen.findByRole("navigation", { name: "Pages" });
    expect(within(nav).getAllByRole("link").map((a) => a.textContent)).toEqual(["Matrix", "Workers"]);
    expect(screen.getByTestId("whoami")).toHaveTextContent("Site Supervisor B · supervisor");
  });

  it("opens the matrix by default, and marks the current page", async () => {
    openAs(ADMIN_SESSION);

    expect(await screen.findByRole("heading", { name: "Job matrix" })).toBeInTheDocument();
    expect(screen.getByRole("link", { name: "Matrix" })).toHaveAttribute("aria-current", "page");
    expect(screen.getByRole("link", { name: "Workers" })).not.toHaveAttribute("aria-current");
  });

  it("follows the link to another page", async () => {
    openAs(ADMIN_SESSION);
    const user = userEvent.setup();

    await user.click(await screen.findByRole("link", { name: "Workers" }));

    expect(await screen.findByRole("heading", { name: "Workers" })).toBeInTheDocument();
    expect(screen.getByRole("link", { name: "Workers" })).toHaveAttribute("aria-current", "page");
  });

  it("opens a page straight from its address", async () => {
    openAs(ADMIN_SESSION, "#/audit");

    expect(await screen.findByRole("heading", { name: "Audit log" })).toBeInTheDocument();
  });

  it("says so for an address that isn't a page", async () => {
    openAs(ADMIN_SESSION, "#/nowhere");

    expect(await screen.findByRole("heading", { name: "Page not found" })).toBeInTheDocument();
  });

  it("tells a supervisor an admin page isn't for them, without asking the server for it", async () => {
    const backend = startMockBackend(SUPERVISOR_SESSION.user);
    const asked = vi.fn(backend.fetch); // every call the dashboard makes to /admin/*
    setAdminTransport(asked);
    mockApi({ "GET /api/v1/auth/me": () => ok(SUPERVISOR_SESSION) });
    window.location.hash = "#/users";
    renderApp();

    expect(await screen.findByRole("heading", { name: "You don't have access to this page" })).toBeInTheDocument();
    expect(screen.queryByRole("heading", { name: "Users" })).toBeNull();
    // The Users page was never opened, so nothing was asked of the server. (Its `403 forbidden`
    // is still the real rule, covered by the mock backend's own tests.)
    expect(asked).not.toHaveBeenCalled();
  });

  it("says plainly that the admin data is sample data while the API is faked", async () => {
    openAs(ADMIN_SESSION);

    expect(await screen.findByTestId("mock-banner")).toHaveTextContent("Sample data");
  });
});

describe("the admin API: real by default, a stand-in only on request", () => {
  it("shows no sample-data banner when the pages talk to the real server", async () => {
    setAdminTransport(null); // the real fetch, which this test answers with empty lists
    mockApi({
      "GET /api/v1/auth/me": () => ok(ADMIN_SESSION),
      "GET /api/v1/admin/items": () => ok([]),
      "GET /api/v1/admin/jobs": () => ok([]),
    });
    renderApp();

    expect(await screen.findByRole("heading", { name: "Job matrix" })).toBeInTheDocument();
    expect(screen.queryByTestId("mock-banner")).toBeNull();
  });

  it("asks the real server for /api/v1/admin/* when no stand-in is switched on", async () => {
    setAdminTransport(null);
    const fetchMock = mockApi({
      "GET /api/v1/auth/me": () => ok(ADMIN_SESSION),
      "GET /api/v1/admin/items": () => ok([]),
      "GET /api/v1/admin/jobs": () => ok([]),
    });
    renderApp();
    await screen.findByRole("heading", { name: "Job matrix" });

    const asked = fetchMock.mock.calls.map(([url]) => url);
    expect(asked).toContain("/api/v1/admin/items");
    expect(asked).toContain("/api/v1/admin/jobs");
  });
});
