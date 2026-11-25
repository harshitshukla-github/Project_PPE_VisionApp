import { act, render, screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import { App } from "./App";
import { REQUEST_TIMEOUT_MS, resetClientState, setAdminTransport } from "./api/client";
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

  it("says so in amber when the server can't be reached, rather than showing a sign-in form", async () => {
    vi.stubGlobal("fetch", vi.fn().mockRejectedValue(new TypeError("Failed to fetch")));

    renderApp();

    expect(await screen.findByTestId("problem-notice")).toHaveTextContent("Can't reach the server.");
    expect(screen.queryByTestId("problem-error")).toBeNull(); // an outage isn't the person's doing
    expect(screen.queryByLabelText("Email")).toBeNull();
    expect(screen.getByRole("button", { name: "Try again" })).toBeInTheDocument();
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
    // Like a real server: nobody is signed in until the password is accepted, and the browser then
    // keeps the session cookie, so asking who is signed in afterwards gets an answer.
    let loggedIn = false;
    const fetchMock = mockApi({
      "GET /api/v1/auth/me": () => (loggedIn ? ok(ADMIN_SESSION) : NOT_SIGNED_IN()),
      "POST /api/v1/auth/login": () => {
        loggedIn = true;
        return ok(ADMIN_SESSION);
      },
    });
    renderApp();

    await signInWith("  admin@example.com ", "correct horse battery staple");

    expect(await screen.findByTestId("whoami")).toHaveTextContent("Test Admin · admin");
    const login = fetchMock.mock.calls.find(([url]) => url === "/api/v1/auth/login")!;
    expect(JSON.parse(login[1]?.body as string)).toEqual({
      email: "admin@example.com",
      password: "correct horse battery staple",
    });
    // The page-load check, then one more right after the sign-in, to see the session was kept.
    expect(fetchMock.mock.calls.filter(([url]) => url === "/api/v1/auth/me")).toHaveLength(2);
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
      "POST /api/v1/auth/login": () =>
        fail(503, { detail: "The server can't reach its database right now. Try again shortly.", code: "auth_unavailable" }),
    });
    renderApp();

    await signInWith("admin@example.com", "correct horse battery staple");

    expect(await screen.findByTestId("problem-notice")).toHaveTextContent(/Try again shortly/);
    expect(screen.queryByTestId("problem-error")).toBeNull();
  });

  it("a 504 on sign-in says 'try again shortly', not 'may or may not have been saved': it saves nothing", async () => {
    mockApi({
      "GET /api/v1/auth/me": NOT_SIGNED_IN,
      "POST /api/v1/auth/login": () => new Response("Gateway Timeout", { status: 504, statusText: "Gateway Timeout" }),
    });
    renderApp();

    await signInWith("admin@example.com", "correct horse battery staple");

    expect(await screen.findByTestId("problem-notice")).toHaveTextContent(
      "The server can't answer right now. Try again shortly.",
    );
    expect(screen.queryByText(/may or may not have been saved/)).toBeNull();
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

  it("started at #/password, a forced change ends on the dashboard, not on a second, empty password form", async () => {
    // Someone on the Change password page forgets their password, an admin resets it, and they sign
    // in again in the same tab: the address still says #/password when the forced change opens.
    window.location.hash = "#/password";
    mockApi({
      "GET /api/v1/auth/me": () => ok(MUST_CHANGE_SESSION),
      "POST /api/v1/auth/password": () => noContent(),
    });
    renderApp();
    await screen.findByText("Choose a new password");
    const user = userEvent.setup();

    await fill(user, "temporary-pass", NEW, NEW);
    await user.click(screen.getByRole("button", { name: "Save new password" }));

    expect(await screen.findByRole("heading", { name: "Job matrix" })).toBeInTheDocument();
    expect(screen.queryByLabelText("Current password")).toBeNull(); // no blank form that looks like the save failed
    expect(window.location.hash).toBe("#/matrix");
  });

  it("while the change is forced, an address moved back to #/password doesn't stick", async () => {
    const user = await openChangePage();

    window.location.hash = "#/password";
    await waitFor(() => expect(window.location.hash).toBe("#/matrix"));

    await fill(user, "temporary-pass", NEW, NEW);
    await user.click(screen.getByRole("button", { name: "Save new password" }));
    expect(await screen.findByRole("heading", { name: "Job matrix" })).toBeInTheDocument();
    expect(screen.queryByLabelText("Current password")).toBeNull();
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

  describe("a page that breaks", () => {
    /** The Matrix, given an answer it can't draw: an AP50 that arrives as text, not a number. */
    function openWithABrokenMatrix() {
      const quiet = vi.spyOn(console, "error").mockImplementation(() => {}); // React reports the caught error
      const backend = startMockBackend(ADMIN_SESSION.user);
      backend.state.items[0].detector_ap50 = "0.909" as unknown as number;
      mockApi({ "GET /api/v1/auth/me": () => ok(ADMIN_SESSION) });
      window.location.hash = "#/matrix";
      renderApp();
      return { backend, quiet };
    }

    it("says so in place of that page, and keeps the rest of the dashboard, instead of a blank screen", async () => {
      const { quiet } = openWithABrokenMatrix();

      expect(await screen.findByRole("heading", { name: "Something went wrong on this page" })).toBeInTheDocument();
      expect(screen.getByRole("navigation", { name: "Pages" })).toBeInTheDocument(); // the frame is still there
      expect(screen.getByTestId("whoami")).toBeInTheDocument();
      expect(screen.getByRole("button", { name: "Reload the page" })).toBeInTheDocument();
      quiet.mockRestore();
    });

    it("another page still opens, and coming back tries the broken one again", async () => {
      const { backend, quiet } = openWithABrokenMatrix();
      const user = userEvent.setup();
      await screen.findByRole("heading", { name: "Something went wrong on this page" });

      await user.click(screen.getByRole("link", { name: "Workers" }));
      expect(await screen.findByRole("heading", { name: "Workers" })).toBeInTheDocument();

      backend.state.items[0].detector_ap50 = 0.909; // fixed meanwhile
      await user.click(screen.getByRole("link", { name: "Matrix" }));
      expect(await screen.findByRole("heading", { name: "Job matrix" })).toBeInTheDocument();
      expect(screen.queryByText("Something went wrong on this page")).toBeNull();
      quiet.mockRestore();
    });
  });
});

describe("when the server never answers", () => {
  afterEach(() => vi.useRealTimers());

  it("stops waiting after a while and offers Try again, instead of 'Loading…' for ever", async () => {
    vi.useFakeTimers();
    vi.stubGlobal(
      "fetch",
      vi.fn(
        (_url: string, init?: RequestInit) =>
          new Promise<Response>((_resolve, reject) => {
            init?.signal?.addEventListener("abort", () => reject(new DOMException("Aborted.", "AbortError")));
          }),
      ),
    );
    renderApp();
    expect(screen.getByText("Loading…")).toBeInTheDocument();

    await act(async () => {
      await vi.advanceTimersByTimeAsync(REQUEST_TIMEOUT_MS);
    });

    expect(screen.getByRole("button", { name: "Try again" })).toBeInTheDocument();
    expect(screen.getByTestId("problem-notice")).toHaveTextContent("The server didn't answer in time.");
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

    // The heading comes first, and the page asks for jobs only once items have arrived, so wait
    // for the requests rather than checking the moment the heading appears (it was flaky).
    const asked = () => fetchMock.mock.calls.map(([url]) => url);
    await waitFor(() => expect(asked()).toContain("/api/v1/admin/items"));
    await waitFor(() => expect(asked()).toContain("/api/v1/admin/jobs"));
  });
});

describe("changing your own password later", () => {
  const SUPERVISOR = {
    ...ADMIN_SESSION,
    user: {
      ...ADMIN_SESSION.user, id: 101, email: "supervisor.b@example.com", display_name: "Site Supervisor B",
      role: "supervisor" as const, site: "pune_tower_b",
    },
  };
  const NEW = "a-long-new-passphrase";

  function openAs(session: typeof ADMIN_SESSION, passwordAnswer: () => Response = () => noContent()) {
    startMockBackend(session.user);
    const fetchMock = mockApi({
      "GET /api/v1/auth/me": () => ok(session),
      "POST /api/v1/auth/password": passwordAnswer,
    });
    window.location.hash = "";
    renderApp();
    return fetchMock;
  }

  async function openPage(user: ReturnType<typeof userEvent.setup>) {
    await user.click(await screen.findByRole("link", { name: "Change password" }));
    await screen.findByRole("heading", { name: "Change your password" });
  }

  async function fill(user: ReturnType<typeof userEvent.setup>, current: string, next: string, again: string) {
    await user.type(screen.getByLabelText("Current password"), current);
    await user.type(screen.getByLabelText("New password"), next);
    await user.type(screen.getByLabelText("Type the new password again"), again);
  }

  it.each([
    ["an admin", ADMIN_SESSION],
    ["a supervisor", SUPERVISOR],
  ])("%s has a Change password link in the header", async (_who, session) => {
    openAs(session);

    expect(await screen.findByRole("link", { name: "Change password" })).toHaveAttribute("href", "#/password");
  });

  it("opens inside the usual frame with its own wording, and keeps the navigation as the way back", async () => {
    openAs(ADMIN_SESSION);
    const user = userEvent.setup();

    await openPage(user);

    expect(screen.queryByText(/only for your first sign-in/)).toBeNull(); // that wording is for the forced change
    expect(screen.getByText(/You stay signed in here/)).toBeInTheDocument();
    expect(screen.getByRole("navigation", { name: "Pages" })).toBeInTheDocument();
    expect(screen.getByRole("link", { name: "Change password" })).toHaveAttribute("aria-current", "page");
    expect(screen.getAllByRole("button", { name: "Sign out" })).toHaveLength(1); // the header's; the forced page's own is not shown
    // A page inside the frame, not a second sign-in-style card: the frame's <main> is the only one.
    expect(screen.getAllByRole("main")).toHaveLength(1);
  });

  it("moves focus to 'Password changed', so a screen reader says it worked", async () => {
    openAs(ADMIN_SESSION);
    const user = userEvent.setup();
    await openPage(user);

    await fill(user, "my-old-password-1", NEW, NEW);
    await user.click(screen.getByRole("button", { name: "Save new password" }));

    await screen.findByTestId("password-changed");
    expect(screen.getByRole("heading", { name: "Password changed" })).toHaveFocus();
    expect(screen.getAllByRole("main")).toHaveLength(1);
  });

  it("changes the password with the CSRF token, says so, and offers the way back", async () => {
    const fetchMock = openAs(ADMIN_SESSION);
    const user = userEvent.setup();
    await openPage(user);

    await fill(user, "my-old-password-1", NEW, NEW);
    await user.click(screen.getByRole("button", { name: "Save new password" }));

    const done = await screen.findByTestId("password-changed");
    expect(done).toHaveTextContent("Password changed");
    expect(done).toHaveTextContent("Anywhere else you were signed in has been signed out");
    const call = fetchMock.mock.calls.find(([url]) => url === "/api/v1/auth/password")!;
    expect(JSON.parse(call[1]?.body as string)).toEqual({ current_password: "my-old-password-1", new_password: NEW });
    expect((call[1]?.headers as Record<string, string>)["X-CSRF-Token"]).toBe("csrf-1");

    await user.click(within(done).getByRole("link", { name: "Back to the dashboard" }));
    expect(await screen.findByRole("heading", { name: "Job matrix" })).toBeInTheDocument();
    expect(screen.getByTestId("whoami")).toBeInTheDocument(); // still signed in
  });

  it("can be cancelled", async () => {
    openAs(ADMIN_SESSION);
    const user = userEvent.setup();
    await openPage(user);

    await user.click(screen.getByRole("link", { name: "Cancel" }));

    expect(await screen.findByRole("heading", { name: "Job matrix" })).toBeInTheDocument();
  });

  it("says a wrong current password plainly, and stays on the page", async () => {
    openAs(ADMIN_SESSION, () => fail(401, { detail: "Wrong.", code: "invalid_credentials" }));
    const user = userEvent.setup();
    await openPage(user);

    await fill(user, "not-my-password", NEW, NEW);
    await user.click(screen.getByRole("button", { name: "Save new password" }));

    expect(await screen.findByText("That isn't your current password.")).toBeInTheDocument();
    expect(screen.getByRole("heading", { name: "Change your password" })).toBeInTheDocument();
  });

  it("a 504 on the save says it may or may not have been saved, not just 'try again'", async () => {
    // A retry after a change that did go through would fail on the old password.
    openAs(ADMIN_SESSION, () => new Response("Gateway Timeout", { status: 504, statusText: "Gateway Timeout" }));
    const user = userEvent.setup();
    await openPage(user);

    await fill(user, "my-old-password-1", NEW, NEW);
    await user.click(screen.getByRole("button", { name: "Save new password" }));

    expect(await screen.findByTestId("problem-notice")).toHaveTextContent("may or may not have been saved");
    expect(screen.queryByText(/Try again shortly/)).toBeNull();
  });

  it("shows the server's reason for a weak new password", async () => {
    openAs(ADMIN_SESSION, () => fail(422, { detail: "A password can't be your email address.", code: "weak_password" }));
    const user = userEvent.setup();
    await openPage(user);

    await fill(user, "my-old-password-1", NEW, NEW);
    await user.click(screen.getByRole("button", { name: "Save new password" }));

    expect(await screen.findByText("A password can't be your email address.")).toBeInTheDocument();
  });
});

describe("an old server error doesn't linger on the change-password page", () => {
  const NEW = "a-long-new-passphrase";

  async function afterWrongCurrentPassword() {
    startMockBackend(MUST_CHANGE_SESSION.user);
    mockApi({
      "GET /api/v1/auth/me": () => ok(MUST_CHANGE_SESSION),
      "POST /api/v1/auth/password": () => fail(401, { detail: "Wrong.", code: "invalid_credentials" }),
    });
    renderApp();
    await screen.findByText("Choose a new password");
    const user = userEvent.setup();
    await user.type(screen.getByLabelText("Current password"), "not-the-temp-one");
    await user.type(screen.getByLabelText("New password"), NEW);
    await user.type(screen.getByLabelText("Type the new password again"), NEW);
    await user.click(screen.getByRole("button", { name: "Save new password" }));
    await screen.findByText("That isn't your current password.");
    return user;
  }

  it("goes when the person corrects the field it was about", async () => {
    const user = await afterWrongCurrentPassword();

    await user.type(screen.getByLabelText("Current password"), "x");

    expect(screen.queryByText("That isn't your current password.")).toBeNull();
  });

  it("goes when the person edits any field, even though Save is off: no old error beside a new message", async () => {
    const user = await afterWrongCurrentPassword();

    await user.clear(screen.getByLabelText("New password"));
    await user.type(screen.getByLabelText("New password"), "short");

    expect(screen.getByText("Use at least 12 characters.")).toBeInTheDocument();
    expect(screen.queryByText("That isn't your current password.")).toBeNull();
    expect(screen.getByRole("button", { name: "Save new password" })).toBeDisabled();
  });
});

describe("when the server is down as the page loads", () => {
  /** What the dev proxy, or a load balancer, answers when nothing is behind it. */
  const badGateway = () => new Response("Bad Gateway", { status: 502, statusText: "Bad Gateway" });

  it("says 'try again shortly' in amber, not the raw 'Bad Gateway', and shows no sign-in form", async () => {
    mockApi({ "GET /api/v1/auth/me": badGateway });

    renderApp();

    expect(await screen.findByTestId("problem-notice")).toHaveTextContent("Try again shortly");
    expect(screen.queryByText(/Bad Gateway/)).toBeNull();
    expect(screen.queryByLabelText("Email")).toBeNull();
    expect(screen.queryByLabelText("Password")).toBeNull();
  });

  it("says nothing about whether you're still signed in, since nobody knows yet", async () => {
    mockApi({ "GET /api/v1/auth/me": badGateway });

    renderApp();

    expect(await screen.findByText(/no way to tell yet whether you're still signed in/)).toBeInTheDocument();
    expect(screen.queryByText("Your session ended. Sign in again.")).toBeNull();
  });

  it("'Try again' goes straight to the dashboard once the server is back: the session was fine", async () => {
    let up = false;
    startMockBackend(ADMIN_SESSION.user);
    const fetchMock = mockApi({ "GET /api/v1/auth/me": () => (up ? ok(ADMIN_SESSION) : badGateway()) });
    renderApp();
    const user = userEvent.setup();
    await screen.findByRole("button", { name: "Try again" });

    up = true;
    await user.click(screen.getByRole("button", { name: "Try again" }));

    expect(await screen.findByTestId("whoami")).toHaveTextContent("Test Admin · admin");
    expect(screen.queryByLabelText("Email")).toBeNull(); // never asked to sign in again
    expect(fetchMock.mock.calls.filter(([url]) => url === "/api/v1/auth/me")).toHaveLength(2);
  });

  it("stays on the same screen if the server is still down after 'Try again'", async () => {
    mockApi({ "GET /api/v1/auth/me": badGateway });
    renderApp();
    const user = userEvent.setup();

    await user.click(await screen.findByRole("button", { name: "Try again" }));

    expect(await screen.findByRole("button", { name: "Try again" })).toBeInTheDocument();
    expect(screen.queryByLabelText("Email")).toBeNull();
  });

  it("still shows the sign-in form when the server answers that nobody is signed in", async () => {
    mockApi({ "GET /api/v1/auth/me": () => fail(401, { detail: "Sign in.", code: "not_authenticated" }) });

    renderApp();

    expect(await screen.findByLabelText("Email")).toBeInTheDocument();
    expect(screen.queryByRole("button", { name: "Try again" })).toBeNull();
  });

  it.each([
    ["503", () => fail(503, { detail: "The server can't reach its database right now. Try again shortly.", code: "auth_unavailable" })],
    ["504", () => new Response("Gateway Timeout", { status: 504, statusText: "Gateway Timeout" })],
  ])("a %s is the same screen", async (_status, answer) => {
    mockApi({ "GET /api/v1/auth/me": answer });

    renderApp();

    expect(await screen.findByRole("button", { name: "Try again" })).toBeInTheDocument();
    expect(screen.getByTestId("problem-notice")).toHaveTextContent("Try again shortly");
    expect(screen.queryByLabelText("Email")).toBeNull();
  });

  it("with no database configured, gives the server's reason instead of 'try again shortly'", async () => {
    // A lasting fault, not a moment: "try again shortly" would send the person round in circles.
    mockApi({
      "GET /api/v1/auth/me": () =>
        fail(503, { detail: "Sign-in is unavailable: the server has no database configured.", code: "auth_unavailable" }),
    });

    renderApp();

    expect(await screen.findByTestId("problem-notice")).toHaveTextContent(
      "Sign-in is unavailable: the server has no database configured.",
    );
    expect(screen.queryByText(/Try again shortly/)).toBeNull();
    expect(screen.queryByText(/in a moment/)).toBeNull(); // the page's own words don't promise it either
    expect(screen.getByRole("button", { name: "Try again" })).toBeInTheDocument();
  });
});

describe("when the page-load check fails for a reason that isn't an outage", () => {
  /** The server is up and answering, but something in it broke. */
  const serverError = () =>
    new Response("Internal Server Error", { status: 500, statusText: "Internal Server Error" });

  it("shows the sign-in form with a plain amber note: not the outage screen, not the raw status text", async () => {
    mockApi({ "GET /api/v1/auth/me": serverError });

    renderApp();

    expect(await screen.findByLabelText("Email")).toBeInTheDocument();
    expect(screen.getByTestId("problem-notice")).toHaveTextContent("Couldn't check whether you're signed in");
    expect(screen.queryByText(/Internal Server Error/)).toBeNull();
    expect(screen.queryByText(/isn't answering/)).toBeNull();
    expect(screen.queryByRole("button", { name: "Try again" })).toBeNull();
  });

  it("lets the person sign in from there", async () => {
    let loggedIn = false;
    mockApi({
      "GET /api/v1/auth/me": () => (loggedIn ? ok(ADMIN_SESSION) : serverError()),
      "POST /api/v1/auth/login": () => {
        loggedIn = true;
        return ok(ADMIN_SESSION);
      },
    });
    renderApp();

    await signInWith("admin@example.com", "correct horse battery staple");

    expect(await screen.findByTestId("whoami")).toHaveTextContent("Test Admin · admin");
  });

  it("treats an answer that isn't the API's at all (a page of HTML) the same way", async () => {
    mockApi({ "GET /api/v1/auth/me": () => new Response("<html>not the API</html>", { status: 200 }) });

    renderApp();

    expect(await screen.findByLabelText("Email")).toBeInTheDocument();
    expect(screen.getByTestId("problem-notice")).toHaveTextContent("Couldn't check whether you're signed in");
    expect(screen.queryByRole("button", { name: "Try again" })).toBeNull();
  });
});

describe("when the browser doesn't keep the sign-in cookie (item 5: plain http on a LAN address)", () => {
  /** `isSecureContext` is true on https and on localhost, false on plain http at any other address. */
  function pageIsSecure(secure: boolean) {
    Object.defineProperty(window, "isSecureContext", { value: secure, configurable: true });
  }
  afterEach(() => {
    delete (window as unknown as { isSecureContext?: boolean }).isSecureContext;
  });

  /**
   * A server where nobody is signed in until the password is accepted. `cookieKept` is what the
   * browser does with the session cookie that the login sets: keeps it (so the next call is
   * signed in) or drops it (so the next call is a 401, the failure this item is about).
   */
  function serverWith(options: { cookieKept: boolean; meAfterLogin?: () => Response }) {
    let loggedIn = false;
    return mockApi({
      "GET /api/v1/auth/me": () => {
        if (!loggedIn) return NOT_SIGNED_IN();
        if (options.meAfterLogin) return options.meAfterLogin();
        return options.cookieKept ? ok(ADMIN_SESSION) : NOT_SIGNED_IN();
      },
      "POST /api/v1/auth/login": () => {
        loggedIn = true;
        return ok(ADMIN_SESSION);
      },
    });
  }

  it("on plain http: says the address isn't HTTPS, stays on the sign-in page, and never shows the dashboard", async () => {
    pageIsSecure(false);
    serverWith({ cookieKept: false });
    renderApp();

    await signInWith("admin@example.com", "correct horse battery staple");

    const alert = await screen.findByTestId("problem-error");
    expect(alert).toHaveTextContent("You signed in, but your browser didn't keep the session");
    expect(alert).toHaveTextContent(`${window.location.origin} isn't a secure (HTTPS) address`); // names the page's own address
    // localhost only helps on the server's own computer, so it says so, with this page's port and path.
    const { port, pathname } = window.location;
    expect(alert).toHaveTextContent(
      `on the computer that runs the server, at http://localhost${port ? `:${port}` : ""}${pathname}`,
    );
    expect(screen.queryByText("Your session ended. Sign in again.")).toBeNull();
    expect(screen.getByLabelText("Email")).toBeInTheDocument(); // still the sign-in page
    expect(screen.queryByTestId("whoami")).toBeNull(); // the dashboard never flashed up
  });

  it("a second attempt never shows 'Your session ended' while it is in flight", async () => {
    // On a LAN address no attempt on that page can succeed, so people try again. The check after
    // sign-in must not leave a "session ended" note behind for that second attempt to uncover.
    pageIsSecure(false);
    let held: Promise<void> | null = null;
    let release = () => {};
    mockApi({
      "GET /api/v1/auth/me": NOT_SIGNED_IN, // the cookie never comes back
      "POST /api/v1/auth/login": async () => {
        if (held) await held;
        return ok(ADMIN_SESSION);
      },
    });
    renderApp();
    const user = await signInWith("admin@example.com", "correct horse battery staple");
    await screen.findByText(/didn't keep the session/);

    held = new Promise<void>((resolve) => (release = resolve));
    await user.type(screen.getByLabelText("Password"), "correct horse battery staple");
    await user.click(screen.getByRole("button", { name: "Sign in" }));

    expect(await screen.findByRole("button", { name: "Signing in…" })).toBeDisabled();
    expect(screen.queryByText("Your session ended. Sign in again.")).toBeNull();
    expect(screen.queryByTestId("problem-error")).toBeNull(); // nothing to say until the answer is in

    release();
    expect(await screen.findByText(/didn't keep the session/)).toBeInTheDocument();
    expect(screen.queryByText("Your session ended. Sign in again.")).toBeNull();
  });

  it("on a secure page where the cookie still isn't kept: says to check cookies, not the address", async () => {
    pageIsSecure(true);
    serverWith({ cookieKept: false });
    renderApp();

    await signInWith("admin@example.com", "correct horse battery staple");

    const alert = await screen.findByTestId("problem-error");
    expect(alert).toHaveTextContent("allows cookies");
    expect(alert).not.toHaveTextContent("isn't a secure");
  });

  it("signs in normally on plain http when the cookie IS kept (PPB_COOKIE_SECURE=false for development)", async () => {
    pageIsSecure(false);
    serverWith({ cookieKept: true });
    renderApp();

    await signInWith("admin@example.com", "correct horse battery staple");

    expect(await screen.findByTestId("whoami")).toHaveTextContent("Test Admin · admin");
    expect(screen.queryByTestId("problem-error")).toBeNull();
  });

  it("doesn't turn a passing server hiccup right after sign-in into this message", async () => {
    pageIsSecure(false);
    serverWith({ cookieKept: true, meAfterLogin: () => new Response("Bad Gateway", { status: 502 }) });
    renderApp();

    await signInWith("admin@example.com", "correct horse battery staple");

    expect(await screen.findByTestId("whoami")).toBeInTheDocument(); // the sign-in itself was accepted
    expect(screen.queryByText(/didn't keep the session/)).toBeNull();
  });

  it("an ordinary expired session on the same plain-http page still says 'Your session ended'", async () => {
    pageIsSecure(false);
    startMockBackend(ADMIN_SESSION.user);
    mockApi({ "GET /api/v1/auth/me": () => ok(ADMIN_SESSION) }); // already signed in when the page opens
    setAdminTransport(async () => fail(401, { detail: "Sign in.", code: "not_authenticated" })); // then the session goes
    renderApp();

    expect(await screen.findByText("Your session ended. Sign in again.")).toBeInTheDocument();
    expect(screen.queryByText(/didn't keep the session/)).toBeNull();
  });

  it("a wrong password doesn't ask the server who is signed in (no extra request, no cookie message)", async () => {
    pageIsSecure(false);
    const fetchMock = mockApi({
      "GET /api/v1/auth/me": NOT_SIGNED_IN,
      "POST /api/v1/auth/login": () => fail(401, { detail: "Wrong.", code: "invalid_credentials" }),
    });
    renderApp();

    await signInWith("admin@example.com", "wrong password here");

    expect(await screen.findByText("Wrong email or password.")).toBeInTheDocument();
    expect(fetchMock.mock.calls.filter(([url]) => url === "/api/v1/auth/me")).toHaveLength(1); // only the page-load one
  });
});
