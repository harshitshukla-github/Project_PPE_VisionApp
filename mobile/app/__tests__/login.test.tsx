import AsyncStorage from "@react-native-async-storage/async-storage";
import { fireEvent, render, screen, waitFor } from "@testing-library/react-native";
import { Text } from "react-native";

import LoginScreen from "../login";
import { clearToken, getToken } from "../../src/api/client";
import { SessionProvider, useSession, type AuthState } from "../../src/context/SessionContext";
import { SIGNED_OUT, fail, mockRoutes, ok } from "../../src/testing/fixtures";

jest.mock("@react-native-async-storage/async-storage", () =>
  require("@react-native-async-storage/async-storage/jest/async-storage-mock"),
);

jest.mock("expo-router", () => {
  const { Text: RNText } = require("react-native");
  return { Link: ({ children }: { children: string }) => <RNText>{children}</RNText> };
});

const mockFetch = jest.fn();
// @ts-expect-error - test-only global fetch stub
global.fetch = mockFetch;

const LOGIN_OK = {
  token: "tok-123",
  token_type: "bearer",
  expires_at: "2026-10-16T03:12:00Z",
  must_change_pin: false,
  worker: { worker_id: "W-0042", display_name: "Asha Verma" },
};

function Probe() {
  const { auth } = useSession();
  const label =
    auth.kind !== "signedIn"
      ? auth.kind
      : auth.mustChangePin
        ? "signed-in-must-change-pin"
        : "signed-in";
  return <Text testID="auth-probe">{label}</Text>;
}

function renderLogin(initialAuth: AuthState = SIGNED_OUT) {
  return render(
    <SessionProvider initialAuth={initialAuth}>
      <Probe />
      <LoginScreen />
    </SessionProvider>,
  );
}

function fill(workerId: string, pin: string) {
  fireEvent.changeText(screen.getByPlaceholderText("W-0042"), workerId);
  fireEvent.changeText(screen.getByPlaceholderText("6 digits"), pin);
}

beforeEach(async () => {
  await AsyncStorage.clear();
  await clearToken();
  mockFetch.mockReset();
  await AsyncStorage.setItem("@ppb/server_url", "http://192.168.1.5:8000");
  mockRoutes(mockFetch, { "/jobs": ok({}) });
});

describe("LoginScreen", () => {
  it("keeps Sign in disabled until there is a worker ID and a 6-digit PIN", async () => {
    renderLogin();
    const button = await screen.findByRole("button", { name: "Sign in" });
    expect(button).toBeDisabled();

    fill("W-0042", "12345");
    expect(screen.getByRole("button", { name: "Sign in" })).toBeDisabled();

    fill("W-0042", "123456");
    expect(screen.getByRole("button", { name: "Sign in" })).toBeEnabled();
  });

  it("only accepts digits in the PIN field", async () => {
    renderLogin();
    await screen.findByPlaceholderText("W-0042");

    fill("W-0042", "12a4-56");

    expect(screen.getByPlaceholderText("6 digits").props.value).toBe("12456");
  });

  it("signs in: stores the token and moves the session to signed-in", async () => {
    mockRoutes(mockFetch, { "/jobs": ok({}), "/auth/worker/login": ok(LOGIN_OK) });
    renderLogin();
    await screen.findByPlaceholderText("W-0042");

    fill("W-0042", "482913");
    fireEvent.press(screen.getByRole("button", { name: "Sign in" }));

    await waitFor(() => expect(screen.getByTestId("auth-probe")).toHaveTextContent("signed-in"));
    await expect(getToken()).resolves.toBe("tok-123");
    await expect(AsyncStorage.getItem("@ppb/worker_id")).resolves.toBe("W-0042");
  });

  it("goes to the Change PIN step when login says must_change_pin", async () => {
    mockRoutes(mockFetch, {
      "/jobs": ok({}),
      "/auth/worker/login": ok({ ...LOGIN_OK, must_change_pin: true }),
    });
    renderLogin();
    await screen.findByPlaceholderText("W-0042");

    fill("W-0042", "482913");
    fireEvent.press(screen.getByRole("button", { name: "Sign in" }));

    await waitFor(() =>
      expect(screen.getByTestId("auth-probe")).toHaveTextContent("signed-in-must-change-pin"),
    );
  });

  it("401 invalid_credentials: says so in red and clears the PIN", async () => {
    mockRoutes(mockFetch, {
      "/jobs": ok({}),
      "/auth/worker/login": fail(401, { detail: "Wrong worker ID or PIN.", code: "invalid_credentials" }),
    });
    renderLogin();
    await screen.findByPlaceholderText("W-0042");

    fill("W-0042", "000000");
    fireEvent.press(screen.getByRole("button", { name: "Sign in" }));

    expect(await screen.findByText("Wrong worker ID or PIN.")).toBeTruthy();
    expect(screen.getByTestId("problem-error")).toBeTruthy();
    expect(screen.getByPlaceholderText("6 digits").props.value).toBe("");
    expect(screen.getByTestId("auth-probe")).toHaveTextContent("signedOut");
    await expect(getToken()).resolves.toBeNull();
  });

  it("423 account_locked: shows when to retry", async () => {
    mockRoutes(mockFetch, {
      "/jobs": ok({}),
      "/auth/worker/login": fail(423, {
        detail: "Too many wrong PINs. Try again in 14 minutes.",
        code: "account_locked",
        retry_after_s: 840,
      }),
    });
    renderLogin();
    await screen.findByPlaceholderText("W-0042");

    fill("W-0042", "000000");
    fireEvent.press(screen.getByRole("button", { name: "Sign in" }));

    expect(await screen.findByText(/Try again in 14 minutes/)).toBeTruthy();
  });

  it("429 too_many_attempts: tells the worker to wait", async () => {
    mockRoutes(mockFetch, {
      "/jobs": ok({}),
      "/auth/worker/login": fail(429, {
        detail: "Slow down.",
        code: "too_many_attempts",
        retry_after_s: 30,
      }),
    });
    renderLogin();
    await screen.findByPlaceholderText("W-0042");

    fill("W-0042", "000000");
    fireEvent.press(screen.getByRole("button", { name: "Sign in" }));

    expect(await screen.findByText("Too many attempts. Wait 30 seconds and try again.")).toBeTruthy();
  });

  it("503: amber 'try again shortly', not a red error", async () => {
    mockRoutes(mockFetch, {
      "/jobs": ok({}),
      "/auth/worker/login": fail(503, { detail: "Login is unavailable.", code: "auth_unavailable" }),
    });
    renderLogin();
    await screen.findByPlaceholderText("W-0042");

    fill("W-0042", "482913");
    fireEvent.press(screen.getByRole("button", { name: "Sign in" }));

    expect(await screen.findByText(/Try again shortly/)).toBeTruthy();
    expect(screen.getByTestId("problem-notice")).toBeTruthy();
    expect(screen.queryByTestId("problem-error")).toBeNull();
  });

  it("pre-fills the last worker ID that signed in on this phone", async () => {
    await AsyncStorage.setItem("@ppb/worker_id", "W-0042");

    renderLogin();

    expect(await screen.findByDisplayValue("W-0042")).toBeTruthy();
  });

  it("explains why the worker is back here after a session ended", async () => {
    renderLogin({
      kind: "signedOut",
      notice: { message: "Your session ended. Sign in again.", tone: "error" },
    });

    expect(await screen.findByText("Your session ended. Sign in again.")).toBeTruthy();
  });

  it("shows only a spinner while the saved sign-in is being checked", async () => {
    renderLogin({ kind: "loading" });

    expect(screen.UNSAFE_queryByType(require("react-native").ActivityIndicator)).toBeTruthy();
    expect(screen.queryByPlaceholderText("W-0042")).toBeNull();
  });

  it("keeps Settings reachable, since the server URL must be set before the first login", async () => {
    renderLogin();

    expect(await screen.findByText("Settings")).toBeTruthy();
  });
});
