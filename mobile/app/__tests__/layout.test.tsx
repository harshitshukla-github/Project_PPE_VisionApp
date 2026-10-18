import AsyncStorage from "@react-native-async-storage/async-storage";
import { act, fireEvent, renderRouter, screen, waitFor } from "expo-router/testing-library";
import { Text } from "react-native";

import ChangePinScreen from "../change-pin";
import RootLayout from "../_layout";
import LoginScreen from "../login";
import { clearToken, setToken } from "../../src/api/client";
import { fail, mockRoutes, ok } from "../../src/testing/fixtures";

jest.mock("@react-native-async-storage/async-storage", () =>
  require("@react-native-async-storage/async-storage/jest/async-storage-mock"),
);

const mockFetch = jest.fn();
// @ts-expect-error - test-only global fetch stub
global.fetch = mockFetch;

const ME = {
  worker: { worker_id: "W-0042", display_name: "Asha Verma" },
  must_change_pin: false,
  expires_at: "2026-10-16T03:12:00Z",
};

/**
 * The real root layout and the real Login / Change PIN screens, with plain
 * stand-ins for the camera-heavy tabs: this tests the routing gate only.
 */
function renderApp() {
  return renderRouter(
    {
      _layout: RootLayout,
      login: LoginScreen,
      "change-pin": ChangePinScreen,
      "(tabs)/_layout": () => {
        const { Slot } = require("expo-router");
        return <Slot />;
      },
      "(tabs)/index": () => <Text>HOME-STUB</Text>,
      result: () => <Text>RESULT-STUB</Text>,
      "history-detail": () => <Text>DETAIL-STUB</Text>,
      settings: () => <Text>SETTINGS-STUB</Text>,
    },
    { initialUrl: "/" },
  );
}

beforeEach(async () => {
  await AsyncStorage.clear();
  await clearToken();
  mockFetch.mockReset();
  await AsyncStorage.setItem("@ppb/server_url", "http://192.168.1.5:8000");
});

describe("root layout sign-in gate (docs/auth-api.md §8)", () => {
  it("opens Login, not the app, when nobody is signed in", async () => {
    mockRoutes(mockFetch, { "/jobs": ok({}) });

    renderApp();

    expect(await screen.findByPlaceholderText("W-0042")).toBeTruthy();
    expect(screen.queryByText("HOME-STUB")).toBeNull();
  });

  it("opens the app, not Login, for a saved token the server still accepts", async () => {
    await setToken("tok-123");
    mockRoutes(mockFetch, { "/jobs": ok({}), "/auth/worker/me": ok(ME) });

    renderApp();

    expect(await screen.findByText("HOME-STUB")).toBeTruthy();
    expect(screen.queryByPlaceholderText("W-0042")).toBeNull();
  });

  it("opens Change PIN first when the saved PIN is still the issued one", async () => {
    await setToken("tok-123");
    mockRoutes(mockFetch, {
      "/jobs": ok({}),
      "/auth/worker/me": ok({ ...ME, must_change_pin: true }),
    });

    renderApp();

    expect(await screen.findByText("Choose your own PIN")).toBeTruthy();
    expect(screen.queryByText("HOME-STUB")).toBeNull();
  });

  it("falls back to Login, saying the session ended, when the saved token has expired", async () => {
    await setToken("tok-123");
    mockRoutes(mockFetch, {
      "/jobs": ok({}),
      "/auth/worker/me": fail(401, { detail: "expired", code: "not_authenticated" }),
    });

    renderApp();

    expect(await screen.findByText("Your session ended. Sign in again.")).toBeTruthy();
    expect(screen.queryByText("HOME-STUB")).toBeNull();
  });

  it("moves from Login into the app as soon as sign-in succeeds", async () => {
    mockRoutes(mockFetch, {
      "/jobs": ok({}),
      "/auth/worker/login": ok({
        token: "tok-123",
        token_type: "bearer",
        expires_at: "2026-10-16T03:12:00Z",
        must_change_pin: false,
        worker: ME.worker,
      }),
    });
    renderApp();
    const idField = await screen.findByPlaceholderText("W-0042");

    fireEvent.changeText(idField, "W-0042");
    fireEvent.changeText(screen.getByPlaceholderText("6 digits"), "482913");
    await act(async () => {
      fireEvent.press(screen.getByRole("button", { name: "Sign in" }));
    });

    await waitFor(() => expect(screen.getByText("HOME-STUB")).toBeTruthy());
    expect(screen.queryByPlaceholderText("W-0042")).toBeNull();
  });
});
