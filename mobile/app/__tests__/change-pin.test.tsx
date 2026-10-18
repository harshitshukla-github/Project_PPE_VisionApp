import AsyncStorage from "@react-native-async-storage/async-storage";
import { fireEvent, render, screen, waitFor } from "@testing-library/react-native";
import { Text } from "react-native";

import ChangePinScreen from "../change-pin";
import { clearToken, getToken, setToken } from "../../src/api/client";
import { SessionProvider, useSession } from "../../src/context/SessionContext";
import { SIGNED_IN, fail, mockRoutes, noContent, ok } from "../../src/testing/fixtures";

jest.mock("@react-native-async-storage/async-storage", () =>
  require("@react-native-async-storage/async-storage/jest/async-storage-mock"),
);

const mockReplace = jest.fn();
jest.mock("expo-router", () => ({
  useRouter: () => ({ replace: mockReplace, push: jest.fn() }),
}));

const mockFetch = jest.fn();
// @ts-expect-error - test-only global fetch stub
global.fetch = mockFetch;

function Probe() {
  const { auth } = useSession();
  const label =
    auth.kind !== "signedIn"
      ? auth.kind
      : auth.mustChangePin
        ? "must-change-pin"
        : "signed-in";
  return <Text testID="auth-probe">{label}</Text>;
}

function renderChangePin() {
  return render(
    <SessionProvider initialAuth={{ ...SIGNED_IN, mustChangePin: true } as never}>
      <Probe />
      <ChangePinScreen />
    </SessionProvider>,
  );
}

const [CURRENT, NEW, CONFIRM] = ["The PIN you were given", "6 digits", "6 digits"];

function fillAll(current: string, next: string, confirm: string) {
  fireEvent.changeText(screen.getByPlaceholderText(CURRENT), current);
  const sixDigit = screen.getAllByPlaceholderText(NEW);
  fireEvent.changeText(sixDigit[0], next);
  fireEvent.changeText(sixDigit[1], confirm);
}

beforeEach(async () => {
  await AsyncStorage.clear();
  await clearToken();
  await setToken("tok-123");
  mockFetch.mockReset();
  mockReplace.mockReset();
  await AsyncStorage.setItem("@ppb/server_url", "http://192.168.1.5:8000");
  mockRoutes(mockFetch, { "/jobs": ok({}) });
});

describe("ChangePinScreen", () => {
  it("keeps Save disabled until the new PIN is 6 digits and typed the same twice", async () => {
    renderChangePin();
    await screen.findByText("Choose your own PIN");

    fillAll("482913", "730154", "730155");
    expect(screen.getByRole("button", { name: "Save new PIN" })).toBeDisabled();
    expect(screen.getByText("The two PINs don't match.")).toBeTruthy();

    fillAll("482913", "730154", "730154");
    expect(screen.getByRole("button", { name: "Save new PIN" })).toBeEnabled();
  });

  it("changes the PIN with the token, then clears must-change-pin so the app opens", async () => {
    mockRoutes(mockFetch, { "/jobs": ok({}), "/auth/worker/pin": noContent });
    renderChangePin();
    await screen.findByText("Choose your own PIN");

    fillAll("482913", "730154", "730154");
    fireEvent.press(screen.getByRole("button", { name: "Save new PIN" }));

    await waitFor(() => expect(screen.getByTestId("auth-probe")).toHaveTextContent("signed-in"));
    const call = mockFetch.mock.calls.find(([url]) => String(url).endsWith("/auth/worker/pin"));
    expect(JSON.parse(call![1].body)).toEqual({ current_pin: "482913", new_pin: "730154" });
    expect(call![1].headers).toMatchObject({ Authorization: "Bearer tok-123" });
  });

  it("422 weak_pin: shows the server's reason and stays on the screen", async () => {
    mockRoutes(mockFetch, {
      "/jobs": ok({}),
      "/auth/worker/pin": fail(422, {
        detail: "That PIN is too easy to guess. Avoid runs like 123456.",
        code: "weak_pin",
      }),
    });
    renderChangePin();
    await screen.findByText("Choose your own PIN");

    fillAll("482913", "123456", "123456");
    fireEvent.press(screen.getByRole("button", { name: "Save new PIN" }));

    expect(await screen.findByText(/too easy to guess/)).toBeTruthy();
    expect(screen.getByTestId("auth-probe")).toHaveTextContent("must-change-pin");
  });

  it("a wrong current PIN says so plainly", async () => {
    mockRoutes(mockFetch, {
      "/jobs": ok({}),
      "/auth/worker/pin": fail(401, { detail: "Wrong PIN.", code: "invalid_credentials" }),
    });
    renderChangePin();
    await screen.findByText("Choose your own PIN");

    fillAll("000000", "730154", "730154");
    fireEvent.press(screen.getByRole("button", { name: "Save new PIN" }));

    expect(await screen.findByText("That isn't your current PIN.")).toBeTruthy();
  });

  it("an ended session (401 not_authenticated) sends the worker back to sign in", async () => {
    mockRoutes(mockFetch, {
      "/jobs": ok({}),
      "/auth/worker/pin": fail(401, { detail: "Sign in again.", code: "not_authenticated" }),
    });
    renderChangePin();
    await screen.findByText("Choose your own PIN");

    fillAll("482913", "730154", "730154");
    fireEvent.press(screen.getByRole("button", { name: "Save new PIN" }));

    await waitFor(() => expect(screen.getByTestId("auth-probe")).toHaveTextContent("signedOut"));
    await expect(getToken()).resolves.toBeNull();
  });

  it("503 is an amber 'try again shortly' notice", async () => {
    mockRoutes(mockFetch, {
      "/jobs": ok({}),
      "/auth/worker/pin": fail(503, { detail: "unavailable", code: "auth_unavailable" }),
    });
    renderChangePin();
    await screen.findByText("Choose your own PIN");

    fillAll("482913", "730154", "730154");
    fireEvent.press(screen.getByRole("button", { name: "Save new PIN" }));

    expect(await screen.findByTestId("problem-notice")).toBeTruthy();
    expect(screen.getByText(/Try again shortly/)).toBeTruthy();
  });

  it("lets the worker sign out instead", async () => {
    mockRoutes(mockFetch, { "/jobs": ok({}), "/auth/worker/logout": noContent });
    renderChangePin();
    await screen.findByText("Choose your own PIN");

    fireEvent.press(screen.getByText("Sign out"));

    await waitFor(() => expect(mockReplace).toHaveBeenCalledWith("/login"));
    expect(screen.getByTestId("auth-probe")).toHaveTextContent("signedOut");
    await expect(getToken()).resolves.toBeNull();
  });
});
