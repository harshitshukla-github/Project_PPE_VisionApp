import AsyncStorage from "@react-native-async-storage/async-storage";
import { fireEvent, render, screen, waitFor } from "@testing-library/react-native";

import SettingsScreen from "../settings";

jest.mock("@react-native-async-storage/async-storage", () =>
  require("@react-native-async-storage/async-storage/jest/async-storage-mock"),
);

const mockReplace = jest.fn();
jest.mock("expo-router", () => ({
  useRouter: () => ({ replace: mockReplace, push: jest.fn() }),
}));

// Settings only needs the sign-in state and signOut; no provider (and no jobs fetch) required.
const mockSignOut = jest.fn();
let mockAuth: { kind: string; worker?: { worker_id: string; display_name: string } } = {
  kind: "signedOut",
};
jest.mock("../../src/context/SessionContext", () => ({
  useSession: () => ({ auth: mockAuth, signOut: mockSignOut }),
}));

const mockFetch = jest.fn();
// @ts-expect-error - test-only global fetch stub
global.fetch = mockFetch;

beforeEach(async () => {
  await AsyncStorage.clear();
  mockFetch.mockReset();
  mockReplace.mockReset();
  mockSignOut.mockReset();
  mockSignOut.mockResolvedValue(undefined);
  mockAuth = { kind: "signedOut" };
});

describe("SettingsScreen account section", () => {
  it("hides Log out while nobody is signed in (Settings is reachable from Login)", async () => {
    render(<SettingsScreen />);

    await screen.findByText("Not tested yet.");
    expect(screen.queryByTestId("logout-button")).toBeNull();
  });

  it("shows who is signed in and logs out, then lands on Login", async () => {
    mockAuth = {
      kind: "signedIn",
      worker: { worker_id: "W-0042", display_name: "Test Worker" },
    };
    render(<SettingsScreen />);

    expect(await screen.findByText("Signed in as Test Worker (W-0042)")).toBeTruthy();
    fireEvent.press(screen.getByTestId("logout-button"));

    await waitFor(() => expect(mockReplace).toHaveBeenCalledWith("/login"));
    expect(mockSignOut).toHaveBeenCalledTimes(1);
  });
});

describe("SettingsScreen", () => {
  it("shows 'Not tested yet' with no saved URL", async () => {
    render(<SettingsScreen />);

    expect(await screen.findByText("Not tested yet.")).toBeTruthy();
    expect(mockFetch).not.toHaveBeenCalled();
  });

  it("auto-checks a previously saved URL on mount", async () => {
    await AsyncStorage.setItem("@ppb/server_url", "http://192.168.1.5:8000");
    mockFetch.mockResolvedValueOnce({
      ok: true,
      json: async () => ({
        status: "ok",
        model_loaded: true,
        model_name: "yolo26n",
        device: "cpu",
      }),
    });

    render(<SettingsScreen />);

    expect(
      await screen.findByText("Connected ✓ — yolo26n on cpu"),
    ).toBeTruthy();
    expect(mockFetch).toHaveBeenCalledWith(
      "http://192.168.1.5:8000/api/v1/health",
      undefined,
    );
  });

  it("saves the entered URL, trims it, and shows Connected on success", async () => {
    mockFetch.mockResolvedValueOnce({
      ok: true,
      json: async () => ({
        status: "ok",
        model_loaded: true,
        model_name: "yolo26n",
        device: "cpu",
      }),
    });

    render(<SettingsScreen />);

    const input = screen.getByPlaceholderText("http://192.168.1.5:8000");
    fireEvent.changeText(input, "  http://192.168.68.116:8000/  ");
    fireEvent.press(screen.getByText("Save & Test Connection"));

    expect(await screen.findByText("Connected ✓ — yolo26n on cpu")).toBeTruthy();
    await expect(AsyncStorage.getItem("@ppb/server_url")).resolves.toBe(
      "http://192.168.68.116:8000",
    );
  });

  it("shows a 'not connected' error when the health check fails", async () => {
    mockFetch.mockRejectedValueOnce(new TypeError("Network request failed"));

    render(<SettingsScreen />);

    fireEvent.changeText(
      screen.getByPlaceholderText("http://192.168.1.5:8000"),
      "http://192.168.68.116:8000",
    );
    fireEvent.press(screen.getByText("Save & Test Connection"));

    expect(
      await screen.findByText(/Not connected ✗ — Can't reach server/),
    ).toBeTruthy();
  });

  it("shows an error and does not save when the input is empty", async () => {
    render(<SettingsScreen />);

    await waitFor(() => screen.getByText("Not tested yet."));
    fireEvent.press(screen.getByText("Save & Test Connection"));

    expect(
      await screen.findByText("Not connected ✗ — Enter a server URL first."),
    ).toBeTruthy();
    await expect(AsyncStorage.getItem("@ppb/server_url")).resolves.toBeNull();
    expect(mockFetch).not.toHaveBeenCalled();
  });

  it("flags a model-not-loaded health response distinctly", async () => {
    mockFetch.mockResolvedValueOnce({
      ok: true,
      json: async () => ({
        status: "degraded",
        model_loaded: false,
        model_name: "yolo26n",
        device: "cpu",
      }),
    });

    render(<SettingsScreen />);
    fireEvent.changeText(
      screen.getByPlaceholderText("http://192.168.1.5:8000"),
      "http://192.168.68.116:8000",
    );
    fireEvent.press(screen.getByText("Save & Test Connection"));

    expect(
      await screen.findByText(
        "Connected ✓ — yolo26n on cpu (model not loaded)",
      ),
    ).toBeTruthy();
  });
});
