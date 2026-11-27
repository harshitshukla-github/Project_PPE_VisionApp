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

// Settings needs the sign-in state, signOut and the photo consent; no provider (and no jobs fetch) required.
const mockSignOut = jest.fn();
let mockAuth: { kind: string; mustChangePin?: boolean; worker?: { worker_id: string; display_name: string } } = {
  kind: "signedOut",
};
let mockConsent: unknown = null;
const mockRefreshConsent = jest.fn();
const mockAnswerNotice = jest.fn();
jest.mock("../../src/context/SessionContext", () => ({
  useSession: () => ({
    auth: mockAuth,
    signOut: mockSignOut,
    consent: mockConsent,
    refreshConsent: mockRefreshConsent,
    answerNotice: mockAnswerNotice,
  }),
  noticeFrom: (err: { code?: string; body?: { notice?: unknown } }) =>
    err?.code === "notice_changed" ? (err.body?.notice ?? null) : null,
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
  mockConsent = null;
  mockRefreshConsent.mockReset();
  mockRefreshConsent.mockResolvedValue(null);
  mockAnswerNotice.mockReset();
});

describe("SettingsScreen: your photos (docs/checks-api.md §5)", () => {
  const NOTICE_1 = {
    version: 1,
    text: "Photos from your PPE checks are kept on the company's server for up to 90 days.",
    training_question: "Can we also keep your photos for up to 730 days to improve the PPE detector?",
  };
  const NOTICE_2 = { ...NOTICE_1, version: 2, text: "Photos are now kept for up to 60 days." };
  const state = (notice: typeof NOTICE_1, seen: number | null, training: boolean) => ({
    notice,
    seen_version: seen,
    training,
    updated_at: "2026-11-20T06:10:00Z",
  });

  beforeEach(() => {
    mockAuth = { kind: "signedIn", mustChangePin: false, worker: { worker_id: "W-0042", display_name: "Asha Verma" } };
  });

  it("shows the worker's training choice, the notice to read again, and who to ask about deleting photos", async () => {
    mockConsent = state(NOTICE_1, 1, true);
    render(<SettingsScreen />);

    expect(await screen.findByTestId("photo-settings")).toBeTruthy();
    expect(screen.getByTestId("settings-training-switch").props.value).toBe(true);
    expect(screen.getByText("To have your photos deleted, ask your supervisor.")).toBeTruthy();
    expect(screen.queryByText(NOTICE_1.text)).toBeNull();

    fireEvent.press(screen.getByText("Read the photo notice"));
    expect(screen.getByText(NOTICE_1.text)).toBeTruthy();
  });

  it("isn't shown to someone signed out, or still on the issued PIN (the server refuses consent then)", async () => {
    mockAuth = { kind: "signedOut" };
    const { unmount } = render(<SettingsScreen />);
    expect(screen.queryByTestId("photo-settings")).toBeNull();
    unmount();

    mockAuth = { kind: "signedIn", mustChangePin: true, worker: { worker_id: "W-0042", display_name: "Asha Verma" } };
    render(<SettingsScreen />);
    expect(screen.queryByTestId("photo-settings")).toBeNull();
    expect(screen.getByText(/Signed in as Asha Verma/)).toBeTruthy();
  });

  it("asks the server first, then saves the choice against the current version", async () => {
    mockConsent = state(NOTICE_1, 1, false);
    mockRefreshConsent.mockResolvedValue(state(NOTICE_1, 1, false));
    mockAnswerNotice.mockResolvedValue(state(NOTICE_1, 1, true));
    render(<SettingsScreen />);

    fireEvent(await screen.findByTestId("settings-training-switch"), "valueChange", true);

    await waitFor(() => expect(mockAnswerNotice).toHaveBeenCalledWith(1, true));
    expect(mockRefreshConsent).toHaveBeenCalled();
    expect(await screen.findByText("Saved. Your photos may be kept to improve the PPE detector.")).toBeTruthy();
  });

  it("shows a newer notice first, on 'No' though the worker just turned training on, and saves 'No' if left so", async () => {
    // Not a pre-ticked box (§5, "off until they say yes"): the newer notice may ask something
    // different, so the "Yes" given to the old wording doesn't carry over.
    mockConsent = state(NOTICE_1, 1, false);
    mockRefreshConsent.mockResolvedValue(state(NOTICE_2, 1, false));
    mockAnswerNotice.mockResolvedValue(state(NOTICE_2, 2, false));
    render(<SettingsScreen />);

    fireEvent(await screen.findByTestId("settings-training-switch"), "valueChange", true);

    expect(await screen.findByText(NOTICE_2.text)).toBeTruthy();
    expect(mockAnswerNotice).not.toHaveBeenCalled(); // nothing saved before it is read
    expect(screen.getByTestId("training-switch").props.value).toBe(false);

    fireEvent.press(screen.getByRole("button", { name: "Save my answer" }));

    await waitFor(() => expect(mockAnswerNotice).toHaveBeenCalledWith(2, false));
    expect(mockAnswerNotice).toHaveBeenCalledTimes(1);
    expect(await screen.findByText("Saved. Your photos won't be kept for training.")).toBeTruthy();
  });

  it("keeps a 'Yes' given on the newer notice itself, and saves it to that version", async () => {
    mockConsent = state(NOTICE_1, 1, false);
    mockRefreshConsent.mockResolvedValue(state(NOTICE_2, 1, false));
    mockAnswerNotice.mockResolvedValue(state(NOTICE_2, 2, true));
    render(<SettingsScreen />);

    fireEvent(await screen.findByTestId("settings-training-switch"), "valueChange", true);
    fireEvent(await screen.findByTestId("training-switch"), "valueChange", true);
    fireEvent.press(screen.getByRole("button", { name: "Save my answer" }));

    await waitFor(() => expect(mockAnswerNotice).toHaveBeenCalledWith(2, true));
    expect(await screen.findByText("Saved. Your photos may be kept to improve the PPE detector.")).toBeTruthy();
  });

  it("says so when the server can't be asked, and saves nothing", async () => {
    mockConsent = state(NOTICE_1, 1, false);
    mockRefreshConsent.mockResolvedValue(null);
    render(<SettingsScreen />);

    fireEvent(await screen.findByTestId("settings-training-switch"), "valueChange", true);

    expect(await screen.findByText("Couldn't reach the server to change this. Try again.")).toBeTruthy();
    expect(mockAnswerNotice).not.toHaveBeenCalled();
  });
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
