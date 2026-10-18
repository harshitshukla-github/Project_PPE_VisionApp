import AsyncStorage from "@react-native-async-storage/async-storage";
import { fireEvent, render, screen } from "@testing-library/react-native";

import HomeScreen from "../(tabs)/index";
import { SessionProvider } from "../../src/context/SessionContext";
import { SIGNED_IN } from "../../src/testing/fixtures";

jest.mock("@react-native-async-storage/async-storage", () =>
  require("@react-native-async-storage/async-storage/jest/async-storage-mock"),
);

const mockFetch = jest.fn();
// @ts-expect-error - test-only global fetch stub
global.fetch = mockFetch;

function renderHome() {
  return render(
    <SessionProvider initialAuth={SIGNED_IN}>
      <HomeScreen />
    </SessionProvider>,
  );
}

const JOBS_RESPONSE = {
  general_labor: { required: ["hardhat", "safety_vest"], pending: ["boots"] },
  welder: {
    required: ["hardhat", "safety_vest", "goggles", "gloves"],
    pending: ["face_shield", "boots"],
  },
};

beforeEach(async () => {
  await AsyncStorage.clear();
  mockFetch.mockReset();
  await AsyncStorage.setItem("@ppb/server_url", "http://192.168.1.5:8000");
});

describe("HomeScreen", () => {
  it("loads jobs on mount and selects the first one by default", async () => {
    mockFetch.mockResolvedValueOnce({
      ok: true,
      json: async () => JOBS_RESPONSE,
    });

    renderHome();

    expect(await screen.findByText("● General Labor")).toBeTruthy();
    expect(screen.getByText("○ Welder")).toBeTruthy();
    expect(mockFetch).toHaveBeenCalledWith(
      "http://192.168.1.5:8000/api/v1/jobs",
      undefined,
    );
  });

  it("selects a different job on tap", async () => {
    mockFetch.mockResolvedValueOnce({
      ok: true,
      json: async () => JOBS_RESPONSE,
    });

    renderHome();

    await screen.findByText("● General Labor");
    fireEvent.press(screen.getByText("○ Welder"));

    expect(await screen.findByText("● Welder")).toBeTruthy();
    expect(screen.getByText("○ General Labor")).toBeTruthy();
  });

  it("shows an error with a retry option when jobs fail to load", async () => {
    mockFetch.mockRejectedValueOnce(new TypeError("Network request failed"));

    renderHome();

    expect(
      await screen.findByText(/Can't load jobs ✗ — Can't reach server/),
    ).toBeTruthy();

    mockFetch.mockResolvedValueOnce({
      ok: true,
      json: async () => JOBS_RESPONSE,
    });
    fireEvent.press(screen.getByText("Retry"));

    expect(await screen.findByText("● General Labor")).toBeTruthy();
  });

  it("shows an amber 'try again shortly' notice, not a red error, when the server answers 503", async () => {
    mockFetch.mockResolvedValueOnce({
      ok: false,
      status: 503,
      json: async () => ({
        detail: "Job requirements are temporarily unavailable. Try again shortly.",
      }),
    });

    renderHome();

    const notice = await screen.findByText(/Try again shortly/);
    expect(notice).toHaveStyle({ color: "#8a5a0b" });
    expect(screen.queryByText(/Can't load jobs ✗/)).toBeNull();
    expect(screen.getByText("Retry")).toBeTruthy();
  });

  it("shows who is signed in instead of a typed worker ID field", async () => {
    mockFetch.mockResolvedValueOnce({
      ok: true,
      json: async () => JOBS_RESPONSE,
    });

    renderHome();

    expect(await screen.findByText("Test Worker")).toBeTruthy();
    expect(screen.getByText("W-0042")).toBeTruthy();
    expect(screen.queryByPlaceholderText("W-0042")).toBeNull();
  });
});
