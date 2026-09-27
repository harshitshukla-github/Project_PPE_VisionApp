import AsyncStorage from "@react-native-async-storage/async-storage";
import { fireEvent, render, screen, waitFor } from "@testing-library/react-native";

import HomeScreen from "../(tabs)/index";
import { SessionProvider } from "../../src/context/SessionContext";

jest.mock("@react-native-async-storage/async-storage", () =>
  require("@react-native-async-storage/async-storage/jest/async-storage-mock"),
);

const mockFetch = jest.fn();
// @ts-expect-error - test-only global fetch stub
global.fetch = mockFetch;

function renderHome() {
  return render(
    <SessionProvider>
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

  it("pre-fills a previously saved worker ID", async () => {
    await AsyncStorage.setItem("@ppb/worker_id", "W-0042");
    mockFetch.mockResolvedValueOnce({
      ok: true,
      json: async () => JOBS_RESPONSE,
    });

    renderHome();

    expect(await screen.findByDisplayValue("W-0042")).toBeTruthy();
  });

  it("persists the worker ID as it's typed", async () => {
    mockFetch.mockResolvedValueOnce({
      ok: true,
      json: async () => JOBS_RESPONSE,
    });

    renderHome();
    await waitFor(() => screen.getByText("● General Labor"));

    fireEvent.changeText(screen.getByPlaceholderText("W-0042"), "W-0099");

    await expect(AsyncStorage.getItem("@ppb/worker_id")).resolves.toBe(
      "W-0099",
    );
  });
});
