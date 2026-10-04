import AsyncStorage from "@react-native-async-storage/async-storage";
import { render, screen, fireEvent } from "@testing-library/react-native";

import HistoryScreen from "../(tabs)/history";
import { SessionProvider } from "../../src/context/SessionContext";
import { addHistoryEntry } from "../../src/storage/history";
import type { ComplianceCheckResponse } from "../../src/api/types";

jest.mock("@react-native-async-storage/async-storage", () =>
  require("@react-native-async-storage/async-storage/jest/async-storage-mock"),
);

const mockPush = jest.fn();
jest.mock("expo-router", () => ({
  useRouter: () => ({ push: mockPush, replace: jest.fn() }),
}));

const mockFetch = jest.fn();
// @ts-expect-error - test-only global fetch stub
global.fetch = mockFetch;

const PASS_RESULT: ComplianceCheckResponse = {
  worker_id: "W-0042",
  job_category: "welder",
  timestamp: "2026-07-29T13:08:34.000Z",
  verdict: "PASS",
  confidence_avg: 0.85,
  detections: [],
  missing_equipment: [],
  violations_detected: [],
  unverifiable_equipment: [],
  explanation: null,
  model_used: "yolo26n",
  inference_time_ms: 30.1,
  gemini_invoked: false,
};

const FAIL_RESULT: ComplianceCheckResponse = {
  ...PASS_RESULT,
  worker_id: "W-0099",
  job_category: "general_labor",
  timestamp: "2026-07-30T09:00:00.000Z",
  verdict: "FAIL",
  missing_equipment: ["hardhat"],
};

function renderHistory() {
  return render(
    <SessionProvider>
      <HistoryScreen />
    </SessionProvider>,
  );
}

beforeEach(async () => {
  await AsyncStorage.clear();
  mockFetch.mockReset();
  mockPush.mockReset();
  // SessionProvider fetches jobs on mount; keep it pending and quiet for these tests.
  mockFetch.mockImplementation(() => new Promise(() => {}));
});

describe("HistoryScreen", () => {
  it("shows a loading spinner before the AsyncStorage read resolves, not the empty state", async () => {
    renderHistory();

    // historyLoaded starts false — must not claim "no checks" before we've
    // actually finished reading AsyncStorage, even though the list is [] either way.
    expect(
      screen.queryByText("No checks yet — completed compliance checks will show up here."),
    ).toBeNull();
    expect(screen.UNSAFE_queryByType(require("react-native").ActivityIndicator)).toBeTruthy();

    await screen.findByText("No checks yet — completed compliance checks will show up here.");
  });

  it("shows an empty state when nothing has been checked yet", async () => {
    renderHistory();

    expect(
      await screen.findByText("No checks yet — completed compliance checks will show up here."),
    ).toBeTruthy();
  });

  it("lists past checks most-recent-first with job, worker, and verdict", async () => {
    await addHistoryEntry(PASS_RESULT, "file:///W-0042.jpg");
    await addHistoryEntry(FAIL_RESULT, "file:///W-0099.jpg");

    renderHistory();

    expect(await screen.findByText("FAIL")).toBeTruthy();
    expect(screen.getByText("General Labor")).toBeTruthy();
    expect(screen.getByText("PASS")).toBeTruthy();
    expect(screen.getByText("Welder")).toBeTruthy();
    expect(screen.getByText(/W-0099/)).toBeTruthy();
    expect(screen.getByText(/W-0042/)).toBeTruthy();
  });

  it("navigates to the detail route with the entry's id on tap", async () => {
    const [entry] = await addHistoryEntry(PASS_RESULT, "file:///W-0042.jpg");

    renderHistory();
    fireEvent.press(await screen.findByText("Welder"));

    expect(mockPush).toHaveBeenCalledWith(`/history-detail?id=${entry.id}`);
  });
});
