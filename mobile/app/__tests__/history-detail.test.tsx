import AsyncStorage from "@react-native-async-storage/async-storage";
import { render, screen } from "@testing-library/react-native";

import HistoryDetailScreen from "../history-detail";
import { SessionProvider } from "../../src/context/SessionContext";
import { addHistoryEntry } from "../../src/storage/history";
import type { ComplianceCheckResponse } from "../../src/api/types";

jest.mock("@react-native-async-storage/async-storage", () =>
  require("@react-native-async-storage/async-storage/jest/async-storage-mock"),
);

let mockParams: { id?: string } = {};
jest.mock("expo-router", () => ({
  useLocalSearchParams: () => mockParams,
}));

const mockFetch = jest.fn();
// @ts-expect-error - test-only global fetch stub
global.fetch = mockFetch;

const FAIL_RESULT: ComplianceCheckResponse = {
  worker_id: "W-0042",
  job_category: "welder",
  timestamp: "2026-07-29T13:08:34.000Z",
  verdict: "FAIL",
  confidence_avg: 0.4,
  detections: [],
  missing_equipment: ["goggles"],
  violations_detected: ["no_hardhat"],
  unverifiable_equipment: ["boots"],
  explanation: "You're missing your goggles and hardhat.",
  model_used: "yolo26n",
  inference_time_ms: 30.1,
  gemini_invoked: true,
};

function renderDetail() {
  return render(
    <SessionProvider>
      <HistoryDetailScreen />
    </SessionProvider>,
  );
}

beforeEach(async () => {
  await AsyncStorage.clear();
  mockFetch.mockReset();
  mockFetch.mockImplementation(() => new Promise(() => {}));
  mockParams = {};
});

describe("HistoryDetailScreen", () => {
  it("shows a not-found state for an unknown id", async () => {
    mockParams = { id: "does-not-exist" };

    renderDetail();

    expect(await screen.findByText("Not found")).toBeTruthy();
  });

  it("renders the matching entry's full check result", async () => {
    const [entry] = await addHistoryEntry(FAIL_RESULT, "file:///photo.jpg");
    mockParams = { id: entry.id };

    renderDetail();

    expect(await screen.findByText("FAIL")).toBeTruthy();
    expect(screen.getByText("You're missing your goggles and hardhat.")).toBeTruthy();
    expect(screen.getByText("Goggles")).toBeTruthy();
    expect(screen.getByText("No Hardhat")).toBeTruthy();
    expect(screen.getByText("Boots")).toBeTruthy();
  });
});
