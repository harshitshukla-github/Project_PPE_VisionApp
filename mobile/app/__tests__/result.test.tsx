import { fireEvent, render, screen } from "@testing-library/react-native";

import ResultScreen from "../result";
import { SessionProvider, useSession } from "../../src/context/SessionContext";
import type { ComplianceCheckResponse } from "../../src/api/types";
import { useEffect } from "react";
import { Text } from "react-native";

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

const PASS_RESULT: ComplianceCheckResponse = {
  worker_id: "W-0042",
  job_category: "welder",
  timestamp: "2026-07-29T13:08:34Z",
  verdict: "PASS",
  confidence_avg: 0.85,
  detections: [{ class: "hardhat", confidence: 0.85, bbox: [1, 2, 3, 4] }],
  missing_equipment: [],
  violations_detected: [],
  unverifiable_equipment: ["boots"],
  explanation: null,
  model_used: "yolo26n",
  inference_time_ms: 30.1,
  gemini_invoked: false,
};

const FAIL_RESULT: ComplianceCheckResponse = {
  ...PASS_RESULT,
  verdict: "FAIL",
  missing_equipment: ["goggles"],
  violations_detected: ["no_hardhat"],
  explanation: "You're missing your goggles and hardhat.",
  gemini_invoked: true,
};

/** Seeds SessionContext.lastCheck before ResultScreen renders. */
function Seed({ result }: { result: ComplianceCheckResponse }) {
  const { recordCheck } = useSession();
  useEffect(() => {
    recordCheck(result, "file:///photo.jpg");
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);
  return null;
}

function renderResult(result?: ComplianceCheckResponse) {
  return render(
    <SessionProvider>
      {result && <Seed result={result} />}
      <ResultScreen />
    </SessionProvider>,
  );
}

beforeEach(() => {
  mockFetch.mockReset();
  mockReplace.mockReset();
  // SessionProvider fetches jobs on mount; keep it quiet/pending for these tests.
  mockFetch.mockImplementation(() => new Promise(() => {}));
});

describe("ResultScreen", () => {
  it("shows an empty state when there's no check yet", async () => {
    renderResult();

    expect(await screen.findByText("No check yet")).toBeTruthy();
  });

  it("renders a PASS result with detections and unverifiable equipment in amber", async () => {
    renderResult(PASS_RESULT);

    expect(await screen.findByText("PASS")).toBeTruthy();
    expect(screen.getByText("Hardhat — 85%")).toBeTruthy();
    expect(screen.getByText("Verify manually")).toBeTruthy();
    expect(screen.getByText("Boots")).toBeTruthy();
    expect(screen.queryByText(/missing/i)).toBeNull();
  });

  it("renders a FAIL result with explanation, missing equipment, and violations", async () => {
    renderResult(FAIL_RESULT);

    expect(await screen.findByText("FAIL")).toBeTruthy();
    expect(
      screen.getByText("You're missing your goggles and hardhat."),
    ).toBeTruthy();
    expect(screen.getByText("Missing equipment")).toBeTruthy();
    expect(screen.getByText("Goggles")).toBeTruthy();
    expect(screen.getByText("Violations detected")).toBeTruthy();
    expect(screen.getByText("No Hardhat")).toBeTruthy();
  });

  it("renders NEEDS_REVIEW as a non-accusing amber state with 'Retake photo' as the action", async () => {
    renderResult({
      ...PASS_RESULT,
      verdict: "NEEDS_REVIEW",
      detections: [],
      missing_equipment: ["safety_vest"],
      unverifiable_equipment: [],
      explanation: null,
    });

    expect(await screen.findByText("Needs review")).toBeTruthy();
    expect(
      screen.getByText("Couldn't verify: Safety Vest — ask your supervisor to confirm."),
    ).toBeTruthy();
    expect(screen.getByText("Retake photo")).toBeTruthy();
    expect(screen.queryByText("Check again")).toBeNull();
    expect(screen.queryByText("FAIL")).toBeNull();
  });

  it("keeps 'Check again' as the action for PASS and FAIL", async () => {
    renderResult(PASS_RESULT);
    expect(await screen.findByText("Check again")).toBeTruthy();
    expect(screen.queryByText("Retake photo")).toBeNull();
  });

  it("loops back to Capture from 'Retake photo'", async () => {
    renderResult({ ...PASS_RESULT, verdict: "NEEDS_REVIEW", missing_equipment: ["gloves"] });

    fireEvent.press(await screen.findByText("Retake photo"));

    expect(mockReplace).toHaveBeenCalledWith("/capture");
  });
});
