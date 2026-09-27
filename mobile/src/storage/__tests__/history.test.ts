import AsyncStorage from "@react-native-async-storage/async-storage";

import { addHistoryEntry, getHistory } from "../history";
import type { ComplianceCheckResponse } from "../../api/types";

jest.mock("@react-native-async-storage/async-storage", () =>
  require("@react-native-async-storage/async-storage/jest/async-storage-mock"),
);

beforeEach(async () => {
  await AsyncStorage.clear();
});

function makeResult(overrides: Partial<ComplianceCheckResponse> = {}): ComplianceCheckResponse {
  return {
    worker_id: "W-0042",
    job_category: "welder",
    timestamp: "2026-07-29T13:08:34Z",
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
    ...overrides,
  };
}

describe("history storage", () => {
  it("returns an empty list when nothing has been recorded yet", async () => {
    await expect(getHistory()).resolves.toEqual([]);
  });

  it("round-trips a recorded entry", async () => {
    const result = makeResult();
    await addHistoryEntry(result, "file:///photo.jpg");

    const history = await getHistory();
    expect(history).toHaveLength(1);
    expect(history[0]).toMatchObject({ result, imageUri: "file:///photo.jpg" });
    expect(history[0].id).toEqual(expect.any(String));
  });

  it("orders entries most-recent-first", async () => {
    await addHistoryEntry(makeResult({ worker_id: "W-0001" }), "file:///1.jpg");
    await addHistoryEntry(makeResult({ worker_id: "W-0002" }), "file:///2.jpg");

    const history = await getHistory();
    expect(history.map((e) => e.result.worker_id)).toEqual(["W-0002", "W-0001"]);
  });

  it("gives each entry a unique id even for the same timestamp", async () => {
    const result = makeResult();
    await addHistoryEntry(result, "file:///1.jpg");
    await addHistoryEntry(result, "file:///2.jpg");

    const [first, second] = await getHistory();
    expect(first.id).not.toBe(second.id);
  });

  it("caps the list at 100 entries, dropping the oldest", async () => {
    for (let i = 0; i < 101; i++) {
      await addHistoryEntry(makeResult({ worker_id: `W-${i}` }), `file:///${i}.jpg`);
    }

    const history = await getHistory();
    expect(history).toHaveLength(100);
    expect(history[0].result.worker_id).toBe("W-100");
    expect(history[history.length - 1].result.worker_id).toBe("W-1");
  });
});
