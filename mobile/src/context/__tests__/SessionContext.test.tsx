import AsyncStorage from "@react-native-async-storage/async-storage";
import { render, screen, waitFor } from "@testing-library/react-native";
import { Text } from "react-native";

import { clearToken, getToken, setToken } from "../../api/client";
import { addHistoryEntry } from "../../storage/history";
import type { ComplianceCheckResponse } from "../../api/types";
import { fail, mockRoutes, ok } from "../../testing/fixtures";
import { SessionProvider, useSession } from "../SessionContext";

jest.mock("@react-native-async-storage/async-storage", () =>
  require("@react-native-async-storage/async-storage/jest/async-storage-mock"),
);

const mockFetch = jest.fn();
// @ts-expect-error - test-only global fetch stub
global.fetch = mockFetch;

function Probe() {
  const { auth, workerId, history, historyLoaded } = useSession();
  const label =
    auth.kind === "signedIn"
      ? auth.mustChangePin
        ? "signed-in-must-change-pin"
        : "signed-in"
      : auth.kind;
  return (
    <>
      <Text testID="auth">{label}</Text>
      <Text testID="notice">{auth.kind === "signedOut" ? auth.notice?.message ?? "" : ""}</Text>
      <Text testID="worker">{workerId}</Text>
      <Text testID="history">{historyLoaded ? history.map((h) => h.id).join(",") : "…"}</Text>
    </>
  );
}

const ME = {
  worker: { worker_id: "W-0042", display_name: "Asha Verma" },
  must_change_pin: false,
  expires_at: "2026-10-16T03:12:00Z",
};

const RESULT: ComplianceCheckResponse = {
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

beforeEach(async () => {
  await AsyncStorage.clear();
  await clearToken();
  mockFetch.mockReset();
  await AsyncStorage.setItem("@ppb/server_url", "http://192.168.1.5:8000");
});

describe("SessionProvider: restoring a saved sign-in", () => {
  it("is signed out, without asking the server, when no token is saved", async () => {
    mockRoutes(mockFetch, { "/jobs": ok({}) });

    render(<SessionProvider><Probe /></SessionProvider>);

    await waitFor(() => expect(screen.getByTestId("auth")).toHaveTextContent("signedOut"));
    expect(mockFetch.mock.calls.some(([url]) => String(url).endsWith("/auth/worker/me"))).toBe(false);
  });

  it("signs in again from a saved token that the server still accepts", async () => {
    await setToken("tok-123");
    mockRoutes(mockFetch, { "/jobs": ok({}), "/auth/worker/me": ok(ME) });

    render(<SessionProvider><Probe /></SessionProvider>);

    await waitFor(() => expect(screen.getByTestId("auth")).toHaveTextContent("signed-in"));
    expect(screen.getByTestId("worker")).toHaveTextContent("W-0042");
  });

  it("reopens Change PIN for a saved token whose PIN was never changed", async () => {
    await setToken("tok-123");
    mockRoutes(mockFetch, {
      "/jobs": ok({}),
      "/auth/worker/me": ok({ ...ME, must_change_pin: true }),
    });

    render(<SessionProvider><Probe /></SessionProvider>);

    await waitFor(() =>
      expect(screen.getByTestId("auth")).toHaveTextContent("signed-in-must-change-pin"),
    );
  });

  it("drops a token the server has expired or revoked, and says the session ended", async () => {
    await setToken("tok-123");
    mockRoutes(mockFetch, {
      "/jobs": ok({}),
      "/auth/worker/me": fail(401, { detail: "expired", code: "not_authenticated" }),
    });

    render(<SessionProvider><Probe /></SessionProvider>);

    await waitFor(() => expect(screen.getByTestId("auth")).toHaveTextContent("signedOut"));
    expect(screen.getByTestId("notice")).toHaveTextContent("Your session ended. Sign in again.");
    await expect(getToken()).resolves.toBeNull();
  });

  it("keeps the token when the server is merely unreachable or busy (503)", async () => {
    await setToken("tok-123");
    mockRoutes(mockFetch, {
      "/jobs": ok({}),
      "/auth/worker/me": fail(503, { detail: "no db", code: "auth_unavailable" }),
    });

    render(<SessionProvider><Probe /></SessionProvider>);

    await waitFor(() => expect(screen.getByTestId("auth")).toHaveTextContent("signedOut"));
    expect(screen.getByTestId("notice")).toHaveTextContent(/Try again shortly/);
    await expect(getToken()).resolves.toBe("tok-123");
  });
});

describe("SessionProvider: history on a shared phone", () => {
  it("shows only the signed-in worker's checks and none when signed out", async () => {
    const mine = (await addHistoryEntry(RESULT, "file:///mine.jpg"))[0];
    await addHistoryEntry({ ...RESULT, worker_id: "W-0099" }, "file:///theirs.jpg");
    await setToken("tok-123");
    mockRoutes(mockFetch, { "/jobs": ok({}), "/auth/worker/me": ok(ME) });

    render(<SessionProvider><Probe /></SessionProvider>);

    await waitFor(() => expect(screen.getByTestId("history")).toHaveTextContent(mine.id));
    expect(screen.getByTestId("history").props.children).not.toContain(",");
  });
});
