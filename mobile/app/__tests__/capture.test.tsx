import AsyncStorage from "@react-native-async-storage/async-storage";
import { render, screen, fireEvent, waitFor } from "@testing-library/react-native";
import { Text } from "react-native";

import CaptureScreen from "../(tabs)/capture";
import { clearToken, setToken } from "../../src/api/client";
import { SessionProvider, useSession } from "../../src/context/SessionContext";
import { SIGNED_IN } from "../../src/testing/fixtures";

jest.mock("@react-native-async-storage/async-storage", () =>
  require("@react-native-async-storage/async-storage/jest/async-storage-mock"),
);

const mockPush = jest.fn();
jest.mock("expo-router", () => ({
  useRouter: () => ({ push: mockPush, replace: jest.fn() }),
}));

const mockTakePictureAsync = jest.fn();
const mockUseCameraPermissions = jest.fn();
jest.mock("expo-camera", () => {
  const React = require("react");
  const { View: RNView } = require("react-native");
  return {
    CameraView: React.forwardRef((props: any, ref: any) => {
      React.useImperativeHandle(ref, () => ({
        takePictureAsync: mockTakePictureAsync,
      }));
      return <RNView testID="camera-view">{props.children}</RNView>;
    }),
    useCameraPermissions: () => mockUseCameraPermissions(),
  };
});

const mockLaunchImageLibraryAsync = jest.fn();
jest.mock("expo-image-picker", () => ({
  requestMediaLibraryPermissionsAsync: jest.fn(() =>
    Promise.resolve({ granted: true }),
  ),
  launchImageLibraryAsync: (...args: unknown[]) =>
    mockLaunchImageLibraryAsync(...args),
}));

const mockFetch = jest.fn();
// @ts-expect-error - test-only global fetch stub
global.fetch = mockFetch;

const JOBS_RESPONSE = {
  welder: {
    required: ["hardhat", "safety_vest", "goggles", "gloves"],
    pending: ["face_shield", "boots"],
  },
};

const CHECK_RESPONSE = {
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
};

/** Shows the session state so tests can see what Capture did to it. */
function AuthProbe() {
  const { auth } = useSession();
  const label =
    auth.kind !== "signedIn"
      ? "signed-out"
      : auth.mustChangePin
        ? "must-change-pin"
        : "signed-in";
  return <Text testID="auth-probe">{label}</Text>;
}

function renderCapture() {
  return render(
    <SessionProvider initialAuth={SIGNED_IN}>
      <AuthProbe />
      <CaptureScreen />
    </SessionProvider>,
  );
}

beforeEach(async () => {
  await clearToken();
  await setToken("test-token");
  await AsyncStorage.clear();
  await AsyncStorage.setItem("@ppb/server_url", "http://192.168.1.5:8000");
  mockFetch.mockReset();
  mockPush.mockReset();
  mockTakePictureAsync.mockReset();
  mockLaunchImageLibraryAsync.mockReset();
  mockUseCameraPermissions.mockReturnValue([{ granted: true }, jest.fn()]);
  mockFetch.mockResolvedValueOnce({
    ok: true,
    json: async () => JOBS_RESPONSE,
  });
});

describe("CaptureScreen", () => {
  it("prompts to pick a job on Home while no job category is loaded", async () => {
    mockFetch.mockReset();
    mockFetch.mockRejectedValueOnce(new TypeError("Network request failed"));
    renderCapture();

    expect(await screen.findByText(/Pick a job category on Home first/)).toBeTruthy();
  });

  it("prompts for camera permission when not yet granted", async () => {
    mockUseCameraPermissions.mockReturnValue([{ granted: false }, jest.fn()]);
    renderCapture();

    expect(await screen.findByText("Grant camera access")).toBeTruthy();
  });

  it("shows the camera preview with a framing hint once identity + permission are ready", async () => {
    renderCapture();

    expect(await screen.findByText("Fit your full body in frame")).toBeTruthy();
    expect(screen.getByTestId("camera-view")).toBeTruthy();
  });

  it("submits a captured photo and navigates to Result on success", async () => {
    renderCapture();
    await screen.findByText("Fit your full body in frame");

    mockTakePictureAsync.mockResolvedValueOnce({ uri: "file:///shot.jpg" });
    mockFetch.mockResolvedValueOnce({
      ok: true,
      json: async () => CHECK_RESPONSE,
    });

    fireEvent.press(screen.getByTestId("shutter-button"));

    await waitFor(() => expect(mockPush).toHaveBeenCalledWith("/result"));
    expect(mockFetch).toHaveBeenLastCalledWith(
      "http://192.168.1.5:8000/api/v1/compliance/check",
      expect.objectContaining({ method: "POST" }),
    );
  });

  it("identifies the worker by token only: sends Authorization and no worker_id field", async () => {
    renderCapture();
    await screen.findByText("Fit your full body in frame");

    mockTakePictureAsync.mockResolvedValueOnce({ uri: "file:///shot.jpg" });
    mockFetch.mockResolvedValueOnce({ ok: true, json: async () => CHECK_RESPONSE });

    fireEvent.press(screen.getByTestId("shutter-button"));
    await waitFor(() => expect(mockPush).toHaveBeenCalledWith("/result"));

    const [, init] = mockFetch.mock.calls[mockFetch.mock.calls.length - 1];
    expect(init.headers).toEqual({ Authorization: "Bearer test-token" });
    const body = init.body as FormData;
    expect(body.has("job_category")).toBe(true);
    expect(body.has("worker_id")).toBe(false);
  });

  it("sends the worker back to sign in when the session has ended (401 not_authenticated)", async () => {
    renderCapture();
    await screen.findByText("Fit your full body in frame");

    mockTakePictureAsync.mockResolvedValueOnce({ uri: "file:///shot.jpg" });
    mockFetch.mockResolvedValueOnce({
      ok: false,
      status: 401,
      json: async () => ({ detail: "Sign in again.", code: "not_authenticated" }),
    });

    fireEvent.press(screen.getByTestId("shutter-button"));

    await waitFor(() =>
      expect(screen.getByTestId("auth-probe")).toHaveTextContent("signed-out"),
    );
    expect(mockPush).not.toHaveBeenCalled();
  });

  it("opens Change PIN when the issued PIN must be replaced (403 pin_change_required)", async () => {
    renderCapture();
    await screen.findByText("Fit your full body in frame");

    mockTakePictureAsync.mockResolvedValueOnce({ uri: "file:///shot.jpg" });
    mockFetch.mockResolvedValueOnce({
      ok: false,
      status: 403,
      json: async () => ({
        detail: "Choose your own PIN first.",
        code: "pin_change_required",
      }),
    });

    fireEvent.press(screen.getByTestId("shutter-button"));

    await waitFor(() =>
      expect(screen.getByTestId("auth-probe")).toHaveTextContent("must-change-pin"),
    );
  });

  it("shows 'try again shortly' and no verdict when the server answers 503", async () => {
    renderCapture();
    await screen.findByText("Fit your full body in frame");

    mockTakePictureAsync.mockResolvedValueOnce({ uri: "file:///shot.jpg" });
    mockFetch.mockResolvedValueOnce({
      ok: false,
      status: 503,
      json: async () => ({
        detail: "Job requirements are temporarily unavailable. Try again shortly.",
      }),
    });

    fireEvent.press(screen.getByTestId("shutter-button"));

    expect(await screen.findByText(/Try again shortly/)).toBeTruthy();
    expect(mockPush).not.toHaveBeenCalled();
    expect(screen.getByTestId("auth-probe")).toHaveTextContent("signed-in");
  });

  it("shows an inline error banner when the check request fails", async () => {
    renderCapture();
    await screen.findByText("Fit your full body in frame");

    mockTakePictureAsync.mockResolvedValueOnce({ uri: "file:///shot.jpg" });
    mockFetch.mockRejectedValueOnce(new TypeError("Network request failed"));

    fireEvent.press(screen.getByTestId("shutter-button"));

    expect(
      await screen.findByText(/Can't reach server — check the URL in Settings/),
    ).toBeTruthy();
    expect(mockPush).not.toHaveBeenCalled();
  });

  it("picks from the gallery and submits when the camera isn't used", async () => {
    renderCapture();
    await screen.findByText("Fit your full body in frame");

    mockLaunchImageLibraryAsync.mockResolvedValueOnce({
      canceled: false,
      assets: [{ uri: "file:///gallery.jpg" }],
    });
    mockFetch.mockResolvedValueOnce({
      ok: true,
      json: async () => CHECK_RESPONSE,
    });

    fireEvent.press(screen.getByText("Gallery"));

    await waitFor(() => expect(mockPush).toHaveBeenCalledWith("/result"));
  });
});
