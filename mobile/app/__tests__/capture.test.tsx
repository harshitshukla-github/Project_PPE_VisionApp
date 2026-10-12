import AsyncStorage from "@react-native-async-storage/async-storage";
import { render, screen, fireEvent, waitFor } from "@testing-library/react-native";
import { useEffect } from "react";

import CaptureScreen from "../(tabs)/capture";
import { SessionProvider, useSession } from "../../src/context/SessionContext";

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

/** Seeds SessionContext.workerId before CaptureScreen renders. */
function Seed({ workerId }: { workerId: string }) {
  const { setWorkerId } = useSession();
  useEffect(() => {
    setWorkerId(workerId);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);
  return null;
}

function renderCapture({ withIdentity = true } = {}) {
  return render(
    <SessionProvider>
      {withIdentity && <Seed workerId="W-0042" />}
      <CaptureScreen />
    </SessionProvider>,
  );
}

beforeEach(async () => {
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
  it("prompts to set up identity on Home when worker ID / job aren't set", async () => {
    renderCapture({ withIdentity: false });

    expect(
      await screen.findByText(/Enter your Worker ID and pick a job category/),
    ).toBeTruthy();
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
