import AsyncStorage from "@react-native-async-storage/async-storage";
import { render, screen, fireEvent, waitFor } from "@testing-library/react-native";
import { Text } from "react-native";

import CaptureScreen from "../(tabs)/capture";
import { clearToken, setToken } from "../../src/api/client";
import { SessionProvider, useSession } from "../../src/context/SessionContext";
import { SIGNED_IN, fail, mockRoutes, ok } from "../../src/testing/fixtures";

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

const WELDER = {
  required: ["hardhat", "safety_vest", "goggles", "gloves"],
  pending: ["face_shield", "boots"],
};
const JOBS_RESPONSE = { welder: WELDER };
/** The same list after an admin deactivated Welder. */
const JOBS_WITHOUT_WELDER = {
  general_labor: { required: ["hardhat", "safety_vest"], pending: ["boots"] },
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

// What the fake server answers; a test changes these to change the server's mind.
let jobsAnswer: () => ReturnType<typeof ok>;
let checkAnswer: () => ReturnType<typeof ok>;

function serve() {
  mockRoutes(mockFetch, {
    "/jobs": () => jobsAnswer(),
    "/compliance/check": () => checkAnswer(),
  });
}

/** Every request so far, as "GET /jobs"-style labels, in order. */
function requests(): string[] {
  return mockFetch.mock.calls.map(([url, init]) => {
    const path = String(url).replace("http://192.168.1.5:8000/api/v1", "");
    return `${init?.method ?? "GET"} ${path}`;
  });
}

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

async function openCamera() {
  renderCapture();
  await screen.findByText("Fit your full body in frame");
  mockTakePictureAsync.mockResolvedValueOnce({ uri: "file:///shot.jpg" });
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
  jobsAnswer = () => ok(JOBS_RESPONSE);
  checkAnswer = () => ok(CHECK_RESPONSE);
  serve();
});

describe("CaptureScreen", () => {
  it("prompts to pick a job on Home while no job category is loaded", async () => {
    jobsAnswer = () => {
      throw new TypeError("Network request failed");
    };
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
    await openCamera();

    fireEvent.press(screen.getByTestId("shutter-button"));

    await waitFor(() => expect(mockPush).toHaveBeenCalledWith("/result"));
    expect(mockFetch).toHaveBeenLastCalledWith(
      "http://192.168.1.5:8000/api/v1/compliance/check",
      expect.objectContaining({ method: "POST" }),
    );
  });

  it("identifies the worker by token only: sends Authorization and no worker_id field", async () => {
    await openCamera();

    fireEvent.press(screen.getByTestId("shutter-button"));
    await waitFor(() => expect(mockPush).toHaveBeenCalledWith("/result"));

    const [, init] = mockFetch.mock.calls[mockFetch.mock.calls.length - 1];
    expect(init.headers).toEqual({ Authorization: "Bearer test-token" });
    const body = init.body as FormData;
    expect(body.has("job_category")).toBe(true);
    expect(body.has("worker_id")).toBe(false);
  });

  it("sends the worker back to sign in when the session has ended (401 not_authenticated)", async () => {
    checkAnswer = () => fail(401, { detail: "Sign in again.", code: "not_authenticated" });
    await openCamera();

    fireEvent.press(screen.getByTestId("shutter-button"));

    await waitFor(() =>
      expect(screen.getByTestId("auth-probe")).toHaveTextContent("signed-out"),
    );
    expect(mockPush).not.toHaveBeenCalled();
  });

  it("opens Change PIN when the issued PIN must be replaced (403 pin_change_required)", async () => {
    checkAnswer = () =>
      fail(403, { detail: "Choose your own PIN first.", code: "pin_change_required" });
    await openCamera();

    fireEvent.press(screen.getByTestId("shutter-button"));

    await waitFor(() =>
      expect(screen.getByTestId("auth-probe")).toHaveTextContent("must-change-pin"),
    );
  });

  it("shows 'try again shortly' and no verdict when the server answers 503", async () => {
    checkAnswer = () =>
      fail(503, { detail: "Job requirements are temporarily unavailable. Try again shortly." });
    await openCamera();

    fireEvent.press(screen.getByTestId("shutter-button"));

    expect(await screen.findByText(/Try again shortly/)).toBeTruthy();
    expect(mockPush).not.toHaveBeenCalled();
    expect(screen.getByTestId("auth-probe")).toHaveTextContent("signed-in");
  });

  it("shows an inline error banner when the check request fails", async () => {
    checkAnswer = () => {
      throw new TypeError("Network request failed");
    };
    await openCamera();

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

    fireEvent.press(screen.getByText("Gallery"));

    await waitFor(() => expect(mockPush).toHaveBeenCalledWith("/result"));
  });
});

describe("job list is refreshed before every check (docs/admin-api.md §2)", () => {
  it("asks for the job list again, then sends the photo", async () => {
    await openCamera();
    const before = requests().length;

    fireEvent.press(screen.getByTestId("shutter-button"));
    await waitFor(() => expect(mockPush).toHaveBeenCalledWith("/result"));

    expect(requests().slice(before)).toEqual(["GET /jobs", "POST /compliance/check"]);
  });

  it("does not send the photo when an admin deactivated the job; asks the worker to pick again", async () => {
    await openCamera();
    jobsAnswer = () => ok(JOBS_WITHOUT_WELDER);

    fireEvent.press(screen.getByTestId("shutter-button"));

    expect(await screen.findByText(/Welder isn't available any more/)).toBeTruthy();
    expect(requests()).not.toContain("POST /compliance/check");
    expect(mockPush).not.toHaveBeenCalled();
    // The camera is gone until a job is picked again: no check against some other job.
    expect(screen.queryByTestId("camera-view")).toBeNull();
  });

  it("still sends the check when the job list can't be refreshed; the check reports the problem", async () => {
    await openCamera();
    jobsAnswer = () => {
      throw new TypeError("Network request failed");
    };

    fireEvent.press(screen.getByTestId("shutter-button"));

    await waitFor(() => expect(mockPush).toHaveBeenCalledWith("/result"));
    expect(requests()).toContain("POST /compliance/check");
  });

  it("422 unknown_job after the refresh: clears the selection and says the job is gone", async () => {
    let lists = 0;
    // Initial load and the pre-check refresh still list Welder; the look after the 422 doesn't.
    jobsAnswer = () => (++lists <= 2 ? ok(JOBS_RESPONSE) : ok(JOBS_WITHOUT_WELDER));
    checkAnswer = () =>
      fail(422, { detail: "'welder' isn't an active job.", code: "unknown_job", field: "job_category" });
    await openCamera();

    fireEvent.press(screen.getByTestId("shutter-button"));

    expect(await screen.findByText(/Welder isn't available any more/)).toBeTruthy();
    expect(mockPush).not.toHaveBeenCalled();
    expect(screen.queryByText(/isn't an active job/)).toBeNull(); // not the server's raw sentence
  });

  it("422 unknown_job while the list still has the job: amber 'try again shortly', selection kept", async () => {
    checkAnswer = () =>
      fail(422, { detail: "'welder' isn't an active job.", code: "unknown_job", field: "job_category" });
    await openCamera();

    fireEvent.press(screen.getByTestId("shutter-button"));

    expect(await screen.findByText(/That job isn't available right now/)).toBeTruthy();
    expect(screen.getByTestId("camera-view")).toBeTruthy();
    expect(mockPush).not.toHaveBeenCalled();
  });
});
