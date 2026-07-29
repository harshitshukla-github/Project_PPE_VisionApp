import AsyncStorage from "@react-native-async-storage/async-storage";

import {
  getHealth,
  getJobs,
  getServerUrl,
  postComplianceCheck,
  setServerUrl,
} from "../client";

// The community AsyncStorage package ships an official in-memory jest mock —
// use it instead of hand-rolling one, so persistence semantics match reality.
jest.mock("@react-native-async-storage/async-storage", () =>
  require("@react-native-async-storage/async-storage/jest/async-storage-mock"),
);

const mockFetch = jest.fn();
// @ts-expect-error - test-only global fetch stub
global.fetch = mockFetch;

beforeEach(async () => {
  await AsyncStorage.clear();
  mockFetch.mockReset();
});

describe("server URL persistence (Settings screen depends on this)", () => {
  it("returns null when nothing has been configured yet", async () => {
    await expect(getServerUrl()).resolves.toBeNull();
  });

  it("round-trips a URL, trimmed and without a trailing slash", async () => {
    await setServerUrl("  http://192.168.1.5:8000/  ");
    await expect(getServerUrl()).resolves.toBe("http://192.168.1.5:8000");
  });
});

describe("requestJson error handling (exercised via getHealth/getJobs)", () => {
  it("throws ApiError(0) with a clear message when no server URL is set", async () => {
    await expect(getHealth()).rejects.toMatchObject({
      name: "ApiError",
      status: 0,
      message: expect.stringContaining("No server URL configured"),
    });
    expect(mockFetch).not.toHaveBeenCalled();
  });

  it("throws a user-readable ApiError on network failure (INSTRUCTIONS.md §7)", async () => {
    await setServerUrl("http://192.168.1.5:8000");
    mockFetch.mockRejectedValueOnce(new TypeError("Network request failed"));

    await expect(getHealth()).rejects.toMatchObject({
      status: 0,
      message: expect.stringContaining("Can't reach server"),
    });
  });

  it("surfaces a string `detail` from a non-2xx response (e.g. 400 bad image)", async () => {
    await setServerUrl("http://192.168.1.5:8000");
    mockFetch.mockResolvedValueOnce({
      ok: false,
      status: 400,
      statusText: "Bad Request",
      json: async () => ({ detail: "Could not decode image 'x.jpg'." }),
    });

    await expect(getJobs()).rejects.toMatchObject({
      status: 400,
      message: "Could not decode image 'x.jpg'.",
    });
  });

  it("surfaces a FastAPI validation-error array `detail` (e.g. 422 bad job_category)", async () => {
    await setServerUrl("http://192.168.1.5:8000");
    mockFetch.mockResolvedValueOnce({
      ok: false,
      status: 422,
      statusText: "Unprocessable Entity",
      json: async () => ({ detail: [{ msg: "field required" }] }),
    });

    await expect(getJobs()).rejects.toMatchObject({
      status: 422,
      message: "field required",
    });
  });

  it("falls back to statusText when the error body isn't JSON", async () => {
    await setServerUrl("http://192.168.1.5:8000");
    mockFetch.mockResolvedValueOnce({
      ok: false,
      status: 500,
      statusText: "Internal Server Error",
      json: async () => {
        throw new SyntaxError("Unexpected end of JSON input");
      },
    });

    await expect(getJobs()).rejects.toMatchObject({
      status: 500,
      message: "Internal Server Error",
    });
  });
});

describe("getHealth / getJobs happy paths", () => {
  it("resolves the parsed HealthResponse on 200", async () => {
    await setServerUrl("http://192.168.1.5:8000");
    const body = {
      status: "ok",
      model_loaded: true,
      model_name: "yolo26n",
      device: "cpu",
    };
    mockFetch.mockResolvedValueOnce({ ok: true, json: async () => body });

    await expect(getHealth()).resolves.toEqual(body);
    expect(mockFetch).toHaveBeenCalledWith(
      "http://192.168.1.5:8000/api/v1/health",
      undefined,
    );
  });

  it("resolves the flat job map from GET /jobs (NOT the job_categories shape in INSTRUCTIONS.md)", async () => {
    await setServerUrl("http://192.168.1.5:8000");
    const body = { welder: { required: ["hardhat"], pending: ["boots"] } };
    mockFetch.mockResolvedValueOnce({ ok: true, json: async () => body });

    await expect(getJobs()).resolves.toEqual(body);
    expect(mockFetch).toHaveBeenCalledWith(
      "http://192.168.1.5:8000/api/v1/jobs",
      undefined,
    );
  });
});

describe("postComplianceCheck", () => {
  it("POSTs multipart form data with image, worker_id, and job_category to /compliance/check", async () => {
    await setServerUrl("http://192.168.1.5:8000");
    mockFetch.mockResolvedValueOnce({
      ok: true,
      json: async () => ({ verdict: "PASS" }),
    });

    await postComplianceCheck({
      imageUri: "file:///tmp/photo.jpg",
      workerId: "W-0042",
      jobCategory: "welder",
    });

    expect(mockFetch).toHaveBeenCalledTimes(1);
    const [url, init] = mockFetch.mock.calls[0];
    expect(url).toBe("http://192.168.1.5:8000/api/v1/compliance/check");
    expect(init.method).toBe("POST");
    expect(init.body).toBeInstanceOf(FormData);
  });
});
