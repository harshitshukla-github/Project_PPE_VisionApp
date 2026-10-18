import AsyncStorage from "@react-native-async-storage/async-storage";
import * as SecureStore from "expo-secure-store";

import {
  changeWorkerPin,
  clearToken,
  getJobs,
  getToken,
  setServerUrl,
  setToken,
  workerLogin,
  workerLogout,
} from "../client";

jest.mock("@react-native-async-storage/async-storage", () =>
  require("@react-native-async-storage/async-storage/jest/async-storage-mock"),
);

const mockFetch = jest.fn();
// @ts-expect-error - test-only global fetch stub
global.fetch = mockFetch;

beforeEach(async () => {
  await AsyncStorage.clear();
  await clearToken();
  mockFetch.mockReset();
  await setServerUrl("http://192.168.1.5:8000");
});

describe("session token (docs/auth-api.md §8)", () => {
  it("keeps the token in secure storage, not AsyncStorage", async () => {
    await setToken("abc123");

    expect(SecureStore.setItemAsync).toHaveBeenCalledWith("ppb_worker_token", "abc123");
    await expect(getToken()).resolves.toBe("abc123");
    await expect(AsyncStorage.getAllKeys()).resolves.not.toContain("ppb_worker_token");
  });

  it("sends Authorization: Bearer on requests once signed in", async () => {
    await setToken("abc123");
    mockFetch.mockResolvedValueOnce({ ok: true, json: async () => ({}) });

    await getJobs();

    expect(mockFetch).toHaveBeenCalledWith("http://192.168.1.5:8000/api/v1/jobs", {
      headers: { Authorization: "Bearer abc123" },
    });
  });

  it("falls back to memory if secure storage can't save, so this session still works", async () => {
    (SecureStore.setItemAsync as jest.Mock).mockRejectedValueOnce(new Error("keystore locked"));

    await setToken("abc123");

    await expect(getToken()).resolves.toBe("abc123");
  });

  it("forgets the token even if secure storage can't delete it", async () => {
    await setToken("abc123");
    (SecureStore.deleteItemAsync as jest.Mock).mockRejectedValueOnce(new Error("keystore locked"));

    await clearToken();

    await expect(getToken()).resolves.toBeNull();
  });
});

describe("worker auth endpoints", () => {
  it("POSTs the worker ID (trimmed) and PIN as JSON to /auth/worker/login", async () => {
    const body = {
      token: "t",
      token_type: "bearer",
      expires_at: "2026-10-16T03:12:00Z",
      must_change_pin: false,
      worker: { worker_id: "W-0042", display_name: "Asha" },
    };
    mockFetch.mockResolvedValueOnce({ ok: true, json: async () => body });

    await expect(workerLogin("  W-0042 ", "482913")).resolves.toEqual(body);

    const [url, init] = mockFetch.mock.calls[0];
    expect(url).toBe("http://192.168.1.5:8000/api/v1/auth/worker/login");
    expect(init.method).toBe("POST");
    expect(init.headers).toMatchObject({ "Content-Type": "application/json" });
    expect(JSON.parse(init.body)).toEqual({ worker_id: "W-0042", pin: "482913" });
  });

  it("carries the error `code` and `retry_after_s` onto the ApiError", async () => {
    mockFetch.mockResolvedValueOnce({
      ok: false,
      status: 423,
      json: async () => ({
        detail: "Too many wrong PINs. Try again in 14 minutes.",
        code: "account_locked",
        retry_after_s: 840,
      }),
    });

    await expect(workerLogin("W-0042", "000000")).rejects.toMatchObject({
      status: 423,
      code: "account_locked",
      retryAfterS: 840,
      message: "Too many wrong PINs. Try again in 14 minutes.",
    });
  });

  it("treats 204 No Content from /auth/worker/pin as success", async () => {
    mockFetch.mockResolvedValueOnce({ ok: true, status: 204 });

    await expect(changeWorkerPin("482913", "730154")).resolves.toBeUndefined();
    expect(JSON.parse(mockFetch.mock.calls[0][1].body)).toEqual({
      current_pin: "482913",
      new_pin: "730154",
    });
  });

  it("deletes the stored token on logout whatever the server answers", async () => {
    await setToken("abc123");
    mockFetch.mockResolvedValueOnce({
      ok: false,
      status: 401,
      json: async () => ({ detail: "expired", code: "not_authenticated" }),
    });

    await workerLogout();

    await expect(getToken()).resolves.toBeNull();
    expect(mockFetch.mock.calls[0][0]).toBe(
      "http://192.168.1.5:8000/api/v1/auth/worker/logout",
    );
  });

  it("deletes the stored token on logout even when the server can't be reached", async () => {
    await setToken("abc123");
    mockFetch.mockRejectedValueOnce(new TypeError("Network request failed"));

    await expect(workerLogout()).resolves.toBeUndefined();

    await expect(getToken()).resolves.toBeNull();
  });
});
