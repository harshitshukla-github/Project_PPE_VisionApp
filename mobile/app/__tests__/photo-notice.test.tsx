import AsyncStorage from "@react-native-async-storage/async-storage";
import { act, fireEvent, renderRouter, screen, waitFor } from "expo-router/testing-library";
import { Text } from "react-native";

import ChangePinScreen from "../change-pin";
import PhotoNoticeScreen from "../photo-notice";
import RootLayout from "../_layout";
import LoginScreen from "../login";
import { clearToken, setToken } from "../../src/api/client";

jest.mock("@react-native-async-storage/async-storage", () =>
  require("@react-native-async-storage/async-storage/jest/async-storage-mock"),
);

const mockFetch = jest.fn();
// @ts-expect-error - test-only global fetch stub
global.fetch = mockFetch;

const ME = {
  worker: { worker_id: "W-0042", display_name: "Asha Verma" },
  must_change_pin: false,
  expires_at: "2026-12-16T03:12:00Z",
};

const NOTICE_1 = {
  version: 1,
  text: "Photos from your PPE checks are kept on the company's server for up to 90 days.",
  training_question: "Can we also keep your photos for up to 730 days to improve the PPE detector?",
};
const NOTICE_2 = { ...NOTICE_1, version: 2, text: "Photos are now kept for up to 60 days." };

const consent = (notice: typeof NOTICE_1, seen: number | null, training = false) => ({
  notice,
  seen_version: seen,
  training,
  updated_at: seen === null ? null : "2026-11-20T06:10:00Z",
});

const json = (status: number, body: unknown) => ({ ok: status < 400, status, json: async () => body });

/**
 * A fake server that tells GET from PUT on /auth/worker/consent. `put` decides
 * the answer to each PUT; every PUT body is recorded in `answers`.
 */
function server(options: {
  meChangePin?: boolean;
  get: () => ReturnType<typeof json>;
  put?: (body: { version: number; training: boolean }) => ReturnType<typeof json>;
}) {
  const answers: Array<{ version: number; training: boolean }> = [];
  mockFetch.mockImplementation(async (url: string, init?: RequestInit) => {
    const method = init?.method ?? "GET";
    if (url.endsWith("/jobs")) return json(200, {});
    if (url.endsWith("/auth/worker/me")) return json(200, { ...ME, must_change_pin: !!options.meChangePin });
    if (url.endsWith("/auth/worker/pin")) return { ok: true, status: 204 };
    if (url.endsWith("/auth/worker/consent") && method === "GET") return options.get();
    if (url.endsWith("/auth/worker/consent") && method === "PUT") {
      const body = JSON.parse(String(init?.body));
      answers.push(body);
      return options.put ? options.put(body) : json(200, consent(NOTICE_1, body.version, body.training));
    }
    throw new TypeError(`Network request failed (no route for ${method} ${url})`);
  });
  return answers;
}

/** The real layout, Login, Change PIN and notice screens; plain stand-ins for the rest. */
function renderApp() {
  return renderRouter(
    {
      _layout: RootLayout,
      login: LoginScreen,
      "change-pin": ChangePinScreen,
      "photo-notice": PhotoNoticeScreen,
      "(tabs)/_layout": () => {
        const { Slot } = require("expo-router");
        return <Slot />;
      },
      "(tabs)/index": () => <Text>HOME-STUB</Text>,
      result: () => <Text>RESULT-STUB</Text>,
      "history-detail": () => <Text>DETAIL-STUB</Text>,
      settings: () => <Text>SETTINGS-STUB</Text>,
    },
    { initialUrl: "/" },
  );
}

beforeEach(async () => {
  await AsyncStorage.clear();
  await clearToken();
  await setToken("tok-123");
  mockFetch.mockReset();
  await AsyncStorage.setItem("@ppb/server_url", "http://192.168.1.5:8000");
});

async function pressContinue() {
  await act(async () => {
    fireEvent.press(screen.getByRole("button", { name: "Continue" }));
  });
}

describe("when the notice shows (docs/checks-api.md §5)", () => {
  it("shows it, in the server's words, to a worker who has never answered it", async () => {
    server({ get: () => json(200, consent(NOTICE_1, null)) });

    renderApp();

    expect(await screen.findByText(NOTICE_1.text)).toBeTruthy();
    expect(screen.getByText(NOTICE_1.training_question)).toBeTruthy();
    expect(screen.queryByText("HOME-STUB")).toBeNull();
  });

  it("shows it again when there is a newer version than the one they answered", async () => {
    server({ get: () => json(200, consent(NOTICE_2, 1, true)) });

    renderApp();

    expect(await screen.findByText(NOTICE_2.text)).toBeTruthy();
  });

  it("opens the app when the current version is already answered", async () => {
    server({ get: () => json(200, consent(NOTICE_1, 1)) });

    renderApp();

    expect(await screen.findByText("HOME-STUB")).toBeTruthy();
    expect(screen.queryByTestId("photo-notice")).toBeNull();
  });

  it("opens the app when the server can't say (a check is never refused for consent)", async () => {
    server({ get: () => json(404, { detail: "Not Found" }) });

    renderApp();

    expect(await screen.findByText("HOME-STUB")).toBeTruthy();
    await waitFor(() =>
      expect(mockFetch.mock.calls.some(([url]) => String(url).endsWith("/auth/worker/consent"))).toBe(true),
    );
    expect(screen.queryByTestId("photo-notice")).toBeNull();
  });

  it("comes after the PIN change, not before: the server refuses consent until the PIN is the worker's own", async () => {
    server({ meChangePin: true, get: () => json(200, consent(NOTICE_1, null)) });
    renderApp();
    expect(await screen.findByText("Choose your own PIN")).toBeTruthy();
    expect(mockFetch.mock.calls.some(([url]) => String(url).endsWith("/auth/worker/consent"))).toBe(false);

    fireEvent.changeText(screen.getByPlaceholderText("The PIN you were given"), "482913");
    const [next, again] = screen.getAllByPlaceholderText("6 digits");
    fireEvent.changeText(next, "730154");
    fireEvent.changeText(again, "730154");
    await act(async () => {
      fireEvent.press(screen.getByRole("button", { name: "Save new PIN" }));
    });

    expect(await screen.findByText(NOTICE_1.text)).toBeTruthy();
  });
});

describe("answering it", () => {
  it("sends the version shown, with training off unless the worker says yes, then opens the app", async () => {
    const answers = server({ get: () => json(200, consent(NOTICE_1, null)) });
    renderApp();
    await screen.findByText(NOTICE_1.text);
    expect(screen.getByTestId("training-switch").props.value).toBe(false);

    await pressContinue();

    expect(answers).toEqual([{ version: 1, training: false }]);
    expect(await screen.findByText("HOME-STUB")).toBeTruthy();
  });

  it("sends training: true when the worker turns the switch on", async () => {
    const answers = server({ get: () => json(200, consent(NOTICE_1, null)) });
    renderApp();
    await screen.findByText(NOTICE_1.text);

    fireEvent(screen.getByTestId("training-switch"), "valueChange", true);
    expect(screen.getByText("Yes")).toBeTruthy();
    await pressContinue();

    expect(answers).toEqual([{ version: 1, training: true }]);
  });

  it("on 409 notice_changed shows the new wording, asks again from 'No', and answers the new version", async () => {
    let calls = 0;
    const answers = server({
      get: () => json(200, consent(NOTICE_1, null)),
      put: (body) =>
        ++calls === 1
          ? json(409, { detail: "The notice has changed since the app showed it.", code: "notice_changed", notice: NOTICE_2 })
          : json(200, consent(NOTICE_2, body.version, body.training)),
    });
    renderApp();
    await screen.findByText(NOTICE_1.text);
    fireEvent(screen.getByTestId("training-switch"), "valueChange", true);

    await pressContinue();

    expect(await screen.findByText(NOTICE_2.text)).toBeTruthy();
    expect(screen.getByText(/The notice changed while you were reading it/)).toBeTruthy();
    expect(screen.getByTestId("training-switch").props.value).toBe(false);
    expect(screen.queryByText("HOME-STUB")).toBeNull();

    await pressContinue();

    expect(answers).toEqual([
      { version: 1, training: true },
      { version: 2, training: false },
    ]);
    expect(await screen.findByText("HOME-STUB")).toBeTruthy();
  });

  it("stays on the notice and says so when the answer can't be saved", async () => {
    server({ get: () => json(200, consent(NOTICE_1, null)), put: () => json(503, { detail: "Busy.", code: "auth_unavailable" }) });
    renderApp();
    await screen.findByText(NOTICE_1.text);

    await pressContinue();

    expect(await screen.findByTestId("photo-notice")).toBeTruthy();
    expect(screen.queryByText("HOME-STUB")).toBeNull();
    expect(screen.getByRole("button", { name: "Continue" })).toBeTruthy();
  });
});
