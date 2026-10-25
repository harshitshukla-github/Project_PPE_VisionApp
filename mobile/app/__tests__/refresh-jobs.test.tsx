import AsyncStorage from "@react-native-async-storage/async-storage";
import { act, fireEvent, render, screen, waitFor } from "@testing-library/react-native";
import { Pressable, Text } from "react-native";

import HomeScreen from "../(tabs)/index";
import { SessionProvider, useSession } from "../../src/context/SessionContext";
import { SIGNED_IN, fail, mockRoutes, ok } from "../../src/testing/fixtures";

jest.mock("@react-native-async-storage/async-storage", () =>
  require("@react-native-async-storage/async-storage/jest/async-storage-mock"),
);

jest.mock("expo-router", () => {
  const { Text: RNText } = require("react-native");
  return { Link: ({ children }: { children: string }) => <RNText>{children}</RNText> };
});

const mockFetch = jest.fn();
// @ts-expect-error - test-only global fetch stub
global.fetch = mockFetch;

const ALL_JOBS = {
  general_labor: { required: ["hardhat", "safety_vest"], pending: ["boots"] },
  welder: { required: ["hardhat", "goggles"], pending: [] },
};
const WITHOUT_WELDER = { general_labor: ALL_JOBS.general_labor };

let jobsAnswer: () => ReturnType<typeof ok>;
let lastRefresh: unknown;

/** A button standing in for "Capture is about to send a check", plus the ways a worker's turn ends. */
function RefreshButton() {
  const { refreshJobs, jobCategory, jobNotice, signOut, expireSession } = useSession();
  return (
    <>
      <Text testID="selected">{jobCategory ?? "none"}</Text>
      <Text testID="job-notice">{jobNotice?.message ?? "no notice"}</Text>
      <Pressable testID="sign-out" onPress={() => void signOut()}>
        <Text>sign out</Text>
      </Pressable>
      <Pressable testID="expire" onPress={expireSession}>
        <Text>expire</Text>
      </Pressable>
      <Pressable
        testID="refresh"
        onPress={async () => {
          lastRefresh = await refreshJobs();
        }}
      >
        <Text>refresh</Text>
      </Pressable>
    </>
  );
}

function renderHome() {
  return render(
    <SessionProvider initialAuth={SIGNED_IN}>
      <RefreshButton />
      <HomeScreen />
    </SessionProvider>,
  );
}

async function refresh() {
  await act(async () => {
    fireEvent.press(screen.getByTestId("refresh"));
  });
}

beforeEach(async () => {
  await AsyncStorage.clear();
  await AsyncStorage.setItem("@ppb/server_url", "http://192.168.1.5:8000");
  mockFetch.mockReset();
  lastRefresh = undefined;
  jobsAnswer = () => ok(ALL_JOBS);
  mockRoutes(mockFetch, { "/jobs": () => jobsAnswer() });
});

describe("refreshJobs", () => {
  it("returns the current list and keeps the worker's job when it is still there", async () => {
    renderHome();
    fireEvent.press(await screen.findByText("○ Welder"));
    expect(screen.getByTestId("selected")).toHaveTextContent("welder");

    await refresh();

    expect(lastRefresh).toEqual(ALL_JOBS);
    expect(screen.getByTestId("selected")).toHaveTextContent("welder");
    expect(screen.queryByTestId("problem-notice")).toBeNull();
  });

  it("shows a job an admin added since the app loaded its list", async () => {
    renderHome();
    await screen.findByText("● General Labor");
    jobsAnswer = () =>
      ok({ ...ALL_JOBS, painter: { required: ["hardhat"], pending: [] } });

    await refresh();

    expect(await screen.findByText("○ Painter")).toBeTruthy();
  });

  it("clears the selection when the job was deactivated, and never swaps in another job", async () => {
    renderHome();
    fireEvent.press(await screen.findByText("○ Welder"));
    jobsAnswer = () => ok(WITHOUT_WELDER);

    await refresh();

    expect(screen.getByTestId("selected")).toHaveTextContent("none");
    expect(await screen.findByTestId("problem-notice")).toHaveTextContent(
      "Welder isn't available any more. Pick your job again.",
    );
    // General Labor is listed but is NOT selected for them.
    expect(screen.getByText("○ General Labor")).toBeTruthy();
    expect(screen.queryByText("● General Labor")).toBeNull();
  });

  it("shows that note in amber on Home, and removes it once the worker picks a job", async () => {
    renderHome();
    fireEvent.press(await screen.findByText("○ Welder"));
    jobsAnswer = () => ok(WITHOUT_WELDER);
    await refresh();

    expect(await screen.findByTestId("problem-notice")).toBeTruthy();
    expect(screen.queryByTestId("problem-error")).toBeNull();

    fireEvent.press(screen.getByText("○ General Labor"));

    await waitFor(() => expect(screen.queryByTestId("problem-notice")).toBeNull());
    expect(screen.getByTestId("selected")).toHaveTextContent("general_labor");
  });

  it.each([
    ["signing out", "sign-out"],
    ["an ended session", "expire"],
  ])("%s clears the note, so the next worker on a shared phone doesn't see it", async (_label, button) => {
    renderHome();
    fireEvent.press(await screen.findByText("○ Welder"));
    jobsAnswer = () => ok(WITHOUT_WELDER);
    await refresh();
    expect(screen.getByTestId("job-notice")).toHaveTextContent(/Welder isn't available any more/);

    await act(async () => {
      fireEvent.press(screen.getByTestId(button));
    });

    await waitFor(() => expect(screen.getByTestId("job-notice")).toHaveTextContent("no notice"));
  });

  it("returns null and changes nothing when the server can't be reached", async () => {
    renderHome();
    fireEvent.press(await screen.findByText("○ Welder"));
    jobsAnswer = () => {
      throw new TypeError("Network request failed");
    };

    await refresh();

    expect(lastRefresh).toBeNull();
    expect(screen.getByTestId("selected")).toHaveTextContent("welder");
    expect(screen.queryByTestId("problem-notice")).toBeNull();
  });

  it("returns null and keeps the selection on a 503", async () => {
    renderHome();
    fireEvent.press(await screen.findByText("○ Welder"));
    jobsAnswer = () => fail(503, { detail: "Job requirements are temporarily unavailable." });

    await refresh();

    expect(lastRefresh).toBeNull();
    expect(screen.getByTestId("selected")).toHaveTextContent("welder");
  });
});
