import { Stack } from "expo-router";

import { SessionProvider } from "../src/context/SessionContext";

/**
 * Root stack. The 3 persistent screens (Home, Capture, History) live in the
 * (tabs) group; Result and Settings are pushed on top as regular stack
 * screens since they aren't permanent navigation destinations.
 *
 * Wrapped in SessionProvider so the worker ID / job category selected on
 * Home, and the last check result, are available to Capture and Result
 * without prop-drilling through the tab navigator (INSTRUCTIONS.md §3).
 */
export default function RootLayout() {
  return (
    <SessionProvider>
      <Stack>
        <Stack.Screen name="(tabs)" options={{ headerShown: false }} />
        <Stack.Screen name="result" options={{ title: "Result" }} />
        <Stack.Screen
          name="settings"
          options={{ title: "Settings", presentation: "modal" }}
        />
      </Stack>
    </SessionProvider>
  );
}
