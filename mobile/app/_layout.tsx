import { Stack } from "expo-router";

/**
 * Root stack. The 3 persistent screens (Home, Capture, History) live in the
 * (tabs) group; Result and Settings are pushed on top as regular stack
 * screens since they aren't permanent navigation destinations.
 */
export default function RootLayout() {
  return (
    <Stack>
      <Stack.Screen name="(tabs)" options={{ headerShown: false }} />
      <Stack.Screen name="result" options={{ title: "Result" }} />
      <Stack.Screen
        name="settings"
        options={{ title: "Settings", presentation: "modal" }}
      />
    </Stack>
  );
}
