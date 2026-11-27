import { Stack } from "expo-router";

import { SessionProvider, useSession } from "../src/context/SessionContext";

/**
 * Wrapped in SessionProvider so the signed-in worker, the job selected on
 * Home, and the last check result are available to every screen without
 * prop-drilling through the navigators (INSTRUCTIONS.md §3).
 */
export default function RootLayout() {
  return (
    <SessionProvider>
      <RootStack />
    </SessionProvider>
  );
}

/**
 * Root stack, gated by who is signed in (docs/auth-api.md §8). A screen whose
 * guard is false isn't reachable at all, and when a guard flips — login,
 * logout, an expired session, a forced PIN change — the router moves to the
 * screen the worker belongs on (`redirectTo`). That is what sends a worker to Login the
 * moment a request answers 401, with no per-screen redirect code.
 *
 * - Login: signed out (or still checking the saved token).
 * - Change PIN: signed in, but still on the PIN the supervisor issued.
 * - The 3 persistent screens (Home, Capture, History) live in the (tabs)
 *   group; Result and History detail are pushed on top.
 * - Settings is always reachable: the server URL must be set before the
 *   first login can work.
 */
function RootStack() {
  const { auth } = useSession();
  const signedIn = auth.kind === "signedIn";
  const mustChangePin = signedIn && auth.mustChangePin;

  // Where a worker belongs right now. Every guarded group redirects here when its guard turns
  // off: since SDK 58 the router's own fallback is the anchor or the first unguarded route in its
  // own order (Settings, here), not the next screen in this list.
  const home = !signedIn ? "/login" : mustChangePin ? "/change-pin" : "/";

  return (
    <Stack>
      <Stack.Protected guard={!signedIn} redirectTo={home}>
        <Stack.Screen name="login" options={{ title: "Sign in", headerShown: false }} />
      </Stack.Protected>
      <Stack.Protected guard={mustChangePin} redirectTo={home}>
        <Stack.Screen name="change-pin" options={{ title: "Choose a PIN", headerShown: false }} />
      </Stack.Protected>
      <Stack.Protected guard={signedIn && !mustChangePin} redirectTo={home}>
        <Stack.Screen name="(tabs)" options={{ headerShown: false }} />
        {/* Without a back title, iOS labels the button with the (tabs) group's route name. */}
        <Stack.Screen name="result" options={{ title: "Result", headerBackTitle: "Back" }} />
        <Stack.Screen
          name="history-detail"
          options={{ title: "Check detail", headerBackTitle: "Back" }}
        />
      </Stack.Protected>
      <Stack.Screen
        name="settings"
        options={{ title: "Settings", presentation: "modal" }}
      />
    </Stack>
  );
}
