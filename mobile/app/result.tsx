import { useRouter } from "expo-router";
import { Pressable, StyleSheet, Text, View } from "react-native";

import CheckResultView from "../src/components/CheckResultView";
import { useSession } from "../src/context/SessionContext";

/**
 * Result screen — Phase 3 (demo milestone).
 *
 * Renders the last POST /compliance/check response from SessionContext via
 * the shared CheckResultView, plus a "Check again" loop back to Capture.
 */
export default function ResultScreen() {
  const router = useRouter();
  const { lastCheck, lastImageUri } = useSession();

  if (!lastCheck) {
    return (
      <View style={styles.center}>
        <Text style={styles.title}>No check yet</Text>
        <Text style={styles.subtitle}>
          Go to Capture to take a compliance photo.
        </Text>
        <Pressable style={styles.button} onPress={() => router.replace("/capture")}>
          <Text style={styles.buttonText}>Go to Capture</Text>
        </Pressable>
      </View>
    );
  }

  // On PASS/FAIL the loop is "check again". On NEEDS_REVIEW a clearer shot is
  // the way to resolve it, so that's the primary action (INSTRUCTIONS.md §2).
  const actionLabel = lastCheck.verdict === "PASS" || lastCheck.verdict === "FAIL"
    ? "Check again"
    : "Retake photo";

  return (
    <View style={styles.screen}>
      <CheckResultView result={lastCheck} imageUri={lastImageUri} />
      <View style={styles.footer}>
        <Pressable style={styles.button} onPress={() => router.replace("/capture")}>
          <Text style={styles.buttonText}>{actionLabel}</Text>
        </Pressable>
      </View>
    </View>
  );
}

const styles = StyleSheet.create({
  screen: {
    flex: 1,
  },
  center: {
    flex: 1,
    alignItems: "center",
    justifyContent: "center",
    gap: 12,
    padding: 20,
  },
  title: {
    fontSize: 20,
    fontWeight: "600",
  },
  subtitle: {
    color: "#666",
    textAlign: "center",
  },
  footer: {
    padding: 20,
    paddingTop: 0,
  },
  button: {
    backgroundColor: "#1d6fd6",
    borderRadius: 8,
    paddingVertical: 12,
    alignItems: "center",
  },
  buttonText: {
    color: "#fff",
    fontWeight: "600",
    fontSize: 15,
  },
});
