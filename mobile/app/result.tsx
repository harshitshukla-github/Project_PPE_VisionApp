import { useRouter } from "expo-router";
import { Image, Pressable, ScrollView, StyleSheet, Text, View } from "react-native";

import { useSession } from "../src/context/SessionContext";

/** "no_hardhat" -> "No Hardhat" for readable list items. */
function titleCase(key: string): string {
  return key
    .split("_")
    .map((word) => word.charAt(0).toUpperCase() + word.slice(1))
    .join(" ");
}

/**
 * Result screen — Phase 3 (demo milestone).
 *
 * Renders the last POST /compliance/check response from SessionContext.
 * PASS: green check + detections. FAIL: red cross + explanation +
 * missing/violations. unverifiable_equipment always renders separately in
 * amber "verify manually" — it is never a failure (INSTRUCTIONS.md §2, §4).
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

  const isPass = lastCheck.verdict === "PASS";

  return (
    <ScrollView style={styles.screen} contentContainerStyle={styles.container}>
      {lastImageUri && (
        <>
          <Image
            source={{ uri: lastImageUri }}
            style={styles.photo}
            resizeMode="contain"
          />
          <Text style={styles.photoHint}>
            The full photo above is exactly what was sent for detection.
          </Text>
        </>
      )}

      <View style={styles.verdictRow}>
        <Text style={[styles.verdictIcon, isPass ? styles.passColor : styles.failColor]}>
          {isPass ? "✓" : "✗"}
        </Text>
        <Text style={[styles.verdictText, isPass ? styles.passColor : styles.failColor]}>
          {lastCheck.verdict}
        </Text>
      </View>

      {!isPass && lastCheck.explanation && (
        <View style={styles.section}>
          <Text style={styles.explanation}>{lastCheck.explanation}</Text>
        </View>
      )}

      {lastCheck.detections.length > 0 && (
        <View style={styles.section}>
          <Text style={styles.sectionTitle}>Detected</Text>
          {lastCheck.detections.map((d, i) => (
            <Text key={`${d.class}-${i}`} style={styles.listItem}>
              {titleCase(d.class)} — {Math.round(d.confidence * 100)}%
            </Text>
          ))}
        </View>
      )}

      {lastCheck.missing_equipment.length > 0 && (
        <View style={styles.section}>
          <Text style={styles.sectionTitle}>Missing equipment</Text>
          {lastCheck.missing_equipment.map((item) => (
            <Text key={item} style={[styles.listItem, styles.failColor]}>
              {titleCase(item)}
            </Text>
          ))}
        </View>
      )}

      {lastCheck.violations_detected.length > 0 && (
        <View style={styles.section}>
          <Text style={styles.sectionTitle}>Violations detected</Text>
          {lastCheck.violations_detected.map((item) => (
            <Text key={item} style={[styles.listItem, styles.failColor]}>
              {titleCase(item)}
            </Text>
          ))}
        </View>
      )}

      {lastCheck.unverifiable_equipment.length > 0 && (
        <View style={styles.section}>
          <Text style={styles.sectionTitle}>Verify manually</Text>
          {lastCheck.unverifiable_equipment.map((item) => (
            <Text key={item} style={[styles.listItem, styles.amberColor]}>
              {titleCase(item)}
            </Text>
          ))}
        </View>
      )}

      <Pressable style={styles.button} onPress={() => router.replace("/capture")}>
        <Text style={styles.buttonText}>Check again</Text>
      </Pressable>
    </ScrollView>
  );
}

const styles = StyleSheet.create({
  screen: {
    flex: 1,
  },
  container: {
    padding: 20,
    gap: 16,
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
  photo: {
    width: "100%",
    height: 360,
    borderRadius: 12,
    backgroundColor: "#eee",
  },
  photoHint: {
    color: "#999",
    fontSize: 11,
    textAlign: "center",
    marginTop: -8,
  },
  verdictRow: {
    alignItems: "center",
    gap: 4,
  },
  verdictIcon: {
    fontSize: 56,
    fontWeight: "700",
  },
  verdictText: {
    fontSize: 24,
    fontWeight: "700",
  },
  passColor: {
    color: "#1a7f37",
  },
  failColor: {
    color: "#c0392b",
  },
  amberColor: {
    color: "#b8860b",
  },
  section: {
    gap: 4,
  },
  sectionTitle: {
    fontSize: 14,
    fontWeight: "600",
    marginBottom: 2,
  },
  explanation: {
    fontSize: 15,
    lineHeight: 21,
  },
  listItem: {
    fontSize: 14,
  },
  button: {
    backgroundColor: "#1d6fd6",
    borderRadius: 8,
    paddingVertical: 12,
    alignItems: "center",
    marginTop: 8,
  },
  buttonText: {
    color: "#fff",
    fontWeight: "600",
    fontSize: 15,
  },
});
