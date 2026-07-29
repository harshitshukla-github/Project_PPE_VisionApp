import { StyleSheet, Text, View } from "react-native";

/**
 * Result screen — placeholder for Phase 0.
 * Phase 3 adds: PASS (green check + detections) / FAIL (red cross +
 * explanation + missing_equipment + violations_detected) states, plus
 * unverifiable_equipment shown separately in amber, and "Check again".
 */
export default function ResultScreen() {
  return (
    <View style={styles.container}>
      <Text style={styles.title}>Result</Text>
      <Text style={styles.subtitle}>PASS / FAIL rendering — Phase 3</Text>
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    alignItems: "center",
    justifyContent: "center",
    gap: 12,
  },
  title: {
    fontSize: 20,
    fontWeight: "600",
  },
  subtitle: {
    color: "#666",
  },
});
