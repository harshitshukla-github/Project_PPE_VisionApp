import { StyleSheet, Text, View } from "react-native";

/**
 * History screen — placeholder for Phase 0.
 * Phase 4 adds: local AsyncStorage-backed list of past checks
 * (timestamp, job, verdict), tap-through to detail.
 */
export default function HistoryScreen() {
  return (
    <View style={styles.container}>
      <Text style={styles.title}>History</Text>
      <Text style={styles.subtitle}>Local past checks — Phase 4</Text>
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
