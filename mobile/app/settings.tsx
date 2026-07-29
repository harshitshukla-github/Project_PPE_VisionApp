import { StyleSheet, Text, View } from "react-native";

/**
 * Settings screen — placeholder for Phase 0.
 * Phase 1 adds: editable server URL persisted via
 * src/api/client.ts#setServerUrl, and a GET /health "Connected ✓/✗" check
 * (INSTRUCTIONS.md §5 — base URL must never be hardcoded).
 */
export default function SettingsScreen() {
  return (
    <View style={styles.container}>
      <Text style={styles.title}>Settings</Text>
      <Text style={styles.subtitle}>Server URL + health check — Phase 1</Text>
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
