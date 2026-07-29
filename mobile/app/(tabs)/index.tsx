import { Link } from "expo-router";
import { StyleSheet, Text, View } from "react-native";

/**
 * Home / Identify screen — placeholder for Phase 0.
 * Phase 2 adds: worker_id input (remembered via AsyncStorage) and the
 * job_category dropdown populated from GET /jobs.
 */
export default function HomeScreen() {
  return (
    <View style={styles.container}>
      <Text style={styles.title}>Home / Identify</Text>
      <Text style={styles.subtitle}>Worker ID + job selection — Phase 2</Text>
      <Link href="/settings">Settings</Link>
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
