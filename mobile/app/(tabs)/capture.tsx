import { StyleSheet, Text, View } from "react-native";

/**
 * Capture screen — placeholder for Phase 0.
 * Phase 3 adds: expo-camera live preview + shutter, expo-image-picker
 * fallback, and the multipart POST to /compliance/check.
 */
export default function CaptureScreen() {
  return (
    <View style={styles.container}>
      <Text style={styles.title}>Capture</Text>
      <Text style={styles.subtitle}>Camera preview + shutter — Phase 3</Text>
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
