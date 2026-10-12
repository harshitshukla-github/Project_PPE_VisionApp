import { useLocalSearchParams } from "expo-router";
import { StyleSheet, Text, View } from "react-native";

import CheckResultView from "../src/components/CheckResultView";
import { useSession } from "../src/context/SessionContext";

/** History detail screen — Phase 4. Renders one past entry via CheckResultView. */
export default function HistoryDetailScreen() {
  const { id } = useLocalSearchParams<{ id: string }>();
  const { history } = useSession();
  const entry = history.find((e) => e.id === id);

  if (!entry) {
    return (
      <View style={styles.center}>
        <Text style={styles.title}>Not found</Text>
        <Text style={styles.subtitle}>This history entry is no longer available.</Text>
      </View>
    );
  }

  return <CheckResultView result={entry.result} imageUri={entry.imageUri} />;
}

const styles = StyleSheet.create({
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
});
