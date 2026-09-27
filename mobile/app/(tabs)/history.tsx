import { useRouter } from "expo-router";
import { FlatList, Pressable, StyleSheet, Text, View } from "react-native";

import { useSession } from "../../src/context/SessionContext";
import type { HistoryEntry } from "../../src/storage/history";

/** "general_labor" -> "General Labor" — same convention as the Home job picker. */
function titleCase(key: string): string {
  return key
    .split("_")
    .map((word) => word.charAt(0).toUpperCase() + word.slice(1))
    .join(" ");
}

/**
 * History screen — Phase 4.
 *
 * Local on-device list of past checks (INSTRUCTIONS.md §2: "past checks
 * stored locally on-device (AsyncStorage): timestamp, job, verdict. No
 * backend persistence exists yet"), most-recent-first, tapping through to
 * a detail view rendered with the same CheckResultView as Result.
 */
export default function HistoryScreen() {
  const router = useRouter();
  const { history } = useSession();

  if (history.length === 0) {
    return (
      <View style={styles.empty}>
        <Text style={styles.title}>History</Text>
        <Text style={styles.subtitle}>
          No checks yet — completed compliance checks will show up here.
        </Text>
      </View>
    );
  }

  return (
    <FlatList
      style={styles.screen}
      contentContainerStyle={styles.container}
      data={history}
      keyExtractor={(entry) => entry.id}
      ItemSeparatorComponent={() => <View style={styles.separator} />}
      renderItem={({ item }) => <HistoryRow entry={item} onPress={() => router.push(`/history-detail?id=${item.id}`)} />}
    />
  );
}

function HistoryRow({ entry, onPress }: { entry: HistoryEntry; onPress: () => void }) {
  const isPass = entry.result.verdict === "PASS";
  const when = new Date(entry.result.timestamp).toLocaleString();

  return (
    <Pressable style={styles.row} onPress={onPress}>
      <View style={styles.rowMain}>
        <Text style={styles.job}>{titleCase(entry.result.job_category)}</Text>
        <Text style={styles.meta}>
          {entry.result.worker_id} · {when}
        </Text>
      </View>
      <Text style={[styles.verdict, isPass ? styles.passColor : styles.failColor]}>
        {isPass ? "PASS" : "FAIL"}
      </Text>
    </Pressable>
  );
}

const styles = StyleSheet.create({
  screen: {
    flex: 1,
  },
  container: {
    padding: 20,
  },
  empty: {
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
  separator: {
    height: 1,
    backgroundColor: "#eee",
  },
  row: {
    flexDirection: "row",
    alignItems: "center",
    justifyContent: "space-between",
    paddingVertical: 14,
  },
  rowMain: {
    gap: 2,
  },
  job: {
    fontSize: 16,
    fontWeight: "600",
  },
  meta: {
    fontSize: 13,
    color: "#666",
  },
  verdict: {
    fontSize: 14,
    fontWeight: "700",
  },
  passColor: {
    color: "#1a7f37",
  },
  failColor: {
    color: "#c0392b",
  },
});
