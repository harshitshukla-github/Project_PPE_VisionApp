import { Link } from "expo-router";
import {
  ActivityIndicator,
  Pressable,
  ScrollView,
  StyleSheet,
  Text,
  View,
} from "react-native";

import { useSession } from "../../src/context/SessionContext";

/** "general_labor" -> "General Labor" — INSTRUCTIONS.md §4: title-case the object key for the label. */
function titleCase(key: string): string {
  return key
    .split("_")
    .map((word) => word.charAt(0).toUpperCase() + word.slice(1))
    .join(" ");
}

/**
 * Home / Identify screen — Phase 2.
 *
 * Who is signed in (the worker ID now comes from login, not a typed field) +
 * job category picker populated live from GET /jobs (INSTRUCTIONS.md §2, §4 —
 * never hardcode the job list). Both live in SessionContext so Capture
 * can read them without re-fetching or re-entering anything.
 */
export default function HomeScreen() {
  const {
    auth,
    jobsStatus,
    reloadJobs,
    jobCategory,
    setJobCategory,
  } = useSession();

  return (
    <ScrollView
      style={styles.screen}
      contentContainerStyle={styles.container}
      keyboardShouldPersistTaps="handled"
    >
      <Text style={styles.title}>Home / Identify</Text>

      {auth.kind === "signedIn" && (
        <View style={styles.identity} testID="home-identity">
          <Text style={styles.identityName}>{auth.worker.display_name}</Text>
          <Text style={styles.identityId}>{auth.worker.worker_id}</Text>
        </View>
      )}

      <Text style={styles.label}>Job category</Text>
      {jobsStatus.kind === "loading" && (
        <View style={styles.statusRow}>
          <ActivityIndicator />
          <Text style={styles.statusText}>Loading jobs…</Text>
        </View>
      )}
      {jobsStatus.kind === "error" && (
        <View style={styles.statusRow}>
          <Text
            style={[
              styles.statusText,
              jobsStatus.tone === "notice" ? styles.noticeText : styles.errorText,
            ]}
          >
            {jobsStatus.tone === "notice"
              ? jobsStatus.message
              : `Can't load jobs ✗ — ${jobsStatus.message}`}
          </Text>
          <Pressable onPress={reloadJobs}>
            <Text style={styles.retryText}>Retry</Text>
          </Pressable>
        </View>
      )}
      {jobsStatus.kind === "loaded" && (
        <View style={styles.jobList}>
          {Object.keys(jobsStatus.jobs).map((key) => {
            const selected = key === jobCategory;
            return (
              <Pressable
                key={key}
                style={[styles.jobItem, selected && styles.jobItemSelected]}
                onPress={() => setJobCategory(key)}
              >
                <Text
                  style={[styles.jobItemText, selected && styles.jobItemTextSelected]}
                >
                  {selected ? "● " : "○ "}
                  {titleCase(key)}
                </Text>
              </Pressable>
            );
          })}
        </View>
      )}

      <View style={styles.footer}>
        <Link href="/settings">Settings</Link>
      </View>
    </ScrollView>
  );
}

const styles = StyleSheet.create({
  screen: {
    flex: 1,
  },
  container: {
    padding: 20,
    gap: 12,
  },
  title: {
    fontSize: 20,
    fontWeight: "600",
    marginBottom: 8,
  },
  label: {
    fontSize: 14,
    fontWeight: "500",
  },
  identity: {
    borderWidth: 1,
    borderColor: "#ccc",
    borderRadius: 8,
    paddingHorizontal: 12,
    paddingVertical: 10,
  },
  identityName: {
    fontSize: 16,
    fontWeight: "600",
  },
  identityId: {
    color: "#666",
    fontSize: 13,
  },
  statusRow: {
    flexDirection: "row",
    alignItems: "center",
    gap: 8,
  },
  statusText: {
    fontSize: 14,
  },
  errorText: {
    color: "#c0392b",
    fontWeight: "500",
    flexShrink: 1,
  },
  noticeText: {
    color: "#8a5a0b",
    fontWeight: "500",
    flexShrink: 1,
  },
  retryText: {
    color: "#1d6fd6",
    fontWeight: "600",
  },
  jobList: {
    gap: 8,
  },
  jobItem: {
    borderWidth: 1,
    borderColor: "#ccc",
    borderRadius: 8,
    paddingHorizontal: 12,
    paddingVertical: 10,
  },
  jobItemSelected: {
    borderColor: "#1d6fd6",
    backgroundColor: "#eaf2fd",
  },
  jobItemText: {
    fontSize: 15,
  },
  jobItemTextSelected: {
    color: "#1d6fd6",
    fontWeight: "600",
  },
  footer: {
    marginTop: 12,
    alignItems: "center",
  },
});
