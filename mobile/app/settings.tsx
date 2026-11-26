import { useRouter } from "expo-router";
import { useEffect, useState } from "react";
import {
  ActivityIndicator,
  Pressable,
  ScrollView,
  StyleSheet,
  Switch,
  Text,
  TextInput,
  View,
} from "react-native";

import { ApiError, getHealth, getServerUrl, setServerUrl } from "../src/api/client";
import { problemFrom, type Problem } from "../src/api/problems";
import type { HealthResponse, PhotoNotice } from "../src/api/types";
import { PhotoNoticeView } from "../src/components/PhotoNoticeView";
import { ProblemBanner } from "../src/components/ProblemBanner";
import { noticeFrom, useSession } from "../src/context/SessionContext";

type ConnectionStatus =
  | { kind: "idle" }
  | { kind: "checking" }
  | { kind: "connected"; health: HealthResponse }
  | { kind: "error"; message: string };

/**
 * Settings screen — Phase 1.
 *
 * Persists the backend base URL (INSTRUCTIONS.md §5 — never hardcode it,
 * since every dev's LAN IP differs and "localhost" on a phone means the
 * phone itself) and verifies it against GET /health.
 */
export default function SettingsScreen() {
  const router = useRouter();
  const { auth, signOut } = useSession();
  const [urlInput, setUrlInput] = useState("");
  const [status, setStatus] = useState<ConnectionStatus>({ kind: "idle" });

  useEffect(() => {
    (async () => {
      const saved = await getServerUrl();
      if (saved) {
        setUrlInput(saved);
        await checkConnection();
      }
    })();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  async function checkConnection() {
    setStatus({ kind: "checking" });
    try {
      const health = await getHealth();
      setStatus({ kind: "connected", health });
    } catch (err) {
      const message =
        err instanceof ApiError ? err.message : "Unexpected error checking connection.";
      setStatus({ kind: "error", message });
    }
  }

  async function handleSave() {
    const trimmed = urlInput.trim();
    if (!trimmed) {
      setStatus({ kind: "error", message: "Enter a server URL first." });
      return;
    }
    await setServerUrl(trimmed);
    setUrlInput(trimmed.replace(/\/+$/, ""));
    await checkConnection();
  }

  async function handleLogOut() {
    await signOut();
    // Settings is a modal over the app; replace it so Login is what's left.
    router.replace("/login");
  }

  return (
    <ScrollView
      style={styles.screen}
      contentContainerStyle={styles.container}
      keyboardShouldPersistTaps="handled"
    >
      <Text style={styles.title}>Settings</Text>

      <Text style={styles.label}>Backend server URL</Text>
      <TextInput
        style={styles.input}
        value={urlInput}
        onChangeText={setUrlInput}
        placeholder="http://192.168.1.5:8000"
        placeholderTextColor="#999"
        autoCapitalize="none"
        autoCorrect={false}
        autoComplete="off"
        keyboardType="url"
      />
      <Text style={styles.hint}>
        Use your computer&apos;s LAN IP, not "localhost" — on a phone,
        localhost means the phone itself. Android emulators (not physical
        phones) should use 10.0.2.2 instead of the host IP.
      </Text>

      <Pressable style={styles.button} onPress={handleSave}>
        <Text style={styles.buttonText}>Save &amp; Test Connection</Text>
      </Pressable>

      <View style={styles.statusRow}>
        {status.kind === "idle" && (
          <Text style={styles.statusText}>Not tested yet.</Text>
        )}
        {status.kind === "checking" && (
          <>
            <ActivityIndicator />
            <Text style={styles.statusText}>Checking…</Text>
          </>
        )}
        {status.kind === "connected" && (
          <Text style={[styles.statusText, styles.connected]}>
            Connected ✓ — {status.health.model_name} on {status.health.device}
            {status.health.model_loaded ? "" : " (model not loaded)"}
          </Text>
        )}
        {status.kind === "error" && (
          <Text style={[styles.statusText, styles.errorText]}>
            Not connected ✗ — {status.message}
          </Text>
        )}
      </View>

      {auth.kind === "signedIn" && !auth.mustChangePin && <PhotoSettings />}

      {auth.kind === "signedIn" && (
        <View style={styles.account}>
          <Text style={styles.label}>Account</Text>
          <Text style={styles.statusText}>
            Signed in as {auth.worker.display_name} ({auth.worker.worker_id})
          </Text>
          <Pressable
            testID="logout-button"
            style={styles.logoutButton}
            onPress={handleLogOut}
            accessibilityRole="button"
          >
            <Text style={styles.logoutText}>Log out</Text>
          </Pressable>
        </View>
      )}
    </ScrollView>
  );
}

/**
 * "Your photos" (docs/checks-api.md §5): the training choice, the notice to
 * read again, and who to ask about deleting photos.
 *
 * Changing the choice asks the server first. If a newer notice has come out
 * since the worker last answered, it is shown here before anything is saved,
 * and the answer goes with that version; otherwise the choice is saved at once.
 */
function PhotoSettings() {
  const { consent, refreshConsent, answerNotice } = useSession();
  // A newer notice to read before the choice is saved, with the choice the worker just made.
  const [pending, setPending] = useState<{ notice: PhotoNotice; training: boolean } | null>(null);
  const [busy, setBusy] = useState(false);
  const [problem, setProblem] = useState<Problem | null>(null);
  const [saved, setSaved] = useState<string | null>(null);
  const [showNotice, setShowNotice] = useState(false);

  useEffect(() => {
    if (!consent) void refreshConsent();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  async function save(version: number, training: boolean) {
    try {
      const answer = await answerNotice(version, training);
      setPending(null);
      setSaved(
        answer.training
          ? "Saved. Your photos may be kept to improve the PPE detector."
          : "Saved. Your photos won't be kept for training.",
      );
    } catch (err) {
      const newer = noticeFrom(err);
      if (newer) {
        setPending({ notice: newer, training: false });
        setProblem({ message: "The notice changed. Please read it and answer again.", tone: "notice" });
      } else {
        setProblem(problemFrom(err, "Couldn't save that. Try again."));
      }
    } finally {
      setBusy(false);
    }
  }

  async function change(training: boolean) {
    setBusy(true);
    setProblem(null);
    setSaved(null);
    const fresh = await refreshConsent();
    if (!fresh) {
      setProblem({ message: "Couldn't reach the server to change this. Try again.", tone: "error" });
      setBusy(false);
      return;
    }
    if (fresh.seen_version === null || fresh.seen_version < fresh.notice.version) {
      setPending({ notice: fresh.notice, training });
      setBusy(false);
      return;
    }
    await save(fresh.notice.version, training);
  }

  return (
    <View style={styles.account} testID="photo-settings">
      <Text style={styles.label}>Your photos</Text>

      {pending ? (
        <>
          <Text style={styles.hint}>There&apos;s a new photo notice. Read it before your choice is saved.</Text>
          <PhotoNoticeView
            notice={pending.notice}
            training={pending.training}
            onTrainingChange={(training) => setPending({ ...pending, training })}
            onSubmit={() => {
              setBusy(true);
              setProblem(null);
              void save(pending.notice.version, pending.training);
            }}
            submitting={busy}
            problem={problem}
            submitLabel="Save my answer"
          />
        </>
      ) : (
        <>
          {problem && <ProblemBanner problem={problem} />}
          <View style={styles.switchRow}>
            <Text style={[styles.statusText, styles.switchLabel]}>Keep my photos to improve the PPE detector</Text>
            {busy ? (
              <ActivityIndicator />
            ) : (
              <Switch
                value={consent?.training ?? false}
                onValueChange={(value) => void change(value)}
                disabled={!consent}
                accessibilityLabel="Keep my photos to improve the PPE detector"
                testID="settings-training-switch"
              />
            )}
          </View>
          {saved && <Text style={[styles.statusText, styles.connected]}>{saved}</Text>}
          {consent && (
            <Pressable onPress={() => setShowNotice((shown) => !shown)} accessibilityRole="button">
              <Text style={styles.linkText}>{showNotice ? "Hide the photo notice" : "Read the photo notice"}</Text>
            </Pressable>
          )}
          {consent && showNotice && <Text style={styles.hint}>{consent.notice.text}</Text>}
        </>
      )}

      <Text style={styles.hint}>To have your photos deleted, ask your supervisor.</Text>
    </View>
  );
}

const styles = StyleSheet.create({
  screen: {
    flex: 1,
  },
  switchRow: {
    flexDirection: "row",
    alignItems: "center",
    justifyContent: "space-between",
    gap: 12,
  },
  switchLabel: {
    flex: 1,
  },
  linkText: {
    color: "#1d6fd6",
    fontWeight: "600",
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
  input: {
    borderWidth: 1,
    borderColor: "#ccc",
    borderRadius: 8,
    paddingHorizontal: 12,
    paddingVertical: 10,
    fontSize: 16,
  },
  hint: {
    color: "#666",
    fontSize: 12,
    lineHeight: 16,
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
  statusRow: {
    flexDirection: "row",
    alignItems: "center",
    gap: 8,
    marginTop: 8,
  },
  statusText: {
    fontSize: 14,
  },
  account: {
    marginTop: 24,
    paddingTop: 16,
    borderTopWidth: 1,
    borderTopColor: "#ddd",
    gap: 8,
  },
  logoutButton: {
    borderWidth: 1,
    borderColor: "#c0392b",
    borderRadius: 8,
    paddingVertical: 12,
    alignItems: "center",
  },
  logoutText: {
    color: "#c0392b",
    fontWeight: "600",
    fontSize: 15,
  },
  connected: {
    color: "#1a7f37",
    fontWeight: "500",
  },
  errorText: {
    color: "#c0392b",
    fontWeight: "500",
  },
});
