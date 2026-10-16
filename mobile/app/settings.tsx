import { useRouter } from "expo-router";
import { useEffect, useState } from "react";
import {
  ActivityIndicator,
  Pressable,
  ScrollView,
  StyleSheet,
  Text,
  TextInput,
  View,
} from "react-native";

import { ApiError, getHealth, getServerUrl, setServerUrl } from "../src/api/client";
import type { HealthResponse } from "../src/api/types";
import { useSession } from "../src/context/SessionContext";

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
