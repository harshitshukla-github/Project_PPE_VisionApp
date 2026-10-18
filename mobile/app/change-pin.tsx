import { useState } from "react";
import {
  ActivityIndicator,
  Pressable,
  ScrollView,
  StyleSheet,
  Text,
  TextInput,
  View,
} from "react-native";
import { useRouter } from "expo-router";

import { ApiError } from "../src/api/client";
import { problemFrom, type Problem } from "../src/api/problems";
import { ProblemBanner } from "../src/components/ProblemBanner";
import { useSession } from "../src/context/SessionContext";

const digitsOnly = (text: string) => text.replace(/\D/g, "").slice(0, 6);

/**
 * Change PIN screen — docs/auth-api.md §4, §8.
 *
 * Shown by the root layout while the signed-in worker is still on the PIN
 * their supervisor issued (login said must_change_pin, or the server answered
 * 403 pin_change_required). Nothing else works until a new PIN is chosen, so
 * the only other way out is signing out. The server judges whether a PIN is
 * too easy to guess (422 weak_pin) and says why; we only check the basics.
 */
export default function ChangePinScreen() {
  const router = useRouter();
  const { auth, changePin, signOut, expireSession } = useSession();
  const [currentPin, setCurrentPin] = useState("");
  const [newPin, setNewPin] = useState("");
  const [confirmPin, setConfirmPin] = useState("");
  const [submitting, setSubmitting] = useState(false);
  const [problem, setProblem] = useState<Problem | null>(null);

  const mismatch = confirmPin.length === 6 && confirmPin !== newPin;
  const canSubmit =
    currentPin.length === 6 && newPin.length === 6 && confirmPin === newPin && !submitting;

  async function handleSubmit() {
    if (!canSubmit) return;
    setSubmitting(true);
    setProblem(null);
    try {
      await changePin(currentPin, newPin);
      // The root layout now sees mustChangePin = false and opens the app.
    } catch (err) {
      if (err instanceof ApiError && err.code === "not_authenticated") {
        expireSession();
        return;
      }
      if (err instanceof ApiError && err.code === "invalid_credentials") {
        setProblem({ message: "That isn't your current PIN.", tone: "error" });
      } else {
        setProblem(problemFrom(err, "Couldn't change your PIN. Try again."));
      }
      setSubmitting(false);
    }
  }

  async function handleSignOut() {
    await signOut();
    router.replace("/login");
  }

  return (
    <ScrollView
      style={styles.screen}
      contentContainerStyle={styles.container}
      keyboardShouldPersistTaps="handled"
    >
      <Text style={styles.title}>Choose your own PIN</Text>
      <Text style={styles.hint}>
        {auth.kind === "signedIn" ? `${auth.worker.display_name}, the` : "The"} PIN you were
        given is only for your first sign-in. Pick a 6-digit PIN that only you know — not
        something like 111111 or 123456.
      </Text>

      {problem && <ProblemBanner problem={problem} />}

      <Text style={styles.label}>Current PIN</Text>
      <TextInput
        style={styles.input}
        value={currentPin}
        onChangeText={(t) => setCurrentPin(digitsOnly(t))}
        placeholder="The PIN you were given"
        placeholderTextColor="#999"
        keyboardType="number-pad"
        secureTextEntry
        maxLength={6}
        autoComplete="off"
        editable={!submitting}
      />

      <Text style={styles.label}>New PIN</Text>
      <TextInput
        style={styles.input}
        value={newPin}
        onChangeText={(t) => setNewPin(digitsOnly(t))}
        placeholder="6 digits"
        placeholderTextColor="#999"
        keyboardType="number-pad"
        secureTextEntry
        maxLength={6}
        autoComplete="off"
        editable={!submitting}
      />

      <Text style={styles.label}>Type the new PIN again</Text>
      <TextInput
        style={styles.input}
        value={confirmPin}
        onChangeText={(t) => setConfirmPin(digitsOnly(t))}
        placeholder="6 digits"
        placeholderTextColor="#999"
        keyboardType="number-pad"
        secureTextEntry
        maxLength={6}
        autoComplete="off"
        editable={!submitting}
        onSubmitEditing={handleSubmit}
      />
      {mismatch && <Text style={styles.mismatch}>The two PINs don&apos;t match.</Text>}

      <Pressable
        style={[styles.button, !canSubmit && styles.buttonDisabled]}
        onPress={handleSubmit}
        disabled={!canSubmit}
        accessibilityRole="button"
      >
        {submitting ? (
          <ActivityIndicator color="#fff" />
        ) : (
          <Text style={styles.buttonText}>Save new PIN</Text>
        )}
      </Pressable>

      <View style={styles.footer}>
        <Pressable onPress={handleSignOut}>
          <Text style={styles.linkText}>Sign out</Text>
        </Pressable>
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
    fontSize: 22,
    fontWeight: "600",
    marginTop: 24,
  },
  hint: {
    color: "#666",
    fontSize: 14,
    lineHeight: 20,
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
    fontSize: 18,
  },
  mismatch: {
    color: "#c0392b",
    fontSize: 13,
  },
  button: {
    backgroundColor: "#1d6fd6",
    borderRadius: 8,
    paddingVertical: 12,
    alignItems: "center",
    marginTop: 8,
  },
  buttonDisabled: {
    opacity: 0.4,
  },
  buttonText: {
    color: "#fff",
    fontWeight: "600",
    fontSize: 16,
  },
  footer: {
    marginTop: 12,
    alignItems: "center",
  },
  linkText: {
    color: "#1d6fd6",
    fontWeight: "600",
  },
});
