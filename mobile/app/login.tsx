import { Link } from "expo-router";
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

import { getWorkerId } from "../src/api/client";
import { problemFrom, type Problem } from "../src/api/problems";
import { ProblemBanner } from "../src/components/ProblemBanner";
import { useSession } from "../src/context/SessionContext";

/**
 * Login screen — docs/auth-api.md §4, §8.
 *
 * Worker ID + 6-digit PIN on a number pad. Signing in flips the session to
 * signedIn and the root layout moves on by itself (to Change PIN first when
 * the PIN is still the issued one), so there is nothing to navigate to here.
 * Settings stays reachable because the server URL has to be set before any
 * login can work.
 */
export default function LoginScreen() {
  const { auth, signIn } = useSession();
  const [workerId, setWorkerId] = useState("");
  const [pin, setPin] = useState("");
  const [submitting, setSubmitting] = useState(false);
  const [problem, setProblem] = useState<Problem | null>(null);

  useEffect(() => {
    (async () => {
      const last = await getWorkerId();
      if (last) setWorkerId((current) => current || last);
    })();
  }, []);

  if (auth.kind === "loading") {
    return (
      <View style={styles.center}>
        <ActivityIndicator />
      </View>
    );
  }

  const canSubmit = workerId.trim().length > 0 && pin.length === 6 && !submitting;
  // A wrong try's message replaces the "session ended" note it followed.
  const shown = problem ?? (auth.kind === "signedOut" ? auth.notice : undefined);

  async function handleSignIn() {
    if (!canSubmit) return;
    setSubmitting(true);
    setProblem(null);
    try {
      await signIn(workerId, pin);
    } catch (err) {
      setPin("");
      setProblem(problemFrom(err, "Couldn't sign in. Try again."));
      setSubmitting(false);
    }
  }

  return (
    <ScrollView
      style={styles.screen}
      contentContainerStyle={styles.container}
      keyboardShouldPersistTaps="handled"
    >
      <Text style={styles.title}>Sign in</Text>
      <Text style={styles.hint}>
        Use the worker ID and PIN your supervisor gave you.
      </Text>

      {shown && <ProblemBanner problem={shown} />}

      <Text style={styles.label}>Worker ID</Text>
      <TextInput
        style={styles.input}
        value={workerId}
        onChangeText={setWorkerId}
        placeholder="W-0042"
        placeholderTextColor="#999"
        autoCapitalize="characters"
        autoCorrect={false}
        autoComplete="off"
        editable={!submitting}
      />

      <Text style={styles.label}>PIN</Text>
      <TextInput
        style={styles.input}
        value={pin}
        onChangeText={(text) => setPin(text.replace(/\D/g, "").slice(0, 6))}
        placeholder="6 digits"
        placeholderTextColor="#999"
        keyboardType="number-pad"
        secureTextEntry
        maxLength={6}
        autoCorrect={false}
        autoComplete="off"
        editable={!submitting}
        onSubmitEditing={handleSignIn}
      />

      <Pressable
        style={[styles.button, !canSubmit && styles.buttonDisabled]}
        onPress={handleSignIn}
        disabled={!canSubmit}
        accessibilityRole="button"
      >
        {submitting ? (
          <ActivityIndicator color="#fff" />
        ) : (
          <Text style={styles.buttonText}>Sign in</Text>
        )}
      </Pressable>

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
  center: {
    flex: 1,
    alignItems: "center",
    justifyContent: "center",
  },
  title: {
    fontSize: 24,
    fontWeight: "600",
    marginTop: 24,
  },
  hint: {
    color: "#666",
    fontSize: 14,
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
});
