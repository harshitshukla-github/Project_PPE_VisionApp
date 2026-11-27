import { useRouter } from "expo-router";
import { useState } from "react";
import { Pressable, ScrollView, StyleSheet, Text, View } from "react-native";

import { problemFrom, type Problem } from "../src/api/problems";
import { PhotoNoticeView } from "../src/components/PhotoNoticeView";
import { noticeFrom, useSession } from "../src/context/SessionContext";

/**
 * The photo notice after sign-in (docs/checks-api.md §5).
 *
 * The root layout shows it while the signed-in worker hasn't answered the
 * current version: never answered, or a newer one since. They answer the
 * training question (off until they say yes) and the app sends that answer
 * for the version they were shown. If the notice changed meanwhile (409
 * notice_changed), the new wording is shown and they're asked again. Once the
 * server has the answer, the layout opens the app.
 */
export default function PhotoNoticeScreen() {
  const router = useRouter();
  const { consent, answerNotice, signOut } = useSession();
  const [training, setTraining] = useState(false);
  const [submitting, setSubmitting] = useState(false);
  const [problem, setProblem] = useState<Problem | null>(null);

  if (!consent) return null; // the layout only shows this screen once the notice has loaded
  const notice = consent.notice;

  async function handleSubmit() {
    setSubmitting(true);
    setProblem(null);
    try {
      await answerNotice(notice.version, training);
      // The root layout now sees the notice answered and opens the app.
    } catch (err) {
      if (noticeFrom(err)) {
        // The session already holds the new wording; ask again from scratch.
        setTraining(false);
        setProblem({
          message: "The notice changed while you were reading it. Please read it again and answer.",
          tone: "notice",
        });
      } else {
        setProblem(problemFrom(err, "Couldn't save your answer. Try again."));
      }
      setSubmitting(false);
    }
  }

  async function handleSignOut() {
    await signOut();
    router.replace("/login");
  }

  return (
    <ScrollView style={styles.screen} contentContainerStyle={styles.container}>
      <Text style={styles.title}>Your photos</Text>
      <PhotoNoticeView
        notice={notice}
        training={training}
        onTrainingChange={setTraining}
        onSubmit={() => void handleSubmit()}
        submitting={submitting}
        problem={problem}
        submitLabel="Continue"
      >
        <View style={styles.footer}>
          <Pressable onPress={handleSignOut} accessibilityRole="button">
            <Text style={styles.linkText}>Sign out</Text>
          </Pressable>
        </View>
      </PhotoNoticeView>
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
  footer: {
    alignItems: "center",
  },
  linkText: {
    color: "#1d6fd6",
    fontWeight: "600",
  },
});
