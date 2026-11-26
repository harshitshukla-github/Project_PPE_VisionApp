import type { ReactNode } from "react";
import { ActivityIndicator, Pressable, StyleSheet, Switch, Text, View } from "react-native";

import type { Problem } from "../api/problems";
import type { PhotoNotice } from "../api/types";
import { ProblemBanner } from "./ProblemBanner";

/**
 * The photo notice in the server's own words (docs/checks-api.md §5): `text`,
 * then `training_question` with a switch. The app writes none of the wording,
 * so a change of wording or of a retention period needs no app release.
 *
 * Used by the notice screen after sign-in and by Settings. The switch is
 * whatever the caller says; both start it off until the worker says yes.
 *
 * The wording must pass the company's privacy check before this ships in an
 * app release (checks-api.md §10, point 1).
 */
export function PhotoNoticeView({
  notice,
  training,
  onTrainingChange,
  onSubmit,
  submitting,
  problem,
  submitLabel,
  children,
}: {
  notice: PhotoNotice;
  training: boolean;
  onTrainingChange: (value: boolean) => void;
  onSubmit: () => void;
  submitting: boolean;
  problem: Problem | null;
  submitLabel: string;
  /** Anything to show under the button, such as a Sign out link. */
  children?: ReactNode;
}) {
  return (
    <View style={styles.notice} testID="photo-notice">
      {problem && <ProblemBanner problem={problem} />}

      <Text style={styles.text}>{notice.text}</Text>

      <View style={styles.question}>
        <Text style={styles.questionText}>{notice.training_question}</Text>
        <View style={styles.choice}>
          <Text style={styles.choiceText}>{training ? "Yes" : "No"}</Text>
          <Switch
            value={training}
            onValueChange={onTrainingChange}
            disabled={submitting}
            accessibilityLabel="Keep my photos to improve the PPE detector"
            testID="training-switch"
          />
        </View>
      </View>

      <Pressable
        style={[styles.button, submitting && styles.buttonDisabled]}
        onPress={onSubmit}
        disabled={submitting}
        accessibilityRole="button"
        accessibilityLabel={submitLabel}
      >
        {submitting ? <ActivityIndicator color="#fff" /> : <Text style={styles.buttonText}>{submitLabel}</Text>}
      </Pressable>

      {children}
    </View>
  );
}

const styles = StyleSheet.create({
  notice: {
    gap: 16,
  },
  text: {
    fontSize: 16,
    lineHeight: 24,
  },
  question: {
    padding: 14,
    gap: 10,
    borderWidth: 1,
    borderColor: "#ccc",
    borderRadius: 8,
  },
  questionText: {
    fontSize: 15,
    lineHeight: 22,
    fontWeight: "500",
  },
  choice: {
    flexDirection: "row",
    alignItems: "center",
    justifyContent: "space-between",
  },
  choiceText: {
    fontSize: 15,
    fontWeight: "600",
  },
  button: {
    backgroundColor: "#1d6fd6",
    borderRadius: 8,
    paddingVertical: 12,
    alignItems: "center",
  },
  buttonDisabled: {
    opacity: 0.4,
  },
  buttonText: {
    color: "#fff",
    fontWeight: "600",
    fontSize: 16,
  },
});
