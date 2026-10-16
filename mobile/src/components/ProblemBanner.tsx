import { StyleSheet, Text, View } from "react-native";

import type { Problem } from "../api/problems";

/**
 * One message box for every failure. Errors are red; notices ("try again
 * shortly") are amber, so a server that is merely busy never looks like the
 * worker did something wrong.
 */
export function ProblemBanner({ problem }: { problem: Problem }) {
  const notice = problem.tone === "notice";
  return (
    <View
      style={[styles.box, notice ? styles.noticeBox : styles.errorBox]}
      testID={notice ? "problem-notice" : "problem-error"}
    >
      <Text style={[styles.text, notice ? styles.noticeText : styles.errorText]}>
        {problem.message}
      </Text>
    </View>
  );
}

const styles = StyleSheet.create({
  box: {
    borderRadius: 8,
    borderWidth: 1,
    paddingHorizontal: 12,
    paddingVertical: 10,
  },
  errorBox: {
    backgroundColor: "#fdecea",
    borderColor: "#c0392b",
  },
  noticeBox: {
    backgroundColor: "#fff4e0",
    borderColor: "#b9770e",
  },
  text: {
    fontSize: 14,
    fontWeight: "500",
  },
  errorText: {
    color: "#c0392b",
  },
  noticeText: {
    color: "#8a5a0b",
  },
});
