import type { Problem } from "../api/problems";

/**
 * One message box for every failure. Errors are red; notices ("try again
 * shortly") are amber, so a busy server never looks like the user did
 * something wrong.
 */
export function ProblemBanner({ problem }: { problem: Problem }) {
  return (
    <div
      role={problem.tone === "error" ? "alert" : "status"}
      className={`banner banner-${problem.tone}`}
      data-testid={`problem-${problem.tone}`}
    >
      {problem.message}
    </div>
  );
}
