/** "general_labor" -> "General Labor" — INSTRUCTIONS.md §4: title-case the job key for the label. */
export function titleCase(key: string): string {
  return key
    .split("_")
    .map((word) => word.charAt(0).toUpperCase() + word.slice(1))
    .join(" ");
}
