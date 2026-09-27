/**
 * Local on-device history of past compliance checks — Phase 4.
 *
 * INSTRUCTIONS.md §2: "past checks stored locally on-device (AsyncStorage):
 * timestamp, job, verdict. No backend persistence exists yet — do not
 * invent endpoints for it." The full check result + photo are kept too so
 * the history detail view can render the same breakdown as Result.
 */
import AsyncStorage from "@react-native-async-storage/async-storage";

import type { ComplianceCheckResponse } from "../api/types";

const HISTORY_KEY = "@ppb/history";

/** Oldest entries are dropped past this to keep AsyncStorage bounded. */
const MAX_ENTRIES = 100;

export interface HistoryEntry {
  /** Unique id for routing to the detail screen (timestamp isn't guaranteed unique). */
  id: string;
  result: ComplianceCheckResponse;
  /** Local file URI of the captured photo, best-effort — may go stale across app restarts. */
  imageUri: string;
}

/** Most-recent-first. Returns [] if nothing has been recorded yet or storage is corrupt. */
export async function getHistory(): Promise<HistoryEntry[]> {
  const raw = await AsyncStorage.getItem(HISTORY_KEY);
  if (!raw) return [];
  try {
    return JSON.parse(raw) as HistoryEntry[];
  } catch {
    return [];
  }
}

/** Prepends a new entry and persists, trimming to MAX_ENTRIES. Returns the updated list. */
export async function addHistoryEntry(
  result: ComplianceCheckResponse,
  imageUri: string,
): Promise<HistoryEntry[]> {
  const existing = await getHistory();
  const entry: HistoryEntry = {
    id: `${result.timestamp}-${Math.random().toString(36).slice(2, 8)}`,
    result,
    imageUri,
  };
  const updated = [entry, ...existing].slice(0, MAX_ENTRIES);
  await AsyncStorage.setItem(HISTORY_KEY, JSON.stringify(updated));
  return updated;
}
