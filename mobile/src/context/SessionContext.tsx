import {
  createContext,
  useContext,
  useEffect,
  useState,
  type ReactNode,
} from "react";

import {
  ApiError,
  getJobs,
  getWorkerId,
  setWorkerId as persistWorkerId,
} from "../api/client";
import type { ComplianceCheckResponse, JobsResponse } from "../api/types";
import { addHistoryEntry, getHistory, type HistoryEntry } from "../storage/history";

export type JobsStatus =
  | { kind: "loading" }
  | { kind: "loaded"; jobs: JobsResponse }
  | { kind: "error"; message: string };

interface SessionState {
  /** Worker ID entered on Home, remembered across launches (AsyncStorage). */
  workerId: string;
  setWorkerId: (id: string) => void;

  /** GET /jobs result, shared so Capture doesn't need to re-fetch. */
  jobsStatus: JobsStatus;
  reloadJobs: () => void;
  /** Selected job_category key (a key of jobsStatus.jobs when loaded). */
  jobCategory: string | null;
  setJobCategory: (key: string) => void;

  /** Most recent POST /compliance/check result, consumed by the Result screen. */
  lastCheck: ComplianceCheckResponse | null;
  lastImageUri: string | null;
  recordCheck: (result: ComplianceCheckResponse, imageUri: string) => void;

  /** Past checks, most-recent-first — local on-device log (INSTRUCTIONS.md §2). */
  history: HistoryEntry[];
}

const SessionContext = createContext<SessionState | undefined>(undefined);

/**
 * App-wide session state — per INSTRUCTIONS.md §3 ("React hooks + context;
 * no Redux"). Holds the worker identity + job selected on Home so Capture
 * and Result can read them without prop-drilling through the tab navigator,
 * and holds the last check result so Result can render it after Capture
 * navigates there.
 */
export function SessionProvider({ children }: { children: ReactNode }) {
  const [workerId, setWorkerIdState] = useState("");
  const [jobsStatus, setJobsStatus] = useState<JobsStatus>({ kind: "loading" });
  const [jobCategory, setJobCategory] = useState<string | null>(null);
  const [lastCheck, setLastCheck] = useState<ComplianceCheckResponse | null>(null);
  const [lastImageUri, setLastImageUri] = useState<string | null>(null);
  const [history, setHistory] = useState<HistoryEntry[]>([]);

  useEffect(() => {
    (async () => {
      const saved = await getWorkerId();
      if (saved) setWorkerIdState(saved);
    })();
    reloadJobs();
    (async () => {
      setHistory(await getHistory());
    })();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  async function reloadJobs() {
    setJobsStatus({ kind: "loading" });
    try {
      const jobs = await getJobs();
      setJobsStatus({ kind: "loaded", jobs });
      setJobCategory((current) =>
        current && jobs[current] ? current : Object.keys(jobs)[0] ?? null,
      );
    } catch (err) {
      const message =
        err instanceof ApiError ? err.message : "Unexpected error loading jobs.";
      setJobsStatus({ kind: "error", message });
    }
  }

  function setWorkerId(id: string) {
    setWorkerIdState(id);
    // Fire-and-forget persist; the id itself is the source of truth for the UI.
    void persistWorkerId(id);
  }

  function recordCheck(result: ComplianceCheckResponse, imageUri: string) {
    setLastCheck(result);
    setLastImageUri(imageUri);
    void addHistoryEntry(result, imageUri).then(setHistory);
  }

  return (
    <SessionContext.Provider
      value={{
        workerId,
        setWorkerId,
        jobsStatus,
        reloadJobs,
        jobCategory,
        setJobCategory,
        lastCheck,
        lastImageUri,
        recordCheck,
        history,
      }}
    >
      {children}
    </SessionContext.Provider>
  );
}

export function useSession(): SessionState {
  const ctx = useContext(SessionContext);
  if (!ctx) {
    throw new Error("useSession must be used within a SessionProvider");
  }
  return ctx;
}
