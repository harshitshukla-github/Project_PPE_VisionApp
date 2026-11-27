import {
  createContext,
  useContext,
  useEffect,
  useMemo,
  useRef,
  useState,
  type ReactNode,
} from "react";

import {
  ApiError,
  answerConsent,
  changeWorkerPin,
  clearToken,
  getConsent,
  getJobs,
  getToken,
  getWorkerMe,
  setToken,
  setWorkerId as rememberWorkerId,
  workerLogin,
  workerLogout,
} from "../api/client";
import { problemFrom, type Problem } from "../api/problems";
import type {
  ComplianceCheckResponse,
  ConsentState,
  JobsResponse,
  PhotoNotice,
  WorkerInfo,
} from "../api/types";
import { titleCase } from "../format";
import { addHistoryEntry, getHistory, type HistoryEntry } from "../storage/history";

export type JobsStatus =
  | { kind: "loading" }
  | { kind: "loaded"; jobs: JobsResponse }
  | { kind: "error"; message: string; tone: Problem["tone"] };

/**
 * Who is using the app. `loading` lasts only until the saved token has been
 * checked with the server; routing treats it like signedOut so the Login
 * screen (showing a spinner) is what's on screen meanwhile.
 */
export type AuthState =
  | { kind: "loading" }
  | { kind: "signedOut"; notice?: Problem }
  | { kind: "signedIn"; worker: WorkerInfo; mustChangePin: boolean };

interface SessionState {
  auth: AuthState;
  /** Signed-in worker's ID, or "" when signed out. */
  workerId: string;
  /** Resolves on success; rejects with the ApiError so the Login screen can show why. */
  signIn: (workerId: string, pin: string) => Promise<void>;
  /** Replaces the issued PIN. Rejects with the ApiError on a weak or wrong PIN. */
  changePin: (currentPin: string, newPin: string) => Promise<void>;
  /** POST /auth/worker/logout, then forget the token whatever the response. */
  signOut: () => Promise<void>;
  /** The server said our token is dead (401 not_authenticated): back to Login with a note. */
  expireSession: () => void;
  /** The server said 403 pin_change_required: open the Change PIN screen. */
  requirePinChange: () => void;

  /** GET /jobs result, shared so Capture doesn't need to re-fetch. */
  jobsStatus: JobsStatus;
  reloadJobs: () => void;
  /**
   * Re-reads GET /jobs without flicker and returns the list, or null if the
   * server couldn't answer. Admins can deactivate jobs while the app is open
   * (docs/admin-api.md §2), so this runs before every check. If the selected
   * job has gone, the selection is cleared and `jobNotice` says so: it is
   * never swapped for another job, because a check against the wrong job
   * would give a wrong verdict.
   */
  refreshJobs: () => Promise<JobsResponse | null>;
  /** Selected job_category key (a key of jobsStatus.jobs when loaded). */
  jobCategory: string | null;
  setJobCategory: (key: string) => void;
  /** Why the selection was cleared ("Welder isn't available any more…"); gone once a job is picked. */
  jobNotice: Problem | null;

  /** Most recent POST /compliance/check result, consumed by the Result screen. */
  lastCheck: ComplianceCheckResponse | null;
  lastImageUri: string | null;
  recordCheck: (result: ComplianceCheckResponse, imageUri: string) => void;

  /**
   * The signed-in worker's past checks, most-recent-first — local on-device
   * log (INSTRUCTIONS.md §2). Other workers' checks stay on the phone but are
   * never shown (docs/auth-api.md §10, open question 2).
   */
  history: HistoryEntry[];
  /** False until the initial AsyncStorage read completes — lets History tell "loading" from "genuinely empty". */
  historyLoaded: boolean;

  /**
   * The photo notice and this worker's answer, as the server last said
   * (docs/checks-api.md §5). Asked after sign-in and the PIN change; null
   * until then, or when the server couldn't say.
   */
  consent: ConsentState | null;
  /**
   * The worker has a notice to answer: never answered, or a newer version
   * since. The root layout shows the notice while this is true. If the
   * server can't say, it stays false: a check is never refused for consent,
   * and photos just aren't kept until the worker answers.
   */
  noticeNeeded: boolean;
  /** GET the notice and answer again; resolves to it, or null if the server couldn't say. */
  refreshConsent: () => Promise<ConsentState | null>;
  /**
   * PUT the worker's answer to notice `version`. Rejects with the ApiError; on
   * 409 notice_changed the newer notice is already in `consent` to show.
   */
  answerNotice: (version: number, training: boolean) => Promise<ConsentState>;
}

/** The notice a 409 notice_changed carries, if it has one (docs/checks-api.md §5). */
export function noticeFrom(err: unknown): PhotoNotice | null {
  if (!(err instanceof ApiError) || err.code !== "notice_changed") return null;
  const notice = err.body?.notice as Partial<PhotoNotice> | undefined;
  return notice && typeof notice.version === "number" && typeof notice.text === "string"
    ? (notice as PhotoNotice)
    : null;
}

const SessionContext = createContext<SessionState | undefined>(undefined);

const SESSION_ENDED: Problem = { message: "Your session ended. Sign in again.", tone: "error" };

/**
 * App-wide session state — per INSTRUCTIONS.md §3 ("React hooks + context;
 * no Redux"). Holds who is signed in, the job selected on Home so Capture
 * and Result can read it without prop-drilling through the tab navigator,
 * and the last check result so Result can render it after Capture
 * navigates there.
 *
 * @param initialAuth Starts in a known sign-in state and skips the saved-token
 *   check — for tests, where there is no server to ask.
 */
export function SessionProvider({
  children,
  initialAuth,
}: {
  children: ReactNode;
  initialAuth?: AuthState;
}) {
  const [auth, setAuth] = useState<AuthState>(initialAuth ?? { kind: "loading" });
  const [jobsStatus, setJobsStatus] = useState<JobsStatus>({ kind: "loading" });
  const [jobCategory, setJobCategory] = useState<string | null>(null);
  const [jobNotice, setJobNotice] = useState<Problem | null>(null);
  // The latest selection, readable from async code that started before it changed.
  const jobCategoryRef = useRef<string | null>(null);
  jobCategoryRef.current = jobCategory;
  const [lastCheck, setLastCheck] = useState<ComplianceCheckResponse | null>(null);
  const [lastImageUri, setLastImageUri] = useState<string | null>(null);
  const [allHistory, setAllHistory] = useState<HistoryEntry[]>([]);
  const [historyLoaded, setHistoryLoaded] = useState(false);
  const [consent, setConsent] = useState<ConsentState | null>(null);

  // The consent belongs to one worker, once their own PIN is chosen (the server refuses before).
  const consentFor =
    auth.kind === "signedIn" && !auth.mustChangePin ? auth.worker.worker_id : null;
  // The worker the latest consent request is for: an answer for anyone else is dropped.
  const consentForRef = useRef<string | null>(null);
  consentForRef.current = consentFor;

  useEffect(() => {
    if (!initialAuth) void restoreSession();
    reloadJobs();
    (async () => {
      setAllHistory(await getHistory());
      setHistoryLoaded(true);
    })();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  // After sign-in and the PIN change (and on start-up with a saved session): is a photo notice owed?
  // Declared after the effect above, so the job list is still asked for first.
  useEffect(() => {
    setConsent(null);
    if (consentFor) void refreshConsent();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [consentFor]);

  /** Is the saved token still good? A reachable server decides; an unreachable one just asks for a fresh sign-in. */
  async function restoreSession() {
    const token = await getToken();
    if (!token) {
      setAuth({ kind: "signedOut" });
      return;
    }
    try {
      const me = await getWorkerMe();
      setAuth({ kind: "signedIn", worker: me.worker, mustChangePin: me.must_change_pin });
    } catch (err) {
      if (err instanceof ApiError && err.status === 401) {
        await clearToken();
        setAuth({ kind: "signedOut", notice: SESSION_ENDED });
      } else {
        // Server down or unreachable: keep the token, but we can't prove who this is.
        setAuth({
          kind: "signedOut",
          notice: problemFrom(err, "Couldn't check your sign-in. Try again."),
        });
      }
    }
  }

  async function reloadJobs() {
    setJobsStatus({ kind: "loading" });
    try {
      const jobs = await getJobs();
      setJobsStatus({ kind: "loaded", jobs });
      setJobCategory((current) =>
        current && jobs[current] ? current : Object.keys(jobs)[0] ?? null,
      );
    } catch (err) {
      const { message, tone } = problemFrom(err, "Unexpected error loading jobs.");
      setJobsStatus({ kind: "error", message, tone });
    }
  }

  async function refreshJobs(): Promise<JobsResponse | null> {
    let jobs: JobsResponse;
    try {
      jobs = await getJobs();
    } catch {
      // Offline or the server is busy: the check itself will say so.
      return null;
    }
    setJobsStatus({ kind: "loaded", jobs });
    const selected = jobCategoryRef.current;
    if (selected && !jobs[selected]) {
      setJobCategory(null);
      setJobNotice({
        message: `${titleCase(selected)} isn't available any more. Pick your job again.`,
        tone: "notice",
      });
    }
    return jobs;
  }

  function selectJob(key: string) {
    setJobCategory(key);
    setJobNotice(null);
  }

  async function signIn(workerId: string, pin: string) {
    const res = await workerLogin(workerId, pin);
    await setToken(res.token);
    void rememberWorkerId(res.worker.worker_id);
    setAuth({ kind: "signedIn", worker: res.worker, mustChangePin: res.must_change_pin });
  }

  async function changePin(currentPin: string, newPin: string) {
    await changeWorkerPin(currentPin, newPin);
    setAuth((current) =>
      current.kind === "signedIn" ? { ...current, mustChangePin: false } : current,
    );
  }

  async function signOut() {
    await workerLogout();
    setLastCheck(null);
    setLastImageUri(null);
    // One worker's "that job isn't available" note isn't the next worker's.
    setJobNotice(null);
    setAuth({ kind: "signedOut" });
  }

  function expireSession() {
    void clearToken();
    setLastCheck(null);
    setLastImageUri(null);
    setJobNotice(null);
    setAuth({ kind: "signedOut", notice: SESSION_ENDED });
  }

  function requirePinChange() {
    setAuth((current) =>
      current.kind === "signedIn" ? { ...current, mustChangePin: true } : current,
    );
  }

  async function refreshConsent(): Promise<ConsentState | null> {
    const asked = consentForRef.current;
    try {
      const answer = await getConsent();
      if (consentForRef.current !== asked) return null; // signed out, or someone else, meanwhile
      setConsent(answer);
      return answer;
    } catch (err) {
      if (err instanceof ApiError && err.code === "not_authenticated") expireSession();
      else if (err instanceof ApiError && err.code === "pin_change_required") requirePinChange();
      // Anything else (offline, a server without this endpoint): no notice now. Checks still work.
      return null;
    }
  }

  async function answerNotice(version: number, training: boolean): Promise<ConsentState> {
    try {
      const answer = await answerConsent(version, training);
      setConsent(answer);
      return answer;
    } catch (err) {
      const newer = noticeFrom(err);
      if (newer) setConsent((current) => (current ? { ...current, notice: newer } : current));
      if (err instanceof ApiError && err.code === "not_authenticated") expireSession();
      else if (err instanceof ApiError && err.code === "pin_change_required") requirePinChange();
      throw err;
    }
  }

  const noticeNeeded =
    consentFor !== null &&
    consent !== null &&
    (consent.seen_version === null || consent.seen_version < consent.notice.version);

  function recordCheck(result: ComplianceCheckResponse, imageUri: string) {
    setLastCheck(result);
    setLastImageUri(imageUri);
    void addHistoryEntry(result, imageUri).then(setAllHistory);
  }

  const workerId = auth.kind === "signedIn" ? auth.worker.worker_id : "";

  const history = useMemo(
    () =>
      workerId
        ? allHistory.filter(
            (entry) => entry.result.worker_id.toUpperCase() === workerId.toUpperCase(),
          )
        : [],
    [allHistory, workerId],
  );

  return (
    <SessionContext.Provider
      value={{
        auth,
        workerId,
        signIn,
        changePin,
        signOut,
        expireSession,
        requirePinChange,
        jobsStatus,
        reloadJobs,
        refreshJobs,
        jobCategory,
        setJobCategory: selectJob,
        jobNotice,
        lastCheck,
        lastImageUri,
        recordCheck,
        history,
        historyLoaded,
        consent,
        noticeNeeded,
        refreshConsent,
        answerNotice,
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
