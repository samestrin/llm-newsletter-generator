import { useEffect, useRef, useState } from "react";
import { ApiError, clearCurrentRun, rememberRun, type Api, type ProgressStep, type Run } from "./api";

/** How often the page asks `GET /api/runs/{id}` while the run goes. */
export const POLL_MS = 2000;
/** How long no step may change before the page says it is still working. */
export const QUIET_MS = 30_000;
/** Failed polls in a row before the page stops asking and says so. */
export const MAX_FAILURES = 10;

const STATES: Record<ProgressStep["state"], { text: string; icon: string }> = {
  waiting: { text: "Waiting", icon: "○" },
  running: { text: "Running", icon: "◐" },
  done: { text: "Done", icon: "✓" },
  failed: { text: "Failed", icon: "✕" },
  skipped: { text: "Skipped", icon: "–" },
};

/** `45 s`, `1 min 20 s`, `2 min`, `1 h 2 min`: how long a step has been running. */
export function elapsed(ms: number): string {
  const seconds = Math.max(0, Math.floor(ms / 1000));
  const [h, m, s] = [Math.floor(seconds / 3600), Math.floor(seconds / 60) % 60, seconds % 60];
  if (h > 0) return m > 0 ? `${h} h ${m} min` : `${h} h`;
  if (m > 0) return s > 0 ? `${m} min ${s} s` : `${m} min`;
  return `${s} s`;
}

/** What a poll changed, as a screen reader hears it: `Reading articles: running`. */
function changes(before: Run | null, after: Run): string {
  if (!before) return "";
  const was = new Map((before.progress ?? []).map((step) => [step.id, step.state]));
  return (after.progress ?? [])
    .filter((step) => was.has(step.id) && was.get(step.id) !== step.state)
    .map((step) => `${step.label}: ${STATES[step.state].text.toLowerCase()}`)
    .join(". ");
}

/** What counts as a change for "Still working": any step's state, counts or detail. */
const signature = (run: Run) =>
  JSON.stringify([run.status, (run.progress ?? []).map((s) => [s.state, s.done, s.total, s.detail])]);

/**
 * The feeds step's detail, `3 of 4 loaded, 1 failed: <url> <reason>; <url> <reason>`, with
 * each failed feed on its own line. The server builds it from URLs and fetch reasons only.
 */
function Detail({ step }: { step: ProgressStep }) {
  const failed = step.id === "feeds" ? /^(.*?, \d+ failed):\s*(.+)$/.exec(step.detail ?? "") : null;
  if (!failed) return step.detail ? <p className="step-detail">{step.detail}</p> : null;
  return (
    <div className="step-detail">
      <p>{failed[1]}:</p>
      <ul aria-label="Feeds that failed">
        {failed[2].split("; ").map((feed, i) => (
          <li key={i}>{feed}</li>
        ))}
      </ul>
    </div>
  );
}

function Step({ step, now }: { step: ProgressStep; now: number }) {
  const { text, icon } = STATES[step.state];
  const counted = step.done !== null && step.total !== null && step.total > 0;
  return (
    <li className={`step step-${step.state}`}>
      <span className="step-icon" aria-hidden="true">
        {icon}
      </span>
      <span className="step-label">{step.label}</span>
      <span className="step-state">{text}</span>
      {step.state === "running" && step.started_at && (
        <span className="step-time">{elapsed(now - Date.parse(step.started_at))}</span>
      )}
      {counted && (
        <progress value={step.done!} max={step.total!} aria-label={`${step.label}: ${step.done} of ${step.total}`} />
      )}
      {step.detail ? <Detail step={step} /> : counted && <p className="step-detail">{`${step.done} of ${step.total}`}</p>}
    </li>
  );
}

type Props = {
  api: Api;
  runId: string;
  /** The run made its draft: the page opens it. */
  onDone: (draftId: string) => void;
  /** Back to the form; the stored run id is cleared already. */
  onStartOver: () => void;
};

/**
 * A run's live progress: polls `GET /api/runs/{id}` every 2 s until it is done or failed,
 * or the page leaves. The stored run id (see `saveCurrentRun`) is cleared when the run is
 * over, missing, or the visitor starts over; a lost connection keeps it, so a reload resumes.
 */
export function Progress({ api, runId, onDone, onStartOver }: Props) {
  const [run, setRun] = useState<Run | null>(null);
  const [failures, setFailures] = useState(0);
  const [missing, setMissing] = useState(false);
  // The server refused the poll with a 4xx other than 404: its status and plain message.
  const [refused, setRefused] = useState<{ status: number; message: string } | null>(null);
  const [attempt, setAttempt] = useState(0);
  const [now, setNow] = useState(() => Date.now());
  const [changedAt, setChangedAt] = useState(() => Date.now());
  const [announcement, setAnnouncement] = useState("");
  // The latest key and callback, without restarting the polling when they change.
  const latest = useRef({ api, onDone });
  latest.current = { api, onDone };
  const last = useRef<Run | null>(null);
  // The key a 401 refused: polling resumes once the page has another.
  const refusedApi = useRef<Api | null>(null);

  const over =
    run?.status === "done" || run?.status === "error" || missing || !!refused || failures >= MAX_FAILURES;

  useEffect(() => {
    let live = true;
    let timer: ReturnType<typeof setTimeout> | undefined;
    let failed = 0;

    function seen(found: Run) {
      const before = last.current;
      last.current = found;
      setRun(found);
      if (!before || signature(before) !== signature(found)) setChangedAt(Date.now());
      const said = changes(before, found);
      if (said) setAnnouncement(said);
    }

    async function poll() {
      timer = undefined;
      try {
        const found = await latest.current.api.getRun(runId);
        if (!live) return;
        failed = 0;
        setFailures(0);
        seen(found);
        if (found.status === "done" || found.status === "error") {
          clearCurrentRun();
          if (found.status === "done") rememberRun(found);
          if (found.draft_id) latest.current.onDone(found.draft_id);
          return;
        }
      } catch (e) {
        if (!live) return;
        if (e instanceof ApiError && e.status === 404) {
          clearCurrentRun();
          setMissing(true);
          return;
        }
        // A refusal is not a lost connection: asking again gets the same answer, and each
        // 401 would reopen the key panel. Stop and say so; a new key or Try again resumes.
        if (e instanceof ApiError && e.status >= 400 && e.status < 500 && e.status !== 408 && e.status !== 429) {
          if (e.status === 401) refusedApi.current = latest.current.api;
          setRefused({ status: e.status, message: e.message });
          return;
        }
        failed += 1;
        setFailures(failed);
        if (failed >= MAX_FAILURES) return;
      }
      timer = setTimeout(poll, POLL_MS);
    }

    void poll();
    return () => {
      live = false;
      clearTimeout(timer);
    };
  }, [runId, attempt]);

  // The key was saved again after a 401: ask again with it.
  useEffect(() => {
    if (refused?.status !== 401 || api === refusedApi.current) return;
    setRefused(null);
    setAttempt((n) => n + 1);
  }, [api, refused]);

  // A clock for the elapsed times and "Still working", only while the run goes.
  useEffect(() => {
    if (over) return;
    const tick = setInterval(() => setNow(Date.now()), 1000);
    return () => clearInterval(tick);
  }, [over]);

  function startOver() {
    clearCurrentRun();
    onStartOver();
  }

  function tryAgain() {
    setFailures(0);
    setRefused(null);
    setChangedAt(Date.now());
    setAttempt((n) => n + 1);
  }

  const steps = run?.progress ?? [];
  const running = steps.find((step) => step.state === "running");
  const failedRun = run?.status === "error" || (run?.status === "done" && !run.draft_id);
  const quiet = run && !over && failures === 0 && now - changedAt >= QUIET_MS;

  return (
    <section className="progress" aria-labelledby="progress-heading">
      <h2 id="progress-heading">Making your newsletter…</h2>
      <p className="sr-only" aria-live="polite">
        {announcement}
      </p>
      {!run && !missing && !refused && failures === 0 && (
        <p role="status" className="state">
          Asking how the run is going…
        </p>
      )}
      {run && steps.length === 0 && !over && (
        <p className="state">The run is {run.status}.</p>
      )}
      {steps.length > 0 && (
        <ol className="steps" aria-label="Steps">
          {steps.map((step) => (
            <Step key={step.id} step={step} now={now} />
          ))}
        </ol>
      )}
      {quiet && (
        <p role="status" className="state">
          {running ? `Still working on ${running.label}…` : "Still working…"}
        </p>
      )}
      {failures > 0 && failures < MAX_FAILURES && (
        <p role="status" className="state">
          Lost contact, retrying…
        </p>
      )}
      {failures >= MAX_FAILURES ? (
        <div role="alert" className="state state-error">
          <p>
            Lost contact with the server after {MAX_FAILURES} tries. Your newsletter may still be on its way.
          </p>
          <button type="button" onClick={tryAgain}>
            Try again
          </button>
          <button type="button" onClick={startOver}>
            Start over
          </button>
        </div>
      ) : refused ? (
        <div role="alert" className="state state-error">
          {refused.status === 401 ? (
            <p>
              The server refused the key: {refused.message}. Your newsletter may still be on its way: give the page
              the key again and it carries on.
            </p>
          ) : (
            <>
              <p>The server refused to say how the run is going: {refused.message}.</p>
              <button type="button" onClick={tryAgain}>
                Try again
              </button>
            </>
          )}
          <button type="button" onClick={startOver}>
            Start over
          </button>
        </div>
      ) : missing ? (
        <div role="alert" className="state state-error">
          <p>This run could not be found.</p>
          <button type="button" onClick={startOver}>
            Start over
          </button>
        </div>
      ) : (
        failedRun && (
          <div role="alert" className="state state-error">
            <p>
              The run stopped:{" "}
              {run.editor_message || run.error || (run.status === "done" ? "it made no draft" : "no reason was given")}
            </p>
            <button type="button" onClick={startOver}>
              Start over
            </button>
          </div>
        )
      )}
    </section>
  );
}
