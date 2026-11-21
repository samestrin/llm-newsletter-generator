import { useEffect, useId, useRef, useState, type FormEvent } from "react";
import { ApiError, type Api, type Preset, type Run } from "./api";

/** The most feeds a demo-mode run takes (plan 08.1 T3 caps `RunBody.feeds` at 6). */
export const MAX_FEEDS = 6;
export const MAX_FOCUS = 300;

/** Where a row's address stands: not checked, being checked, or the check's answer. */
type Check =
  | { state: "idle" }
  | { state: "checking" }
  | { state: "ok"; items: number }
  | { state: "error"; message: string };

type Row = { key: number; url: string; check: Check };

/** The run start: nothing yet, checking and posting, a busy server with its wait, or a failure. */
type Start =
  | { state: "idle" }
  | { state: "starting" }
  | { state: "busy"; wait: number }
  | { state: "error"; message: string };

const BUSY = "Someone else is making one right now. Try again in a minute.";

const message = (e: unknown) => (e instanceof Error ? e.message : String(e));

/**
 * The form a visitor fills to make an issue: feed rows (each checked with
 * `POST /api/feeds/check` when the visitor leaves it and again before the run), presets
 * that fill the rows, an optional focus, and "See an example" off the main path.
 */
export function StartForm({
  api,
  onStarted,
  onExample,
}: {
  api: Api;
  onStarted: (run: Run) => void;
  /** Opens the seeded draft; no button when there is none. */
  onExample?: () => void;
}) {
  const id = useId();
  const next = useRef(1);
  const row = (url = ""): Row => ({ key: next.current++, url, check: { state: "idle" } });
  const [rows, setRows] = useState<Row[]>(() => [row()]);
  const [presets, setPresets] = useState<Preset[]>([]);
  const [focus, setFocus] = useState("");
  const [start, setStart] = useState<Start>({ state: "idle" });

  useEffect(() => {
    let live = true;
    api.presets().then(
      (found) => live && setPresets(Array.isArray(found) ? found : []),
      // No presets is fine: they are a shortcut, the rows still work.
      () => undefined,
    );
    return () => {
      live = false;
    };
  }, [api]);

  // Count the busy server's wait down a second at a time; the retry opens at zero.
  const waiting = start.state === "busy" && start.wait > 0;
  useEffect(() => {
    if (!waiting) return;
    const timer = setInterval(
      () => setStart((s) => (s.state === "busy" && s.wait > 0 ? { state: "busy", wait: s.wait - 1 } : s)),
      1000,
    );
    return () => clearInterval(timer);
  }, [waiting]);

  /** Set a row's check, only while it still holds the address that was checked. */
  const settle = (key: number, url: string, check: Check) =>
    setRows((all) => all.map((r) => (r.key === key && r.url.trim() === url ? { ...r, check } : r)));

  async function check(key: number, raw: string): Promise<boolean> {
    const url = raw.trim();
    if (!url) return false;
    settle(key, url, { state: "checking" });
    try {
      const found = await api.checkFeed(url);
      settle(
        key,
        url,
        found.ok
          ? { state: "ok", items: found.items ?? 0 }
          : { state: "error", message: found.error || "This feed could not be checked" },
      );
      return found.ok;
    } catch (e) {
      settle(key, url, { state: "error", message: message(e) });
      return false;
    }
  }

  function edit(key: number, url: string) {
    setRows((all) => all.map((r) => (r.key === key ? { key, url, check: { state: "idle" } } : r)));
  }

  function leave(r: Row) {
    if (r.check.state === "idle") void check(r.key, r.url);
  }

  function remove(key: number) {
    setRows((all) => {
      const left = all.filter((r) => r.key !== key);
      return left.length > 0 ? left : [row()];
    });
  }

  function fillFrom(preset: Preset) {
    const filled = preset.feeds.slice(0, MAX_FEEDS).map((url) => row(url));
    if (filled.length === 0) return;
    setRows(filled);
    for (const r of filled) void check(r.key, r.url);
  }

  async function submit(event?: FormEvent) {
    event?.preventDefault();
    setStart({ state: "starting" });
    const asked = rows.filter((r) => r.url.trim());
    const results = await Promise.all(asked.map((r) => check(r.key, r.url)));
    const feeds = asked.filter((_, n) => results[n]).map((r) => r.url.trim());
    if (feeds.length === 0) {
      setStart({ state: "idle" });
      return;
    }
    const wanted = focus.trim();
    try {
      const run = await api.startRun(wanted ? { feeds, focus: wanted } : { feeds });
      setStart({ state: "idle" });
      onStarted(run);
    } catch (e) {
      setStart(
        e instanceof ApiError && e.status === 409
          ? { state: "busy", wait: Math.ceil(e.retryAfter ?? 60) }
          : { state: "error", message: message(e) },
      );
    }
  }

  const ready = rows.some((r) => r.check.state === "ok");
  const starting = start.state === "starting";

  return (
    <form className="start" onSubmit={submit} aria-labelledby={`${id}-heading`}>
      <h2 id={`${id}-heading`}>Make a newsletter</h2>
      <p className="intro">
        Give it a few news feeds and it writes an issue from them. Every claim links to its source and is checked
        against it before you see it, and nothing is sent: you read, edit and approve the draft yourself.
      </p>

      {presets.length > 0 && (
        <div role="group" aria-label="Start from a preset" className="presets">
          <span className="presets-label" aria-hidden="true">
            Start from a preset:
          </span>
          {presets.map((p) => (
            <button key={p.id} type="button" onClick={() => fillFrom(p)} disabled={starting}>
              {p.label}
            </button>
          ))}
        </div>
      )}

      <ol className="feed-rows">
        {rows.map((r, n) => {
          const label = `Feed ${n + 1}`;
          const status = `${id}-status-${r.key}`;
          return (
            <li key={r.key}>
              <div role="group" aria-label={label} className="feed-row">
                <label>
                  <span>{label} address</span>
                  <input
                    type="url"
                    inputMode="url"
                    autoComplete="off"
                    placeholder="https://example.com/feed.xml"
                    value={r.url}
                    onChange={(e) => edit(r.key, e.target.value)}
                    onBlur={() => leave(r)}
                    aria-describedby={status}
                    aria-invalid={r.check.state === "error" || undefined}
                    disabled={starting}
                  />
                </label>
                <button
                  type="button"
                  onClick={() => remove(r.key)}
                  disabled={starting}
                  aria-label={`Remove feed ${n + 1}`}
                >
                  Remove
                </button>
                <p id={status} className={`feed-check feed-check-${r.check.state}`} aria-live="polite">
                  {r.check.state === "checking" ? (
                    <>
                      <span aria-hidden="true">…</span> Checking…
                    </>
                  ) : r.check.state === "ok" ? (
                    <>
                      <span aria-hidden="true">✓</span> OK, {r.check.items} item{r.check.items === 1 ? "" : "s"}
                    </>
                  ) : r.check.state === "error" ? (
                    <>
                      <span aria-hidden="true">✗</span> {r.check.message}
                    </>
                  ) : null}
                </p>
              </div>
            </li>
          );
        })}
      </ol>
      <button
        type="button"
        onClick={() => setRows((all) => [...all, row()])}
        disabled={rows.length >= MAX_FEEDS || starting}
      >
        Add a feed
      </button>

      <label className="focus">
        <span>Focus (optional)</span>
        <span className="hint" id={`${id}-focus-hint`}>
          What should this issue be about or emphasize? e.g. practical AI tools for small businesses
        </span>
        <textarea
          value={focus}
          maxLength={MAX_FOCUS}
          rows={2}
          onChange={(e) => setFocus(e.target.value)}
          aria-describedby={`${id}-focus-hint`}
          disabled={starting}
        />
      </label>

      <div className="start-actions">
        <button type="submit" className="primary" disabled={!ready || starting || start.state === "busy"}>
          Make my newsletter
        </button>
        {onExample && (
          <button type="button" className="link-button" onClick={onExample}>
            See an example
          </button>
        )}
      </div>
      {starting && (
        <p role="status" className="state">
          Checking the feeds and starting the run…
        </p>
      )}
      {start.state === "busy" && (
        <div role="alert" className="state state-error">
          <p>{BUSY}</p>
          {start.wait > 0 && <p>You can try again in {start.wait} s.</p>}
          <button type="button" onClick={() => void submit()} disabled={start.wait > 0}>
            Try again
          </button>
        </div>
      )}
      {start.state === "error" && (
        <p role="alert" className="state state-error">
          Could not start the run: {start.message}
        </p>
      )}
    </form>
  );
}
