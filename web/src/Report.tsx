import { Component, useId, useState, type ReactNode } from "react";
import type { Flag, Report } from "./api";

/**
 * `url` if it is an absolute http or https URL, else the empty string: no relative URL, no
 * other scheme. It is the draft renderer's `urlTransform`, so a link the draft's Markdown
 * writes is checked by the same rule as a flag's source link.
 */
export function httpOnly(url: string): string {
  try {
    return ["http:", "https:"].includes(new URL(url).protocol) ? url : "";
  } catch {
    return "";
  }
}

/** A link only for http and https; any other URL is shown as plain text. */
export function SafeLink({ href }: { href: string }) {
  if (!httpOnly(href)) return <code>{href}</code>;
  return (
    <a href={href} target="_blank" rel="noopener noreferrer">
      {href}
    </a>
  );
}

/** A warning sign beside the "Flagged" text; decoration only, so screen readers skip it. */
function FlagIcon() {
  return (
    <svg
      className="flag-icon"
      data-testid="flag-icon"
      viewBox="0 0 24 24"
      aria-hidden="true"
      focusable="false"
    >
      <path fill="currentColor" d="M12 2 1 21h22L12 2Zm-1 7h2v6h-2V9Zm0 8h2v2h-2v-2Z" />
    </svg>
  );
}

/**
 * Why a claim is flagged, in plain words: the server's explanation (the model's reason on a
 * draft from before explanations were kept), the source passage and the source link, marked
 * by text and an icon.
 */
export function FlagNote({ flag }: { flag: Flag }) {
  return (
    <div role="note" className="flag-note">
      <span className="flag-label">
        <FlagIcon />
        Flagged: {flag.status}
      </span>{" "}
      {flag.explanation || flag.reason}
      {flag.passage && (
        <>
          <div className="muted">Source text:</div>
          <blockquote>{flag.passage}</blockquote>
        </>
      )}
      <div>
        Source: <SafeLink href={flag.source_url} />
      </div>
    </div>
  );
}

export function Flags({ flags }: { flags: Flag[] }) {
  return (
    <section aria-labelledby="flags-heading" className="flags">
      <h3 id="flags-heading">Flags ({flags.length})</h3>
      {flags.length === 0 ? (
        <p>No flagged claims.</p>
      ) : (
        <ul>
          {flags.map((flag) => (
            <li key={flag.id} className="flag">
              <q>{flag.text}</q> <span className="muted">{flag.id}</span>
              <FlagNote flag={flag} />
            </li>
          ))}
        </ul>
      )}
    </section>
  );
}

function List<T>({ title, items, show }: { title: string; items?: T[]; show: (item: T) => React.ReactNode }) {
  if (!items?.length) return null;
  return (
    <>
      <h4>
        {title} ({items.length})
      </h4>
      <ul>
        {items.map((item, n) => (
          <li key={n}>{show(item)}</li>
        ))}
      </ul>
    </>
  );
}

const money = (cost: number) => `$${cost.toFixed(4)}`;

/** Shows "could not be read" in place of a report that breaks while it renders. */
class Unreadable extends Component<{ report: Report; children: ReactNode }, { failed: boolean }> {
  state = { failed: false };
  static getDerivedStateFromError() {
    return { failed: true };
  }
  componentDidUpdate(previous: { report: Report }) {
    // A new report, from a reload or a save, gets its own try.
    if (this.state.failed && previous.report !== this.props.report) this.setState({ failed: false });
  }
  render() {
    return this.state.failed ? <ReportError /> : this.props.children;
  }
}

function ReportError() {
  return (
    <p role="alert" className="state state-error">
      The run report could not be read.
    </p>
  );
}

/** The run report section: loading, none saved, unreadable, or the report. */
export function RunReport({
  report,
  loading = false,
  focus,
}: {
  report?: Report | null;
  loading?: boolean;
  /** The run's focus, shown first. */
  focus?: string | null;
}) {
  return (
    <section aria-labelledby="report-heading" className="report">
      <h3 id="report-heading">Run report</h3>
      {focus && <p>Focus: {focus}</p>}
      {loading ? (
        <p className="state">Loading the run report…</p>
      ) : report == null ? (
        <p className="state">No run report was saved with this draft.</p>
      ) : typeof report !== "object" || Array.isArray(report) ? (
        <ReportError />
      ) : (
        <Unreadable report={report}>
          <ReportBody report={report} />
        </Unreadable>
      )}
    </section>
  );
}

/**
 * The run in a few lines, built from the report: the models used (each once), the number
 * of model calls, the total cost, and the editor's plain message when it failed.
 */
function Summary({ report }: { report: Report }) {
  const steps = report.steps ?? [];
  const models = [...new Set(steps.map((s) => s.model).filter(Boolean))];
  // An older report has no cost_complete: a step with no cost means the same.
  const complete = report.cost_complete ?? steps.every((s) => s.cost != null);
  const failed = report.editor_message || (report.editor_error && "The editor failed.");
  return (
    <>
      <ul aria-label="Run summary">
        <li>Models used: {models.length ? models.join(", ") : "none"}</li>
        <li>Steps: {steps.length}</li>
        <li>
          Total cost: {report.total_cost == null ? "unknown" : money(report.total_cost)}
          {report.total_cost != null && !complete && ", some costs unknown"}
        </li>
      </ul>
      {failed && <p className="bad">{failed}</p>}
    </>
  );
}

function ReportBody({ report }: { report: Report }) {
  const [open, setOpen] = useState(false);
  const details = useId();
  const voice = report.voice;
  return (
    <>
      <p className={report.ok === false ? "bad" : "good"}>
        {report.ok === false ? "Problems found" : "Nothing failed, was left out or is flagged"}
      </p>
      <Summary report={report} />
      <List
        title="Failed sources"
        items={report.failed_sources}
        show={(f) => (
          <>
            <code>{f.url}</code>: {f.reason}
          </>
        )}
      />
      <List
        title="Linked articles"
        items={report.linked_articles}
        show={(a) => (
          <>
            {a.title} (<code>{a.link}</code>): {a.status}
            {a.reason ? `, ${a.reason}` : ""}
          </>
        )}
      />
      <List
        title="Dropped stories"
        items={report.dropped_stories}
        show={(s) => (
          <>
            {s.title} (<code>{s.link}</code>): {s.reason}
          </>
        )}
      />
      <List
        title="Dropped claims"
        items={report.dropped_claims}
        show={(c) => (
          <>
            <q>{c.text}</q> ({c.id}): {c.reason}
          </>
        )}
      />
      <List title="Editor rejections" items={report.editor_rejections} show={(r) => r} />
      <List title="Failed steps" items={report.failed_steps} show={(s) => s} />
      {voice && (
        <>
          <h4>Voice</h4>
          {voice.skipped ? (
            <p>Not run: {voice.skipped}</p>
          ) : (
            <p>
              {voice.enabled ? "In the writer's voice" : "Off"}
              {voice.voiced?.length ? `: ${voice.voiced.join(", ")}` : ""}
            </p>
          )}
          <List
            title="Kept neutral"
            items={voice.kept_neutral}
            show={(k) => (
              <>
                {k.id}: {k.reason}
              </>
            )}
          />
        </>
      )}
      <button
        type="button"
        className="link-button"
        aria-expanded={open}
        aria-controls={details}
        onClick={() => setOpen(!open)}
      >
        {open ? "Hide details" : "Show details"}
      </button>
      <div id={details} hidden={!open}>
        {open && <Details report={report} />}
      </div>
    </>
  );
}

/** The editor's raw error, each model call and the cost it left out: behind Show details. */
function Details({ report }: { report: Report }) {
  const steps = report.steps ?? [];
  const unknown = steps.filter((s) => s.cost == null).length;
  return (
    <>
      {report.editor_error && <p className="bad">Editor failed: {report.editor_error}</p>}
      <h4>Model calls</h4>
      {steps.length === 0 ? (
        <p>None.</p>
      ) : (
        <div className="table-scroll" role="region" aria-label="Model calls table" tabIndex={0}>
          <table>
            <thead>
              <tr>
                <th>Step</th>
                <th>Tier</th>
                <th>Model</th>
                <th>Tokens in</th>
                <th>Tokens out</th>
                <th>Cost</th>
              </tr>
            </thead>
            <tbody>
              {steps.map((s, n) => (
                <tr key={n}>
                  <td>{s.step}</td>
                  <td>{s.tier}</td>
                  <td>{s.model}</td>
                  <td>{s.prompt_tokens ?? ""}</td>
                  <td>{s.completion_tokens ?? ""}</td>
                  <td>{s.cost == null ? "unknown" : money(s.cost)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
      {unknown > 0 && (
        <p>
          {unknown} step{unknown === 1 ? "" : "s"} cost unknown, not counted in the total.
        </p>
      )}
    </>
  );
}
