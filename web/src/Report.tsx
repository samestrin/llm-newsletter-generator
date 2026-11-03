import type { Flag, Report } from "./api";

/** A link only for http and https; any other URL is shown as plain text. */
export function SafeLink({ href }: { href: string }) {
  let safe = false;
  try {
    safe = ["http:", "https:"].includes(new URL(href).protocol);
  } catch {
    safe = false;
  }
  if (!safe) return <code>{href}</code>;
  return (
    <a href={href} target="_blank" rel="noopener noreferrer">
      {href}
    </a>
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
            <li key={flag.id}>
              <span className={`status status-${flag.status}`}>{flag.status}</span>{" "}
              <q>{flag.text}</q>
              <div className="muted">
                {flag.id} · {flag.reason}
              </div>
              <div>
                Source: <SafeLink href={flag.source_url} />
              </div>
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

export function RunReport({ report }: { report: Report }) {
  const steps = report.steps ?? [];
  const unknown = steps.filter((s) => s.cost == null).length;
  const voice = report.voice;
  return (
    <section aria-labelledby="report-heading" className="report">
      <h3 id="report-heading">Run report</h3>
      <p className={report.ok === false ? "bad" : "good"}>
        {report.ok === false ? "Problems found" : "Nothing failed, was left out or is flagged"}
      </p>
      {report.editor_error && <p className="bad">Editor failed: {report.editor_error}</p>}
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
      <h4>Model calls</h4>
      {steps.length === 0 ? (
        <p>None.</p>
      ) : (
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
                <td>{s.input_tokens ?? ""}</td>
                <td>{s.output_tokens ?? ""}</td>
                <td>{s.cost == null ? "unknown" : money(s.cost)}</td>
              </tr>
            ))}
          </tbody>
        </table>
      )}
      <p>
        Total cost: {money(report.total_cost ?? 0)}
        {unknown > 0 && ` (${unknown} step${unknown === 1 ? "" : "s"} cost unknown, not counted)`}
      </p>
    </section>
  );
}
