import { useState } from "react";
import type { Api, Draft } from "./api";
import { Flags, RunReport } from "./Report";

/** One draft: its text (as text, never HTML), flags and run report; edit and approve. */
export function DraftView({
  api,
  draft,
  onChange,
}: {
  api: Api;
  draft: Draft;
  onChange: (d: Draft) => void;
}) {
  const [editing, setEditing] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const unsaved = editing !== null && editing !== draft.text;

  async function act(action: () => Promise<Draft>, after?: () => void) {
    setBusy(true);
    setError(null);
    try {
      onChange(await action());
      after?.();
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setBusy(false);
    }
  }

  return (
    <article className="draft">
      <h2>{draft.title}</h2>
      <p className="muted">
        Issue date {draft.issue_date} · updated {draft.updated_at}
      </p>
      <p data-testid="approval" className={draft.approved ? "good" : "bad"}>
        {draft.approved ? `Approved ${draft.approval?.approved_at ?? ""}`.trim() : "Not approved"}
      </p>
      {error && (
        <p role="alert" className="bad">
          {error}
        </p>
      )}
      <div className="actions">
        {editing === null ? (
          <button type="button" onClick={() => setEditing(draft.text)} disabled={busy}>
            Edit
          </button>
        ) : (
          <>
            <button
              type="button"
              className="primary"
              disabled={busy || !unsaved}
              onClick={() =>
                act(
                  () => api.saveText(draft.id, editing),
                  () => setEditing(null),
                )
              }
            >
              Save
            </button>
            <button type="button" disabled={busy} onClick={() => setEditing(null)}>
              Cancel
            </button>
          </>
        )}
        <button
          type="button"
          disabled={busy || unsaved}
          title={unsaved ? "Save or cancel the edit first" : undefined}
          onClick={() =>
            act(
              () => api.getDraft(draft.id),
              () => setEditing(null),
            )
          }
        >
          Reload
        </button>
        {!draft.approved && (
          <button
            type="button"
            className="primary"
            disabled={busy || unsaved}
            title={unsaved ? "Save or cancel the edit first" : undefined}
            onClick={() => act(() => api.approve(draft))}
          >
            Approve
          </button>
        )}
      </div>
      {editing === null ? (
        draft.text ? (
          <pre className="draft-text" data-testid="draft-text">
            {draft.text}
          </pre>
        ) : (
          <p className="state">This draft has no text. Edit it to write some.</p>
        )
      ) : (
        <label className="editor">
          Draft text
          <textarea value={editing} onChange={(e) => setEditing(e.target.value)} rows={24} spellCheck />
        </label>
      )}
      <Flags flags={draft.flags ?? []} />
      <RunReport report={draft.report} />
    </article>
  );
}
