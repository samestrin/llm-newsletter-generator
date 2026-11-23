import { createElement, useMemo, useState, type ReactNode } from "react";
import Markdown, { type Components, type ExtraProps } from "react-markdown";
import type { Api, Draft, Flag } from "./api";
import { FlagNote, Flags, RunReport, httpOnly } from "./Report";

/** A hast node, as the rehype step and react-markdown's components see it. */
type Node = {
  type: string;
  value?: string;
  tagName?: string;
  properties?: Record<string, unknown>;
  children?: Node[];
};

/** The text of a node, leaving out any list nested in it (its items are matched on their own). */
function ownText(node: Node): string {
  if (node.type === "text") return node.value ?? "";
  if (node.tagName === "ul" || node.tagName === "ol") return "";
  return (node.children ?? []).map(ownText).join("");
}

const words = (text: string) => text.replace(/\s+/g, " ").trim();

/**
 * A rehype step: each flag goes to the first list item, in reading order, that holds its
 * claim's text, so the claim line gets its note and the run report's "Claims flagged"
 * list, which repeats the text further down, does not get a second one. The item carries
 * the flags' places in `flags` as `data-flags`.
 */
function markFlags(flags: Flag[]) {
  return () => (tree: Node) => {
    let left = flags.map((flag, n) => ({ n, text: words(flag.text) })).filter((f) => f.text);
    const visit = (node: Node) => {
      if (node.tagName === "li" && left.length) {
        const text = words(ownText(node));
        const mine = left.filter((f) => text.includes(f.text));
        if (mine.length) {
          node.properties = {
            ...node.properties,
            dataFlags: mine.map((f) => f.n).join(" "),
          };
          left = left.filter((f) => !mine.includes(f));
        }
      }
      node.children?.forEach(visit);
    };
    visit(tree);
  };
}

/** The draft's `#` is shown as an h3, below the page's h1 and the draft title's h2. */
function heading(level: number) {
  return ({ node: _node, ...props }: React.ComponentProps<"h1"> & ExtraProps) =>
    createElement(`h${Math.min(level + 2, 6)}`, props);
}

/**
 * How the draft's Markdown is drawn, apart from flags. Raw HTML in it stays text:
 * react-markdown escapes it, and `rehype-raw` is not used. A link is drawn only for an http
 * or https URL (`httpOnly` is the `urlTransform`), opens in a new tab and sends no
 * referrer; anything else is its text alone.
 */
const BASE: Components = {
  h1: heading(1),
  h2: heading(2),
  h3: heading(3),
  h4: heading(4),
  h5: heading(5),
  h6: heading(6),
  a: ({ href, children }) =>
    href ? (
      <a href={href} target="_blank" rel="noopener noreferrer">
        {children}
      </a>
    ) : (
      <span>{children}</span>
    ),
};

/** A list item gets the note of each flag `markFlags` gave it, beside the claim. */
function withFlags(flags: Flag[]): Components {
  return {
    ...BASE,
    li: ({ node, children, ...props }) => {
      const places = String((node as Node | undefined)?.properties?.dataFlags ?? "");
      const mine = places ? places.split(" ").map((n) => flags[Number(n)]) : [];
      return (
        <li {...props} className={mine.length ? "flagged-claim" : undefined}>
          {children}
          {mine.map((flag) => (
            <FlagNote key={flag.id} flag={flag} />
          ))}
        </li>
      );
    },
  };
}

/** One draft: the issue drawn from its Markdown, its flags and run report; edit and approve. */
export function DraftView({ api, draft, onChange }: { api: Api; draft: Draft; onChange: (d: Draft) => void }) {
  const [editing, setEditing] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  // The email for the text that was approved here; an edit that clears the approval hides it.
  const [email, setEmail] = useState<{ text: string; html: string } | null>(null);
  const preview = email && draft.approved && email.text === draft.text ? email.html : null;
  // The text approved here whose preview could not be read, so it can be asked for again.
  const [emailFailed, setEmailFailed] = useState<string | null>(null);
  const retry = preview === null && draft.approved && emailFailed === draft.text;

  async function showEmail(approved: Draft) {
    try {
      setEmail({ text: approved.text, html: (await api.email(approved.id)).html });
      setEmailFailed(null);
    } catch (e) {
      setEmailFailed(approved.text);
      throw e;
    }
  }
  const unsaved = editing !== null && editing !== draft.text;
  const flags = useMemo(() => draft.flags ?? [], [draft.flags]);
  const components = useMemo(() => withFlags(flags), [flags]);
  const rehypePlugins = useMemo(() => [markFlags(flags)], [flags]);
  const focus = draft.focus || draft.report?.focus;

  async function act(action: () => Promise<Draft>, after?: (d: Draft) => void | Promise<void>) {
    setBusy(true);
    setError(null);
    try {
      const updated = await action();
      onChange(updated);
      await after?.(updated);
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setBusy(false);
    }
  }

  let body: ReactNode;
  if (editing !== null) {
    body = (
      <label className="editor">
        Draft text
        <textarea value={editing} onChange={(e) => setEditing(e.target.value)} rows={24} spellCheck />
      </label>
    );
  } else if (draft.text) {
    body = (
      <div className="draft-text" data-testid="draft-text">
        <Markdown rehypePlugins={rehypePlugins} urlTransform={httpOnly} components={components}>
          {draft.text}
        </Markdown>
      </div>
    );
  } else {
    body = <p className="state">This draft has no text. Edit it to write some.</p>;
  }

  return (
    <article className="draft">
      <h2>{draft.title}</h2>
      {focus && (
        <p className="focus-line" data-testid="focus">
          Focus: {focus}
        </p>
      )}
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
                  (saved) => setEditing(saved.text),
                )
              }
            >
              Save
            </button>
            {unsaved && (
              <button type="button" disabled={busy} onClick={() => setEditing(draft.text)}>
                Discard changes
              </button>
            )}
            <button
              type="button"
              disabled={busy || unsaved}
              title={unsaved ? "Save or discard the edit first" : undefined}
              onClick={() => setEditing(null)}
            >
              Done editing
            </button>
          </>
        )}
        <button
          type="button"
          disabled={busy || unsaved}
          title={unsaved ? "Save or discard the edit first" : undefined}
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
            title={unsaved ? "Save or discard the edit first" : undefined}
            onClick={() => act(() => api.approve(draft), showEmail)}
          >
            Approve
          </button>
        )}
        {retry && (
          <button type="button" disabled={busy} onClick={() => act(async () => draft, showEmail)}>
            Try the preview again
          </button>
        )}
      </div>
      {preview !== null && (
        <section className="email-preview" aria-labelledby="email-preview-title">
          <h3 id="email-preview-title">Email preview</h3>
          <p className="state">Nothing was sent. This is what would go out.</p>
          {/* sandbox="" with no allow-* lets the email run no script, submit no form and open nothing. */}
          <iframe title="Email preview" sandbox="" srcDoc={preview} />
        </section>
      )}
      {editing !== null && draft.approved && (
        <p className="state">This draft is approved. Saving a change clears the approval: approve it again after.</p>
      )}
      {body}
      <Flags flags={flags} />
      <RunReport report={draft.report} focus={focus} />
    </article>
  );
}
