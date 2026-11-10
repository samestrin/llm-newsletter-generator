import { useRef, useState, type FormEvent } from "react";
import { createApi, type Api, type Draft, type DraftSummary } from "./api";
import { DraftView } from "./DraftView";
import { RunReport } from "./Report";

const API_BASE: string = import.meta.env.VITE_API_URL ?? "";

/** What the page has of something it loads: nothing yet, waiting, a failure, or the thing. */
type Loaded<T> =
  | { state: "idle" }
  | { state: "loading" }
  | { state: "error"; message: string }
  | { state: "ready"; value: T };

const message = (e: unknown) => (e instanceof Error ? e.message : String(e));

/** The token is kept in memory only: reloading the page asks for it again. */
export default function App() {
  const [api, setApi] = useState<Api | null>(null);
  const [token, setToken] = useState("");
  const [drafts, setDrafts] = useState<Loaded<DraftSummary[]>>({ state: "idle" });
  const [draft, setDraft] = useState<Loaded<Draft>>({ state: "idle" });
  const [draftId, setDraftId] = useState<string | null>(null);
  // Only the draft opened last may fill the page, whatever order the answers come in.
  const latest = useRef<string | null>(null);

  async function loadList(client: Api) {
    setDrafts({ state: "loading" });
    try {
      setDrafts({ state: "ready", value: await client.listDrafts() });
      setApi(client);
    } catch (e) {
      setDrafts({ state: "error", message: message(e) });
    }
  }

  function connect(event: FormEvent) {
    event.preventDefault();
    void loadList(createApi(token, API_BASE));
  }

  async function open(id: string) {
    if (!api) return;
    latest.current = id;
    setDraftId(id);
    setDraft({ state: "loading" });
    try {
      const found = await api.getDraft(id);
      if (latest.current === id) setDraft({ state: "ready", value: found });
    } catch (e) {
      if (latest.current === id) setDraft({ state: "error", message: message(e) });
    }
  }

  function changed(updated: Draft) {
    setDraft({ state: "ready", value: updated });
    setDrafts((list) =>
      list.state !== "ready"
        ? list
        : {
            state: "ready",
            value: list.value.map((d) =>
              d.id === updated.id ? { ...d, approved: updated.approved, updated_at: updated.updated_at } : d,
            ),
          },
    );
  }

  const listStatus =
    drafts.state === "loading" ? (
      <p role="status" className="state">
        Loading the drafts…
      </p>
    ) : drafts.state === "error" ? (
      <p role="alert" className="state state-error">
        Could not load the drafts: {drafts.message}
      </p>
    ) : null;

  return (
    <>
      <a className="skip-link" href="#main">
        Skip to main content
      </a>
      <header className="site-header">
        <h1>Newsletter review</h1>
      </header>
      <div className={api ? "layout layout-split" : "layout"}>
        {api && (
          <nav aria-label="Drafts">
            <h2 className="sr-only">Drafts</h2>
            <button
              type="button"
              className="refresh"
              onClick={() => loadList(api)}
              disabled={drafts.state === "loading"}
            >
              Refresh list
            </button>
            {listStatus}
            {drafts.state === "ready" &&
              (drafts.value.length === 0 ? (
                <p className="state">No drafts yet. Start a run with POST /api/runs, then refresh the list.</p>
              ) : (
                <ul>
                  {drafts.value.map((d) => (
                    <li key={d.id}>
                      <button
                        type="button"
                        onClick={() => open(d.id)}
                        aria-current={draftId === d.id ? "true" : undefined}
                      >
                        <strong>{d.title}</strong> {d.issue_date} · {d.flags} flag{d.flags === 1 ? "" : "s"} ·{" "}
                        {d.approved ? "Approved" : "Not approved"}
                      </button>
                    </li>
                  ))}
                </ul>
              ))}
          </nav>
        )}
        <main id="main" tabIndex={-1} aria-busy={draft.state === "loading" || undefined}>
          {!api ? (
            <form onSubmit={connect} className="connect">
              <label>
                API token
                <input
                  type="password"
                  autoComplete="off"
                  value={token}
                  onChange={(e) => setToken(e.target.value)}
                />
              </label>
              <button type="submit" className="primary" disabled={!token || drafts.state === "loading"}>
                Connect
              </button>
              {listStatus}
            </form>
          ) : draft.state === "ready" ? (
            <DraftView key={draft.value.id} api={api} draft={draft.value} onChange={changed} />
          ) : draft.state === "loading" ? (
            <article className="draft">
              <p role="status" className="state">
                Loading the draft…
              </p>
              <RunReport loading />
            </article>
          ) : draft.state === "error" ? (
            <div role="alert" className="state state-error">
              <p>Could not load the draft: {draft.message}</p>
              <button type="button" onClick={() => draftId && open(draftId)}>
                Try again
              </button>
            </div>
          ) : (
            drafts.state === "ready" &&
            drafts.value.length > 0 && <p className="state">Choose a draft to review.</p>
          )}
        </main>
      </div>
    </>
  );
}
