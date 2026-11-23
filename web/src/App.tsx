import { useEffect, useRef, useState, type FormEvent } from "react";
import {
  ApiError,
  createApi,
  listProviders,
  loadCurrentRun,
  loadMine,
  rememberRun,
  saveCurrentRun,
  type Access,
  type Api,
  type Draft,
  type DraftSummary,
  type Provider,
  type Run,
} from "./api";
import { DraftView } from "./DraftView";
import { KeyPanel } from "./KeyPanel";
import {
  clearCredentials,
  loadCredentials,
  saveCredentials,
  type ModelCredentials,
  type StoredCredentials,
} from "./modelAccess";
import { Progress } from "./Progress";
import { RunReport } from "./Report";
import { StartForm } from "./StartForm";

const API_BASE: string = import.meta.env.VITE_API_URL ?? "";

/** What the page has of something it loads: nothing yet, waiting, a failure, or the thing. */
type Loaded<T> =
  | { state: "idle" }
  | { state: "loading" }
  | { state: "error"; message: string }
  | { state: "ready"; value: T };

const message = (e: unknown) => (e instanceof Error ? e.message : String(e));

/**
 * How the server wants to be reached: still asking, the review host's token (server-key
 * mode), or a visitor's key or demo code for one of its providers.
 */
type Mode = { kind: "checking" } | { kind: "token" } | { kind: "keys"; providers: Provider[] };

/**
 * In server-key mode the token is kept in memory only: reloading the page asks for it again.
 * With providers, the key panel keeps the visitor's key or demo code (see `modelAccess`).
 */
export default function App() {
  const [mode, setMode] = useState<Mode>({ kind: "checking" });
  const [api, setApi] = useState<Api | null>(null);
  const [token, setToken] = useState("");
  const [credentials, setCredentials] = useState<StoredCredentials | null>(null);
  const [panel, setPanel] = useState<{ open: boolean; error: string | null }>({ open: false, error: null });
  const [drafts, setDrafts] = useState<Loaded<DraftSummary[]>>({ state: "idle" });
  const [draft, setDraft] = useState<Loaded<Draft>>({ state: "idle" });
  const [draftId, setDraftId] = useState<string | null>(null);
  // The id of the run this tab started and is watching; kept across a reload, so its live
  // progress shows again once the page has its key back.
  const [run, setRun] = useState<string | null>(loadCurrentRun);
  // Only the draft opened last may fill the page, whatever order the answers come in.
  const latest = useRef<string | null>(null);
  // With providers, a 401 is the key panel's to explain; the list does not say it again.
  const keyed = useRef(false);

  // A refused key or code: say why (the API names the header, never the value) and ask again.
  const refused = (error: ApiError) => setPanel({ open: true, error: error.message });
  const keyApi = (model: ModelCredentials) => createApi({ model } satisfies Access, API_BASE, refused);

  useEffect(() => {
    let live = true;
    void listProviders(API_BASE).then((providers) => {
      if (!live) return;
      if (!providers) {
        setMode({ kind: "token" });
        return;
      }
      keyed.current = true;
      setMode({ kind: "keys", providers });
      const saved = loadCredentials();
      setCredentials(saved);
      if (saved) void loadList(keyApi(saved));
      else setPanel({ open: true, error: null });
    });
    return () => {
      live = false;
    };
    // Once, on the first visit: the providers do not change while the page is open.
  }, []);

  /** List the drafts with `client` and make it the page's client; `false` when it was refused. */
  async function loadList(client: Api): Promise<boolean> {
    const before = drafts;
    setDrafts({ state: "loading" });
    try {
      setDrafts({ state: "ready", value: await client.listDrafts() });
      setApi(client);
    } catch (e) {
      const rejected = e instanceof ApiError && e.status === 401;
      // A refused key keeps the list the working client already showed.
      setDrafts(
        keyed.current && rejected
          ? before.state === "ready"
            ? before
            : { state: "idle" }
          : { state: "error", message: message(e) },
      );
      return !rejected;
    }
    return true;
  }

  function connect(event: FormEvent) {
    event.preventDefault();
    void loadList(createApi({ token }, API_BASE));
  }

  /**
   * Keep a new key unless the API refuses it: a refused one leaves the saved key and its client
   * in place. Any other failure keeps it, so Try again retries with it.
   */
  async function saveKey(model: ModelCredentials, remember: boolean) {
    setPanel({ open: false, error: null });
    if (!(await loadList(keyApi(model)))) return;
    saveCredentials(model, remember);
    setCredentials({ ...model, remember });
  }

  function clearKey() {
    clearCredentials();
    setCredentials(null);
    setApi(null);
    setDrafts({ state: "idle" });
    showForm();
    setPanel({ open: true, error: null });
  }

  /** Back to the form, or to `watch` or the run this tab is still watching: no draft open. */
  function showForm(watch: string | null = null) {
    latest.current = null;
    setDraftId(null);
    setDraft({ state: "idle" });
    setRun(watch ?? loadCurrentRun());
  }

  function started(made: Run) {
    rememberRun(made);
    saveCurrentRun(made.id);
    // Watched from state, not read back from storage: blocked storage still shows the run.
    showForm(made.id);
  }

  /** The run made its draft: open it, and list it with the others. */
  function finished(id: string) {
    void open(id);
    if (api) void loadList(api);
  }

  async function open(id: string) {
    if (!api) return;
    setRun(null);
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

  const demo = mode.kind === "keys";
  // The seeded example: in demo mode the API lists it and this browser's drafts only, so it
  // is the oldest listed draft this browser did not make.
  const example =
    demo && drafts.state === "ready"
      ? drafts.value.filter((d) => !loadMine().drafts.includes(d.id)).at(-1)?.id
      : undefined;

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
        <div className="site-header-bar">
          <h1>Newsletter review</h1>
          {mode.kind === "keys" && (
            <button type="button" aria-expanded={panel.open} onClick={() => setPanel({ open: true, error: null })}>
              Key
            </button>
          )}
        </div>
      </header>
      {mode.kind === "keys" && panel.open && (
        <KeyPanel
          providers={mode.providers}
          stored={credentials}
          error={panel.error}
          onSave={saveKey}
          onClear={clearKey}
          onClose={() => setPanel({ open: false, error: null })}
        />
      )}
      <div className={api ? `layout layout-split${demo ? " layout-demo" : ""}` : "layout"}>
        {api && (
          <nav aria-label={demo ? "Earlier drafts" : "Drafts"}>
            {demo ? (
              <>
                <button type="button" onClick={() => showForm()} aria-current={!draftId ? "page" : undefined}>
                  Make a newsletter
                </button>
                <h2>Earlier drafts</h2>
              </>
            ) : (
              <h2 className="sr-only">Drafts</h2>
            )}
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
                <p className="state">
                  {demo ? "No earlier drafts yet." : "No drafts yet. Start a run with POST /api/runs, then refresh the list."}
                </p>
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
          {mode.kind === "checking" ? (
            <p role="status" className="state">
              Loading…
            </p>
          ) : !api && mode.kind === "keys" ? (
            <>
              {listStatus}
              {drafts.state === "error" && credentials && (
                <button type="button" onClick={() => loadList(keyApi(credentials))}>
                  Try again
                </button>
              )}
              {!panel.open && !credentials && (
                <p className="state">Add a key or a demo code with the Key button to start.</p>
              )}
            </>
          ) : !api ? (
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
          ) : run ? (
            <Progress key={run} api={api} runId={run} onDone={finished} onStartOver={() => showForm()} />
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
          ) : demo ? (
            <StartForm api={api} onStarted={started} onExample={example ? () => open(example) : undefined} />
          ) : (
            drafts.state === "ready" &&
            drafts.value.length > 0 && <p className="state">Choose a draft to review.</p>
          )}
        </main>
      </div>
    </>
  );
}
