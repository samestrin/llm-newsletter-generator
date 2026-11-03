import { useState, type FormEvent } from "react";
import { createApi, type Api, type Draft, type DraftSummary } from "./api";
import { DraftView } from "./DraftView";

const API_BASE: string = import.meta.env.VITE_API_URL ?? "";

/** The token is kept in memory only: reloading the page asks for it again. */
export default function App() {
  const [api, setApi] = useState<Api | null>(null);
  const [token, setToken] = useState("");
  const [drafts, setDrafts] = useState<DraftSummary[]>([]);
  const [draft, setDraft] = useState<Draft | null>(null);
  const [error, setError] = useState<string | null>(null);

  async function run<T>(action: () => Promise<T>): Promise<T | undefined> {
    setError(null);
    try {
      return await action();
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
      return undefined;
    }
  }

  async function connect(event: FormEvent) {
    event.preventDefault();
    const client = createApi(token, API_BASE);
    const list = await run(client.listDrafts);
    if (list) {
      setApi(client);
      setDrafts(list);
    }
  }

  async function open(id: string) {
    if (!api) return;
    const found = await run(() => api.getDraft(id));
    if (found) setDraft(found);
  }

  function changed(updated: Draft) {
    setDraft(updated);
    setDrafts((list) =>
      list.map((d) =>
        d.id === updated.id ? { ...d, approved: updated.approved, updated_at: updated.updated_at } : d,
      ),
    );
  }

  return (
    <main>
      <h1>Newsletter review</h1>
      {error && (
        <p role="alert" className="bad">
          {error}
        </p>
      )}
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
          <button type="submit" disabled={!token}>
            Connect
          </button>
        </form>
      ) : (
        <div className="layout">
          <nav aria-label="Drafts">
            <h2 className="sr-only">Drafts</h2>
            {drafts.length === 0 ? (
              <p>No drafts yet. Start a run with POST /api/runs.</p>
            ) : (
              <ul>
                {drafts.map((d) => (
                  <li key={d.id}>
                    <button
                      type="button"
                      onClick={() => open(d.id)}
                      aria-current={draft?.id === d.id ? "true" : undefined}
                    >
                      <strong>{d.title}</strong> {d.issue_date} · {d.flags} flag{d.flags === 1 ? "" : "s"} ·{" "}
                      {d.approved ? "Approved" : "Not approved"}
                    </button>
                  </li>
                ))}
              </ul>
            )}
          </nav>
          {draft && <DraftView key={draft.id} api={api} draft={draft} onChange={changed} />}
        </div>
      )}
    </main>
  );
}
