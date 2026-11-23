/** The page's calls to `newsletter serve`: check feeds, start a run, list, read, edit and approve drafts. */
import { modelHeaders, type ModelCredentials } from "./modelAccess";

/** One entry of `GET /api/providers` (plan 08's `providers.json`); it holds no secrets. */
export type Provider = {
  id: string;
  label: string;
  kind: "key" | "demo";
  base_url: string;
  fast_model: string;
  reasoning_model: string;
  /** `null` when the provider has no embedding model: stories are then not grouped by meaning. */
  embedding_model: string | null;
};

/**
 * How the page reaches the API: the review host's token (server-key mode, sent as
 * `Authorization`), or the visitor's key or demo code (sent as the `X-Model-*` headers).
 */
export type Access = { token: string } | { model: ModelCredentials };

export type Flag = {
  id: string;
  story_id: string;
  text: string;
  status: string;
  /** The check's own reason: the model's, or why the check failed. */
  reason: string;
  source_url: string;
  /** The source text the claim was tied to; absent from older drafts. */
  passage?: string | null;
  /** Why it is flagged, in plain words built by the server; absent from older drafts. */
  explanation?: string | null;
};

/** `RunReport.to_dict()`; every field optional, so an older or partial report still shows. */
export type Report = {
  ok?: boolean;
  /** What the visitor asked the issue to be about, if they gave a focus. */
  focus?: string | null;
  failed_sources?: { url: string; reason: string }[];
  linked_articles?: { title: string; link: string; status: string; reason?: string | null }[];
  dropped_stories?: { story_id: string; title: string; link: string; reason: string }[];
  dropped_claims?: { id: string; story_id: string; text: string; passage?: string; reason: string }[];
  flagged_claims?: Flag[];
  /** The raw error of an editor pass that failed. */
  editor_error?: string | null;
  /** The same failure in one plain sentence, set only when the editor failed. */
  editor_message?: string | null;
  editor_rejections?: string[];
  steps?: {
    step?: string;
    tier?: string;
    model?: string;
    prompt_tokens?: number | null;
    completion_tokens?: number | null;
    cost?: number | null;
  }[];
  total_cost?: number | null;
  /** False when a step's cost is unknown, so `total_cost` leaves it out. */
  cost_complete?: boolean;
  failed_steps?: string[];
  voice?: {
    enabled?: boolean;
    skipped?: string | null;
    voiced?: string[];
    kept_neutral?: { id: string; reason: string }[];
    ok?: boolean;
  };
};

export type Draft = {
  id: string;
  run_id: string;
  title: string;
  issue_date: string;
  text: string;
  flags: Flag[];
  report: Report;
  approval: { sha256: string; approved_at: string } | null;
  approved: boolean;
  created_at: string;
  updated_at: string;
  /** The run's focus, when the server keeps it on the draft. */
  focus?: string | null;
};

export type DraftSummary = Omit<Draft, "text" | "flags" | "report" | "approval"> & { flags: number };

/** A feed set from `GET /api/presets`: a shortcut the form copies into its rows. */
export type Preset = { id: string; label: string; feeds: string[] };

/** `POST /api/feeds/check`: `error` is a plain sentence when `ok` is false. */
export type FeedCheck = { ok: boolean; title?: string | null; items?: number; error?: string | null };

/** One step of a run's live progress (plan 08's `ProgressStep`). */
export type ProgressStep = {
  id: string;
  label: string;
  state: "waiting" | "running" | "done" | "failed" | "skipped";
  done: number | null;
  total: number | null;
  /** Built by the server from counts, URLs and fetch reasons, never a model's reply. */
  detail: string | null;
  started_at: string | null;
  ended_at: string | null;
};

/** A run as `POST /api/runs` and `GET /api/runs/{id}` return it. */
export type Run = {
  id: string;
  status: "queued" | "running" | "done" | "error";
  error: string | null;
  draft_id: string | null;
  feeds: string[];
  title: string;
  created_at: string;
  started_at: string | null;
  finished_at: string | null;
  /** Demo mode only, in the `POST /api/runs` answer: proves the drafts it makes are ours. */
  owner_token?: string | null;
  /** Every step in order; empty or missing for a run from before progress was kept. */
  progress?: ProgressStep[];
  /** A plain sentence for the visitor when the run failed, if the server has one. */
  editor_message?: string | null;
};

export class ApiError extends Error {
  constructor(
    readonly status: number,
    message: string,
    /** Seconds to wait before asking again, from a 409 or 429 (`retry_after` or `Retry-After`). */
    readonly retryAfter: number | null = null,
  ) {
    super(message);
    this.name = "ApiError";
  }
}

/**
 * What this browser made on a demo-mode server, kept in session storage: the owner token
 * the server issued with a run (sent back as `X-Owner-Token`, so the API shows us our
 * drafts) and the ids of our runs and drafts. Never a key or demo code.
 */
export type Mine = { owner_token: string | null; runs: string[]; drafts: string[] };

export const MINE_KEY = "newsletter.mine";

const strings = (value: unknown) =>
  Array.isArray(value) ? value.filter((v): v is string => typeof v === "string") : [];

export function loadMine(): Mine {
  try {
    const value = JSON.parse(sessionStorage.getItem(MINE_KEY) ?? "null");
    return {
      owner_token: typeof value?.owner_token === "string" && value.owner_token ? value.owner_token : null,
      runs: strings(value?.runs),
      drafts: strings(value?.drafts),
    };
  } catch {
    // Blocked storage or a malformed value: nothing is ours yet.
    return { owner_token: null, runs: [], drafts: [] };
  }
}

function saveMine(mine: Mine) {
  try {
    sessionStorage.setItem(MINE_KEY, JSON.stringify(mine));
  } catch {
    // Blocked storage: this visit still works, the list just will not hold our drafts.
  }
}

/** Keep a started run's id, and the owner token when the server issued one. */
export function rememberRun(run: Run): void {
  const mine = loadMine();
  saveMine({
    owner_token: run.owner_token || mine.owner_token,
    runs: mine.runs.includes(run.id) ? mine.runs : [...mine.runs, run.id],
    drafts: run.draft_id && !mine.drafts.includes(run.draft_id) ? [...mine.drafts, run.draft_id] : mine.drafts,
  });
}

/**
 * The run this tab is watching, kept in session storage so a reload resumes it: its id
 * only, never a key, demo code or owner token.
 */
export const RUN_KEY = "newsletter.run";

export function loadCurrentRun(): string | null {
  try {
    const id = sessionStorage.getItem(RUN_KEY);
    return id ? id : null;
  } catch {
    return null;
  }
}

export function saveCurrentRun(id: string): void {
  try {
    sessionStorage.setItem(RUN_KEY, id);
  } catch {
    // Blocked storage: the run still shows, a reload just will not resume it.
  }
}

export function clearCurrentRun(): void {
  try {
    sessionStorage.removeItem(RUN_KEY);
  } catch {
    // Blocked storage: nothing was kept.
  }
}

/** The hex sha256 of the text's UTF-8 bytes, as the API's `text_sha256` computes it. */
export async function sha256Hex(text: string): Promise<string> {
  if (!globalThis.crypto?.subtle) {
    throw new ApiError(0, "approving needs a secure page: open it on https, localhost or 127.0.0.1");
  }
  const digest = await crypto.subtle.digest("SHA-256", new TextEncoder().encode(text));
  return Array.from(new Uint8Array(digest), (b) => b.toString(16).padStart(2, "0")).join("");
}

/**
 * The providers a visitor may pick, asked with no credentials. `null` means the server is
 * in server-key mode: it answered no list (today's server has no such route), an empty one,
 * or nothing at all.
 */
export async function listProviders(base = ""): Promise<Provider[] | null> {
  try {
    const response = await fetch(`${base}/api/providers`);
    if (!response.ok) return null;
    const data = await response.json();
    return Array.isArray(data) && data.length > 0 ? (data as Provider[]) : null;
  } catch {
    return null;
  }
}

/** `onUnauthorized` hears every 401, so the page can ask for the key or token again. */
export function createApi(access: Access, base = "", onUnauthorized?: (error: ApiError) => void) {
  const secret = "token" in access ? access.token : access.model.secret;
  const credentials: Record<string, string> =
    "token" in access ? { Authorization: `Bearer ${access.token}` } : modelHeaders(access.model);

  async function call<T>(method: string, path: string, body?: object): Promise<T> {
    const headers: Record<string, string> = { ...credentials };
    const owner = "model" in access ? loadMine().owner_token : null;
    if (owner) headers["X-Owner-Token"] = owner;
    if (body) headers["Content-Type"] = "application/json";
    let response: Response;
    try {
      response = await fetch(`${base}${path}`, {
        method,
        headers,
        body: body ? JSON.stringify(body) : undefined,
      });
    } catch {
      throw new ApiError(0, "could not reach the API");
    }
    const data = await response.json().catch(() => undefined);
    if (!response.ok) {
      const detail = data?.detail;
      // A message that echoes the secret is never shown: the page would render it. A secret
      // under 4 characters is not looked for: it would match ordinary words.
      const shown = typeof detail === "string" && !(secret.length >= 4 && detail.includes(secret));
      const wait = Number(data?.retry_after ?? response.headers.get("Retry-After") ?? NaN);
      const error = new ApiError(
        response.status,
        shown ? detail : `the API answered ${response.status}`,
        Number.isFinite(wait) && wait >= 0 ? wait : null,
      );
      if (response.status === 401) onUnauthorized?.(error);
      throw error;
    }
    return data as T;
  }

  return {
    presets: () => call<Preset[]>("GET", "/api/presets"),
    checkFeed: (url: string) => call<FeedCheck>("POST", "/api/feeds/check", { url }),
    /** Start a run; a busy server answers 409 with `retryAfter` on the error. */
    startRun: (run: { feeds: string[]; focus?: string }) => call<Run>("POST", "/api/runs", run),
    getRun: (id: string) => call<Run>("GET", `/api/runs/${encodeURIComponent(id)}`),
    listDrafts: () => call<DraftSummary[]>("GET", "/api/drafts"),
    getDraft: (id: string) => call<Draft>("GET", `/api/drafts/${encodeURIComponent(id)}`),
    saveText: (id: string, text: string) =>
      call<Draft>("PUT", `/api/drafts/${encodeURIComponent(id)}`, { text }),
    /** Approve `draft.text` as read; the API refuses with 409 if it changed since. */
    approve: async (draft: Draft) =>
      call<Draft>("POST", `/api/drafts/${encodeURIComponent(draft.id)}/approve`, {
        sha256: await sha256Hex(draft.text),
      }),
  };
}

export type Api = ReturnType<typeof createApi>;
