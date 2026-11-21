/** The review page's calls to `newsletter serve`: list, read, edit and approve drafts. */
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
  reason: string;
  source_url: string;
};

/** `RunReport.to_dict()`; every field optional, so an older or partial report still shows. */
export type Report = {
  ok?: boolean;
  failed_sources?: { url: string; reason: string }[];
  linked_articles?: { title: string; link: string; status: string; reason?: string | null }[];
  dropped_stories?: { story_id: string; title: string; link: string; reason: string }[];
  dropped_claims?: { id: string; story_id: string; text: string; passage?: string; reason: string }[];
  flagged_claims?: Flag[];
  editor_error?: string | null;
  editor_rejections?: string[];
  steps?: {
    step?: string;
    tier?: string;
    model?: string;
    prompt_tokens?: number | null;
    completion_tokens?: number | null;
    cost?: number | null;
  }[];
  total_cost?: number;
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
};

export type DraftSummary = Omit<Draft, "text" | "flags" | "report" | "approval"> & { flags: number };

export class ApiError extends Error {
  constructor(
    readonly status: number,
    message: string,
  ) {
    super(message);
    this.name = "ApiError";
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
      const error = new ApiError(response.status, shown ? detail : `the API answered ${response.status}`);
      if (response.status === 401) onUnauthorized?.(error);
      throw error;
    }
    return data as T;
  }

  return {
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
