/** The review page's calls to `newsletter serve`: list, read, edit and approve drafts. */

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
  dropped_stories?: { story_id: string; title: string; link: string; reason: string }[];
  dropped_claims?: { id: string; story_id: string; text: string; passage?: string; reason: string }[];
  flagged_claims?: Flag[];
  editor_error?: string | null;
  editor_rejections?: string[];
  steps?: {
    step?: string;
    tier?: string;
    model?: string;
    input_tokens?: number | null;
    output_tokens?: number | null;
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

export function createApi(token: string, base = "") {
  async function call<T>(method: string, path: string, body?: object): Promise<T> {
    const headers: Record<string, string> = { Authorization: `Bearer ${token}` };
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
      throw new ApiError(
        response.status,
        typeof detail === "string" ? detail : `the API answered ${response.status}`,
      );
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
