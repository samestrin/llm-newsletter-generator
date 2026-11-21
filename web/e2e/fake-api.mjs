// A fake of `newsletter serve` for the browser test: the same draft endpoints and the
// same write rules (token, then Origin, then a JSON object body), drafts kept in memory.
// `POST /__reset` puts the one draft back as it started, with any fields in its JSON body
// (title, text, flags, report...) in place of the seed's. `POST /__keys` turns on demo
// mode until the next reset: `GET /api/providers` lists providers, and the draft endpoints
// take the demo code or OpenRouter key below as `X-Model-*` headers instead of the token.
// In demo mode it also serves plan 08.1's `GET /api/presets`, `POST /api/feeds/check`
// (answers from FEEDS below) and `POST /api/runs` (409 with `retry_after` while `busy`,
// set by `POST /__busy`; otherwise a queued run with an owner token).
import { createHash } from "node:crypto";
import { createServer } from "node:http";

const PORT = Number(process.env.FAKE_API_PORT ?? 8787);
const TOKEN = process.env.FAKE_API_TOKEN ?? "e2e-token";
const ORIGINS = (process.env.FAKE_API_ORIGINS ?? "").split(",").filter(Boolean);
const ID = "0123456789abcdef0123456789abcdef";
const DEMO_CODE = "e2e-demo-code";
const OPENROUTER_KEY = "sk-or-e2e-key";
const provider = (id, label, kind, embedding_model) => ({
  id,
  label,
  kind,
  base_url: `https://${id}.example.com/v1`,
  fast_model: `${id}-fast`,
  reasoning_model: `${id}-reasoning`,
  embedding_model,
});
const PROVIDERS = [
  provider("demo", "Demo code", "demo", "demo-embedding"),
  provider("openai", "OpenAI", "key", "text-embedding-3-small"),
  provider("openrouter", "OpenRouter", "key", "thenlper/gte-base"),
  provider("alibaba", "Alibaba", "key", "text-embedding-v4"),
  provider("synthetic", "Synthetic", "key", null),
];
const FEEDS = {
  "https://good.example.com/feed.xml": { ok: true, title: "Good news", items: 20, error: null },
  "https://also.example.com/rss": { ok: true, title: "Also good", items: 4, error: null },
  "https://down.example.com/feed.xml": { ok: false, title: null, items: 0, error: "This address did not answer" },
  "https://page.example.com/": { ok: false, title: null, items: 0, error: "This is a web page, not a feed" },
  "https://empty.example.com/feed.xml": { ok: false, title: null, items: 0, error: "This feed has no items" },
};
const PRESETS = [
  { id: "ai", label: "AI news", feeds: ["https://good.example.com/feed.xml", "https://also.example.com/rss"] },
  { id: "science", label: "Science", feeds: ["https://also.example.com/rss"] },
];
let keys = false;
let busy = false;

const sha256 = (text) => createHash("sha256").update(text, "utf8").digest("hex");

function seed() {
  return {
    id: ID,
    run_id: "fedcba9876543210fedcba9876543210",
    title: "AI Weekly",
    issue_date: "2026-11-03",
    text: '# AI Weekly\n\n<img src=x onerror="alert(1)"><script>alert(2)</script>\n\nSecond paragraph.',
    flags: [
      {
        id: "s1-c1",
        story_id: "s1",
        text: "The model scored 99% on every benchmark.",
        status: "unsupported",
        reason: "The source gives 91%.",
        source_url: "https://example.com/story-1",
      },
    ],
    report: {
      ok: false,
      failed_sources: [{ url: "https://down.example.com/feed.xml", reason: "HTTP 503" }],
      dropped_stories: [],
      dropped_claims: [],
      flagged_claims: [],
      editor_error: null,
      editor_rejections: [],
      steps: [{ step: "summary", tier: "fast", model: "fast-model", prompt_tokens: 10, completion_tokens: 5, cost: 0.001 }],
      total_cost: 0.001,
      cost_complete: true,
      failed_steps: [],
      voice: { enabled: false, skipped: "YOURVOICE_API_KEY is not set", voiced: [], kept_neutral: [], ok: true },
    },
    approval: null,
    created_at: "2026-11-03T10:00:00.000000+00:00",
    updated_at: "2026-11-03T10:00:00.000000+00:00",
  };
}

let draft = seed();
const approved = (d) => Boolean(d.approval) && d.approval.sha256 === sha256(d.text);
const view = (d) => ({ ...d, approved: approved(d) });

function send(res, status, body) {
  res.writeHead(status, { "Content-Type": "application/json" });
  res.end(JSON.stringify(body));
}

async function readJson(req) {
  let raw = "";
  for await (const chunk of req) raw += chunk;
  return JSON.parse(raw);
}

createServer(async (req, res) => {
  const { method, url } = req;
  if (method === "POST" && url === "/__reset") {
    let fields = {};
    try {
      fields = (await readJson(req)) ?? {};
    } catch {
      // No body: the one draft as it started.
    }
    draft = { ...seed(), ...fields };
    keys = false;
    busy = false;
    return send(res, 200, { ok: true });
  }
  if (method === "POST" && url === "/__busy") {
    busy = true;
    return send(res, 200, { ok: true });
  }
  if (method === "POST" && url === "/__keys") {
    keys = true;
    return send(res, 200, { ok: true });
  }
  if (keys) {
    if (method === "GET" && url === "/api/providers") return send(res, 200, PROVIDERS);
    const preset = req.headers["x-model-preset"];
    if (preset === "demo" ? req.headers["x-demo-code"] !== DEMO_CODE : req.headers["x-model-key"] !== OPENROUTER_KEY) {
      return send(res, 401, {
        detail: preset === "demo" ? "X-Demo-Code was not accepted" : "X-Model-Key was refused by the provider",
      });
    }
  } else if (req.headers.authorization !== `Bearer ${TOKEN}`) {
    return send(res, 401, { detail: "missing or wrong token" });
  }
  let body;
  if (method !== "GET") {
    if (!ORIGINS.includes(req.headers.origin)) return send(res, 403, { detail: "this origin may not write" });
    if ((req.headers["content-type"] ?? "").split(";")[0].trim() !== "application/json") {
      return send(res, 415, { detail: "send the body as application/json" });
    }
    try {
      body = await readJson(req);
    } catch {
      return send(res, 400, { detail: "the body is not JSON" });
    }
    if (typeof body !== "object" || body === null || Array.isArray(body)) {
      return send(res, 400, { detail: "the body must be a JSON object" });
    }
  }
  if (keys && method === "GET" && url === "/api/presets") return send(res, 200, PRESETS);
  if (keys && method === "POST" && url === "/api/feeds/check") {
    return send(res, 200, FEEDS[body.url] ?? FEEDS["https://down.example.com/feed.xml"]);
  }
  if (keys && method === "POST" && url === "/api/runs") {
    if (busy) {
      res.setHeader("Retry-After", "1");
      return send(res, 409, { detail: "a run is already going", retry_after: 1 });
    }
    const stamp = new Date().toISOString();
    return send(res, 202, {
      id: "run00000000000000000000000000001",
      status: "queued",
      error: null,
      draft_id: null,
      feeds: body.feeds,
      title: "Newsletter",
      created_at: stamp,
      started_at: null,
      finished_at: null,
      owner_token: "e2e-owner-token",
    });
  }
  if (method === "GET" && url === "/api/drafts") {
    const { text: _t, report: _r, approval: _a, flags, ...summary } = draft;
    return send(res, 200, [{ ...summary, flags: flags.length, approved: approved(draft) }]);
  }
  if (url === `/api/drafts/${ID}`) {
    if (method === "GET") return send(res, 200, view(draft));
    if (method === "PUT") {
      draft = { ...draft, text: body.text, approval: null, updated_at: new Date().toISOString() };
      return send(res, 200, view(draft));
    }
  }
  if (method === "POST" && url === `/api/drafts/${ID}/approve`) {
    const digest = sha256(draft.text);
    if (body.sha256 && body.sha256 !== digest) {
      return send(res, 409, { detail: "the draft changed since it was read; reload it and approve again" });
    }
    draft = { ...draft, approval: { sha256: digest, approved_at: new Date().toISOString() } };
    return send(res, 200, view(draft));
  }
  return send(res, 404, { detail: "not found" });
}).listen(PORT, "127.0.0.1");
