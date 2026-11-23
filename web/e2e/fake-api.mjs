// A fake of `newsletter serve` for the browser test: the same draft endpoints and the
// same write rules (token, then Origin, then a JSON object body), drafts kept in memory.
// `POST /__reset` puts the one seeded draft back as it started, with any fields in its JSON
// body (title, text, flags, report...) in place of the seed's, and forgets every run and the
// drafts they made. `POST /__keys` turns on demo mode until the next reset: `GET /api/providers`
// lists providers, and the API takes the demo code or OpenRouter key below as `X-Model-*`
// headers instead of the token. In demo mode it also serves plan 08.1's `GET /api/presets`,
// `POST /api/feeds/check` (answers from FEEDS below) and `POST /api/runs` (409 with
// `retry_after` while `busy`, set by `POST /__busy`; otherwise a queued run with an owner
// token). `GET /api/runs/{id}` moves the run on one step each time it is asked, as the real
// pipeline reports them: each step runs, with its counts, then is done; a feed in RUN_FAILS
// checked well but fails in the run. After the last step the run is done and has made a
// draft with its focus and a flagged claim. `POST /__hold` keeps every run on its first step.
//
// FAKE_API_REMOVE (comma-separated) takes one part of the fake away, to show the flow test
// catches it: `progress-step` (no "Checking claims" step), `feed-check` (no feeds check
// route), `focus` (the run drops the focus) or `email` (no email route).
import { createHash } from "node:crypto";
import { createServer } from "node:http";

const PORT = Number(process.env.FAKE_API_PORT ?? 8787);
const TOKEN = process.env.FAKE_API_TOKEN ?? "e2e-token";
const ORIGINS = (process.env.FAKE_API_ORIGINS ?? "").split(",").filter(Boolean);
const REMOVE = new Set((process.env.FAKE_API_REMOVE ?? "").split(",").filter(Boolean));
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
  "https://flaky.example.com/feed.xml": { ok: true, title: "Flaky", items: 6, error: null },
};
/** Feeds that pass the check but fail when the run fetches them, with the run's reason. */
const RUN_FAILS = { "https://flaky.example.com/feed.xml": "HTTP 503" };
const PRESETS = [
  { id: "ai", label: "AI news", feeds: ["https://good.example.com/feed.xml", "https://also.example.com/rss"] },
  { id: "science", label: "Science", feeds: ["https://also.example.com/rss"] },
];
let keys = false;
let busy = false;
let hold = false;

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

/** Every draft by id, the seed first; a finished run adds its own. */
let drafts = new Map([[ID, seed()]]);
/** Every run by id, with how many times it was asked about. */
let runs = new Map();
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

const counted = (done, total, detail) => ({ done, total, detail });

/** The feeds step's detail, as `_feeds_detail` builds it: `2 of 3 loaded, 1 failed: <url> <reason>`. */
function feedsDetail(feeds) {
  const failed = feeds.filter((url) => RUN_FAILS[url]);
  const detail = `${feeds.length - failed.length} of ${feeds.length} loaded`;
  if (!failed.length) return detail;
  return `${detail}, ${failed.length} failed: ${failed.map((url) => `${url} ${RUN_FAILS[url]}`).join("; ")}`;
}

/** Each step by id and label, with what it shows while running and once done. */
function plan(feeds) {
  const n = feeds.length;
  return [
    ["feeds", "Fetching feeds", counted(n, n, feedsDetail(feeds)), counted(n, n, feedsDetail(feeds))],
    ["articles", "Reading articles", counted(2, 4, "2 of 4 fetched"), counted(4, 4, "4 of 4 fetched")],
    ["grouping", "Grouping and ranking stories", {}, { detail: "3 stories from 4 items" }],
    ["claims", "Writing claims", counted(1, 3, "story 2 of 3"), counted(3, 3, "3 claims from 3 stories")],
    ["check", "Checking claims against sources", counted(2, 3, "2 of 3"), counted(3, 3, "3 claims checked, 1 flagged")],
    ["editor", "Editing the issue", {}, { detail: "3 stories in the issue" }],
    ["done", "Done", {}, {}],
  ].filter(([id]) => !(REMOVE.has("progress-step") && id === "check"));
}

/** The draft a finished run made: its focus, the feed that failed, and one flagged claim. */
function made(run, id) {
  const stamp = new Date().toISOString();
  const flag = {
    id: "s1-c1",
    story_id: "s1",
    text: "The model scored 95 on the reasoning test.",
    status: "unsupported",
    reason: "The source gives 71.",
    source_url: "https://good.example.com/story-1",
    passage: "It scored 71 on the reasoning test.",
    explanation: "The article says 71; the draft said 95.",
  };
  return {
    id,
    run_id: run.id,
    title: "Newsletter",
    issue_date: stamp.slice(0, 10),
    focus: run.focus,
    text: [
      "# Newsletter",
      "",
      "## [Tiny model tops the chart](<https://good.example.com/story-1>)",
      "",
      "- The model scored 95 on the reasoning test. ([source](<https://good.example.com/story-1>))",
      "- It runs on a laptop. ([source](<https://good.example.com/story-1>))",
    ].join("\n"),
    flags: [flag],
    report: {
      ok: false,
      focus: run.focus,
      failed_sources: run.feeds.filter((url) => RUN_FAILS[url]).map((url) => ({ url, reason: RUN_FAILS[url] })),
      dropped_stories: [],
      dropped_claims: [],
      flagged_claims: [flag],
      editor_error: null,
      editor_rejections: [],
      steps: [{ step: "summary", tier: "fast", model: "fast-model", prompt_tokens: 10, completion_tokens: 5, cost: 0.001 }],
      total_cost: 0.001,
      cost_complete: true,
      failed_steps: [],
      voice: { enabled: false, skipped: "YOURVOICE_API_KEY is not set", voiced: [], kept_neutral: [], ok: true },
    },
    approval: null,
    created_at: stamp,
    updated_at: stamp,
  };
}

/** Ask about a run: it moves on one step (unless held), and makes its draft after the last. */
function advance(run) {
  const steps = plan(run.feeds);
  const now = new Date().toISOString();
  if (!hold || run.asked === 0) run.asked += 1;
  const at = run.asked - 1;
  if (at < steps.length) run.started[at] ??= now;
  if (at >= steps.length && !run.draft_id) {
    run.draft_id = run.id.replace(/^run/, "dft");
    drafts.set(run.draft_id, made(run, run.draft_id));
    run.finished_at = now;
  }
  const progress = steps.map(([id, label, running, done], n) => ({
    id,
    label,
    state: n < at ? "done" : n === at ? "running" : "waiting",
    done: null,
    total: null,
    detail: null,
    ...(n < at ? done : n === at ? running : {}),
    started_at: run.started[n] ?? null,
    ended_at: n < at ? (run.started[n + 1] ?? run.finished_at) : null,
  }));
  return {
    id: run.id,
    status: run.draft_id ? "done" : "running",
    error: null,
    draft_id: run.draft_id,
    feeds: run.feeds,
    title: "Newsletter",
    created_at: run.created_at,
    started_at: run.created_at,
    finished_at: run.finished_at,
    progress,
  };
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
    drafts = new Map([[ID, { ...seed(), ...fields }]]);
    runs = new Map();
    keys = false;
    busy = false;
    hold = false;
    return send(res, 200, { ok: true });
  }
  if (method === "POST" && url === "/__hold") {
    hold = true;
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
  if (keys && method === "POST" && url === "/api/feeds/check" && !REMOVE.has("feed-check")) {
    return send(res, 200, FEEDS[body.url] ?? FEEDS["https://down.example.com/feed.xml"]);
  }
  if (keys && method === "POST" && url === "/api/runs") {
    if (busy) {
      res.setHeader("Retry-After", "1");
      return send(res, 409, { detail: "a run is already going", retry_after: 1 });
    }
    const stamp = new Date().toISOString();
    const run = {
      id: `run${String(runs.size + 1).padStart(29, "0")}`,
      feeds: body.feeds,
      focus: REMOVE.has("focus") ? null : (body.focus ?? null),
      created_at: stamp,
      finished_at: null,
      draft_id: null,
      asked: 0,
      started: [],
    };
    runs.set(run.id, run);
    return send(res, 202, {
      id: run.id,
      status: "queued",
      error: null,
      draft_id: null,
      feeds: run.feeds,
      title: "Newsletter",
      created_at: stamp,
      started_at: null,
      finished_at: null,
      owner_token: "e2e-owner-token",
    });
  }
  const asked = keys && method === "GET" && /^\/api\/runs\/([^/]+)$/.exec(url);
  if (asked) {
    const run = runs.get(decodeURIComponent(asked[1]));
    return run ? send(res, 200, advance(run)) : send(res, 404, { detail: "no such run" });
  }
  if (method === "GET" && url === "/api/drafts") {
    // Newest first, as the API lists them: the seed comes last.
    const list = [...drafts.values()].reverse();
    return send(
      res,
      200,
      list.map(({ text: _t, report: _r, approval: _a, flags, ...summary }, n) => ({
        ...summary,
        flags: flags.length,
        approved: approved(list[n]),
      })),
    );
  }
  const route = /^\/api\/drafts\/([^/]+)(\/approve|\/email)?$/.exec(url);
  const draft = route && drafts.get(decodeURIComponent(route[1]));
  if (draft && !route[2]) {
    if (method === "GET") return send(res, 200, view(draft));
    if (method === "PUT") {
      const edited = { ...draft, text: body.text, approval: null, updated_at: new Date().toISOString() };
      drafts.set(draft.id, edited);
      return send(res, 200, view(edited));
    }
  }
  if (draft && method === "POST" && route[2] === "/approve") {
    const digest = sha256(draft.text);
    if (body.sha256 && body.sha256 !== digest) {
      return send(res, 409, { detail: "the draft changed since it was read; reload it and approve again" });
    }
    const signed = { ...draft, approval: { sha256: digest, approved_at: new Date().toISOString() } };
    drafts.set(draft.id, signed);
    return send(res, 200, view(signed));
  }
  if (draft && method === "GET" && route[2] === "/email" && !REMOVE.has("email")) {
    if (!approved(draft)) {
      return send(res, 409, { detail: "the draft's current text is not approved; approve it first" });
    }
    const html = `<!doctype html><html><body><pre>${draft.text.replace(/[&<>]/g, (c) => `&#${c.charCodeAt(0)};`)}</pre></body></html>`;
    return send(res, 200, { html });
  }
  return send(res, 404, { detail: "not found" });
}).listen(PORT, "127.0.0.1");
