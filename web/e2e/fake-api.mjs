// A fake of `newsletter serve` for the browser test: the same draft endpoints and the
// same write rules (token, then Origin, then a JSON object body), drafts kept in memory.
// `POST /__reset` puts the one draft back as it started.
import { createHash } from "node:crypto";
import { createServer } from "node:http";

const PORT = Number(process.env.FAKE_API_PORT ?? 8787);
const TOKEN = process.env.FAKE_API_TOKEN ?? "e2e-token";
const ORIGINS = (process.env.FAKE_API_ORIGINS ?? "").split(",").filter(Boolean);
const ID = "0123456789abcdef0123456789abcdef";

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
      steps: [{ step: "summary", tier: "fast", model: "fast-model", input_tokens: 10, output_tokens: 5, cost: 0.001 }],
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
    draft = seed();
    return send(res, 200, { ok: true });
  }
  if (req.headers.authorization !== `Bearer ${TOKEN}`) {
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
