import type { Draft, Provider } from "./api";

/** A draft as `GET /api/drafts/{id}` returns it, with markup in the text on purpose. */
export const MARKUP = '<img src=x onerror="alert(1)"><b>bold</b><script>alert(2)</script>';

export function makeDraft(overrides: Partial<Draft> = {}): Draft {
  return {
    id: "0123456789abcdef0123456789abcdef",
    run_id: "fedcba9876543210fedcba9876543210",
    title: "AI Weekly",
    issue_date: "2026-11-03",
    text: `# AI Weekly\n\n${MARKUP}\n\nSecond paragraph.`,
    flags: [
      {
        id: "s1-c1",
        story_id: "s1",
        text: "The model scored 99% on every benchmark.",
        status: "unsupported",
        reason: "The source gives 91%.",
        source_url: "https://example.com/story-1",
      },
      {
        id: "s2-c1",
        story_id: "s2",
        text: "A claim with a bad link.",
        status: "unchecked",
        reason: "The checker failed.",
        source_url: "javascript:alert(3)",
      },
    ],
    report: {
      ok: false,
      failed_sources: [{ url: "https://down.example.com/feed.xml", reason: "HTTP 503" }],
      dropped_stories: [
        { story_id: "s3", title: "Empty story", link: "https://example.com/s3", reason: "no claim to show" },
      ],
      dropped_claims: [
        { id: "s1-c2", story_id: "s1", text: "Dropped claim text", passage: "", reason: "contradicted" },
      ],
      flagged_claims: [],
      editor_error: null,
      editor_rejections: ["intro cited an unknown claim"],
      steps: [
        { step: "summary", tier: "fast", model: "fast-model", prompt_tokens: 1200, completion_tokens: 300, cost: 0.0042 },
        { step: "check", tier: "reasoning", model: "big-model", prompt_tokens: 900, completion_tokens: 50, cost: null },
      ],
      total_cost: 0.0042,
      cost_complete: false,
      failed_steps: ["edit"],
      voice: {
        enabled: true,
        skipped: null,
        voiced: ["intro"],
        kept_neutral: [{ id: "why-s1", reason: "the rewrite changed a number" }],
        ok: false,
      },
    },
    approval: null,
    approved: false,
    created_at: "2026-11-03T10:00:00.000000+00:00",
    updated_at: "2026-11-03T10:00:00.000000+00:00",
    ...overrides,
  };
}

/** `GET /api/providers` as plan 08's `providers.json` serves it: no secrets, Synthetic without embeddings. */
export const PROVIDERS: Provider[] = [
  {
    id: "demo",
    label: "Demo code",
    kind: "demo",
    base_url: "https://demo.example.com/v1",
    fast_model: "demo-fast",
    reasoning_model: "demo-reasoning",
    embedding_model: "demo-embedding",
  },
  {
    id: "openai",
    label: "OpenAI",
    kind: "key",
    base_url: "https://api.openai.com/v1",
    fast_model: "gpt-5-nano",
    reasoning_model: "gpt-5-mini",
    embedding_model: "text-embedding-3-small",
  },
  {
    id: "openrouter",
    label: "OpenRouter",
    kind: "key",
    base_url: "https://openrouter.ai/api/v1",
    fast_model: "qwen/qwen3.7-flash",
    reasoning_model: "qwen/qwen3.7-plus",
    embedding_model: "thenlper/gte-base",
  },
  {
    id: "alibaba",
    label: "Alibaba",
    kind: "key",
    base_url: "https://dashscope-intl.aliyuncs.com/compatible-mode/v1",
    fast_model: "qwen-flash",
    reasoning_model: "qwen-plus",
    embedding_model: "text-embedding-v4",
  },
  {
    id: "synthetic",
    label: "Synthetic",
    kind: "key",
    base_url: "https://api.synthetic.new/openai/v1",
    fast_model: "hf:zai-org/GLM-5.3-Flash",
    reasoning_model: "hf:zai-org/GLM-5.3",
    embedding_model: null,
  },
];
