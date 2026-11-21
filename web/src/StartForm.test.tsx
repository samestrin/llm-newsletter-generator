import { act, render, screen, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import App from "./App";
import { MINE_KEY, createApi, type Draft, type Run } from "./api";
import { PROVIDERS, makeDraft } from "./fixtures";
import { STORAGE_KEY } from "./modelAccess";
import { StartForm } from "./StartForm";

const CODE = "demo-MARKER-51b2";
const GOOD = "https://good.example.com/feed.xml";
const GOOD_TOO = "https://also.example.com/rss";
const DOWN = "https://down.example.com/feed.xml";
const PAGE = "https://page.example.com/";
const EMPTY = "https://empty.example.com/feed.xml";
const SLOW = "https://slow.example.com/feed.xml";

/** What plan 08.1's `POST /api/feeds/check` answers for each address. */
const CHECKS: Record<string, { ok: boolean; title: string | null; items: number; error: string | null }> = {
  [GOOD]: { ok: true, title: "Good news", items: 20, error: null },
  [GOOD_TOO]: { ok: true, title: "Also good", items: 1, error: null },
  [DOWN]: { ok: false, title: null, items: 0, error: "This address did not answer" },
  [PAGE]: { ok: false, title: null, items: 0, error: "This is a web page, not a feed" },
  [EMPTY]: { ok: false, title: null, items: 0, error: "This feed has no items" },
};

const PRESETS = [
  { id: "ai", label: "AI news", feeds: [GOOD, GOOD_TOO, EMPTY] },
  { id: "science", label: "Science", feeds: [GOOD_TOO] },
];

const SEEDED = makeDraft({ id: "seed0000000000000000000000000000", title: "Open models week" });
const MINE = makeDraft({ id: "mine0000000000000000000000000000", title: "My own issue" });

function run(overrides: Partial<Run> = {}): Run {
  return {
    id: "run00000000000000000000000000001",
    status: "queued",
    error: null,
    draft_id: null,
    feeds: [GOOD],
    title: "Newsletter",
    created_at: "2026-11-21T10:00:00+00:00",
    started_at: null,
    finished_at: null,
    ...overrides,
  };
}

type Call = { method: string; url: string; headers: Headers; body: Record<string, unknown> | undefined };

/** A fake of a demo-mode server with plan 08.1's presets, feed check and run start. */
function demoApi(options: { busy?: number; drafts?: Draft[]; ownerToken?: string } = {}) {
  const calls: Call[] = [];
  const json = (status: number, body: unknown, headers: Record<string, string> = {}) =>
    new Response(JSON.stringify(body), { status, headers: { "Content-Type": "application/json", ...headers } });
  let busy = options.busy ?? 0;
  let release: (() => void) | null = null;
  const drafts = options.drafts ?? [SEEDED];
  const fetch = vi.fn(async (url: string, init: RequestInit = {}) => {
    const method = init.method ?? "GET";
    const headers = new Headers(init.headers);
    const body = init.body ? JSON.parse(String(init.body)) : undefined;
    calls.push({ method, url, headers, body });
    if (method === "GET" && url === "/api/providers") return json(200, PROVIDERS);
    if (headers.get("X-Demo-Code") !== CODE) return json(401, { detail: "X-Demo-Code was not accepted" });
    if (method === "GET" && url === "/api/presets") return json(200, PRESETS);
    if (method === "POST" && url === "/api/feeds/check") {
      if (body.url === SLOW) {
        await new Promise<void>((resolve) => (release = resolve));
        return json(200, CHECKS[GOOD]);
      }
      return json(200, CHECKS[body.url] ?? { ok: false, title: null, items: 0, error: "This address did not answer" });
    }
    if (method === "POST" && url === "/api/runs") {
      if (busy > 0) {
        busy -= 1;
        return json(409, { detail: "a run is already going", retry_after: 2 }, { "Retry-After": "2" });
      }
      return json(202, { ...run({ feeds: body.feeds }), owner_token: options.ownerToken });
    }
    if (method === "GET" && url === "/api/drafts") {
      return json(
        200,
        drafts.map(({ text: _t, report: _r, approval: _a, ...d }) => ({ ...d, flags: d.flags.length })),
      );
    }
    const found = drafts.find((d) => url === `/api/drafts/${d.id}`);
    if (method === "GET" && found) return json(200, found);
    return json(404, { detail: "not found" });
  });
  return { fetch, calls, release: () => release?.() };
}

function renderForm(api = demoApi(), onExample?: () => void) {
  vi.stubGlobal("fetch", api.fetch);
  const onStarted = vi.fn();
  const user = userEvent.setup({ advanceTimers: vi.advanceTimersByTime });
  render(
    <StartForm
      api={createApi({ model: { preset: "demo", secret: CODE } })}
      onStarted={onStarted}
      onExample={onExample}
    />,
  );
  return { api, user, onStarted };
}

const row = (n: number) => screen.getByRole("group", { name: `Feed ${n}` });
const field = (n: number) => within(row(n)).getByLabelText(`Feed ${n} address`);
const make = () => screen.getByRole("button", { name: "Make my newsletter" });

/** Type an address into a row and leave it, as a visitor does. */
async function enter(user: ReturnType<typeof userEvent.setup>, n: number, url: string) {
  await user.clear(field(n));
  await user.type(field(n), url);
  await user.tab();
}

beforeEach(() => {
  vi.useFakeTimers({ shouldAdvanceTime: true });
  vi.unstubAllGlobals();
  sessionStorage.clear();
  localStorage.clear();
});
afterEach(() => {
  vi.unstubAllGlobals();
  vi.useRealTimers();
});

describe("the start form", () => {
  it("says what the tool does and starts with one empty row", () => {
    renderForm();
    expect(screen.getByText(/every claim links to its source/i)).toBeInTheDocument();
    expect(screen.getByText(/checked against it before you see it/i)).toBeInTheDocument();
    expect(screen.getByText(/nothing is sent/i)).toBeInTheDocument();
    expect(screen.getAllByRole("group", { name: /^Feed \d$/ })).toHaveLength(1);
    expect(field(1)).toHaveValue("");
  });

  it("adds rows up to 6 and removes the row asked for", async () => {
    const { user } = renderForm();
    const add = screen.getByRole("button", { name: "Add a feed" });
    for (let n = 0; n < 5; n++) await user.click(add);
    expect(screen.getAllByRole("group", { name: /^Feed \d$/ })).toHaveLength(6);
    expect(add).toBeDisabled();
    await user.type(field(2), GOOD);
    await user.type(field(3), DOWN);
    await user.click(within(row(2)).getByRole("button", { name: "Remove feed 2" }));
    expect(screen.getAllByRole("group", { name: /^Feed \d$/ })).toHaveLength(5);
    expect(field(2)).toHaveValue(DOWN);
    expect(add).toBeEnabled();
  });

  it("removing the only row leaves one empty row", async () => {
    const { user } = renderForm();
    await user.type(field(1), GOOD);
    await user.click(screen.getByRole("button", { name: "Remove feed 1" }));
    expect(screen.getAllByRole("group", { name: /^Feed \d$/ })).toHaveLength(1);
    expect(field(1)).toHaveValue("");
  });

  it("a preset fills the rows, checks them, and the rows stay editable", async () => {
    const { user } = renderForm();
    await user.click(await screen.findByRole("button", { name: "AI news" }));
    expect(screen.getAllByRole("group", { name: /^Feed \d$/ })).toHaveLength(3);
    expect([field(1), field(2), field(3)].map((f) => (f as HTMLInputElement).value)).toEqual([GOOD, GOOD_TOO, EMPTY]);
    expect(await within(row(1)).findByText("OK, 20 items")).toBeInTheDocument();
    expect(await within(row(2)).findByText("OK, 1 item")).toBeInTheDocument();
    expect(await within(row(3)).findByText("This feed has no items")).toBeInTheDocument();
    await enter(user, 3, DOWN);
    expect(field(3)).toHaveValue(DOWN);
    expect(await within(row(3)).findByText("This address did not answer")).toBeInTheDocument();
  });

  it("shows no presets when the server has none", async () => {
    const api = demoApi();
    const inner = api.fetch.getMockImplementation()!;
    api.fetch.mockImplementation(async (url: string, init?: RequestInit) =>
      url === "/api/presets" ? new Response("{}", { status: 404 }) : inner(url, init),
    );
    renderForm(api);
    await vi.waitFor(() => expect(api.fetch.mock.calls.some(([url]) => url === "/api/presets")).toBe(true));
    expect(screen.queryByRole("group", { name: "Start from a preset" })).toBeNull();
    expect(field(1)).toBeInTheDocument();
  });

  it("checks a row when the visitor leaves it: Checking…, then the count", async () => {
    const { api, user } = renderForm();
    await enter(user, 1, SLOW);
    const status = within(row(1)).getByText("Checking…").closest("[aria-live]");
    expect(status).toHaveAttribute("aria-live", "polite");
    api.release();
    expect(await within(row(1)).findByText("OK, 20 items")).toBeInTheDocument();
    expect(api.calls.filter((c) => c.url === "/api/feeds/check").map((c) => c.body)).toEqual([{ url: SLOW }]);
  });

  it.each([
    ["an unreachable", DOWN, "This address did not answer"],
    ["a not-a-feed", PAGE, "This is a web page, not a feed"],
    ["an empty-feed", EMPTY, "This feed has no items"],
  ])("%s row shows its error next to it, by text and icon", async (_name, url, error) => {
    const { user } = renderForm();
    await user.click(screen.getByRole("button", { name: "Add a feed" }));
    await enter(user, 2, url);
    const message = await within(row(2)).findByText(error);
    const status = message.closest("[aria-live]")!;
    expect(status).toHaveAttribute("aria-live", "polite");
    expect(status.querySelector("[aria-hidden='true']")?.textContent).toBeTruthy();
    expect(within(row(1)).queryByText(error)).toBeNull();
    expect(field(2)).toHaveAttribute("aria-invalid", "true");
  });

  it("an API error on the check shows its plain message on the row", async () => {
    const api = demoApi();
    const inner = api.fetch.getMockImplementation()!;
    api.fetch.mockImplementation(async (url: string, init?: RequestInit) =>
      url === "/api/feeds/check"
        ? new Response(JSON.stringify({ detail: "too many checks, wait a minute", retry_after: 30 }), { status: 429 })
        : inner(url, init),
    );
    const { user } = renderForm(api);
    await enter(user, 1, GOOD);
    expect(await within(row(1)).findByText("too many checks, wait a minute")).toBeInTheDocument();
  });

  it("keeps Make my newsletter disabled until a row is OK", async () => {
    const { user } = renderForm();
    expect(make()).toBeDisabled();
    await enter(user, 1, DOWN);
    await within(row(1)).findByText("This address did not answer");
    expect(make()).toBeDisabled();
    await enter(user, 1, GOOD);
    await within(row(1)).findByText("OK, 20 items");
    expect(make()).toBeEnabled();
    await user.type(field(1), "x");
    expect(make()).toBeDisabled();
  });

  it("re-checks the rows before the run and sends the OK ones with the focus as `focus`", async () => {
    const { api, user, onStarted } = renderForm(demoApi({ ownerToken: "owner-MARKER" }));
    await enter(user, 1, GOOD);
    await user.click(screen.getByRole("button", { name: "Add a feed" }));
    await enter(user, 2, DOWN);
    await within(row(1)).findByText("OK, 20 items");
    const focus = screen.getByLabelText(/Focus/);
    expect(focus).toHaveAttribute("maxLength", "300");
    expect(screen.getByText(/What should this issue be about or emphasize\?/)).toBeInTheDocument();
    await user.type(focus, "  practical AI tools for small businesses ");
    await user.click(make());
    await vi.waitFor(() => expect(onStarted).toHaveBeenCalled());
    const checks = api.calls.filter((c) => c.url === "/api/feeds/check").map((c) => c.body?.url);
    expect(checks).toEqual([GOOD, DOWN, GOOD, DOWN]);
    const start = api.calls.find((c) => c.url === "/api/runs")!;
    expect(start.method).toBe("POST");
    expect(start.body).toEqual({ feeds: [GOOD], focus: "practical AI tools for small businesses" });
    expect(onStarted.mock.calls[0][0]).toMatchObject({ id: "run00000000000000000000000000001" });
  });

  it("leaves `focus` out when the field is empty", async () => {
    const { api, user, onStarted } = renderForm();
    await enter(user, 1, GOOD);
    await within(row(1)).findByText("OK, 20 items");
    await user.click(make());
    await vi.waitFor(() => expect(onStarted).toHaveBeenCalled());
    expect(api.calls.find((c) => c.url === "/api/runs")!.body).toEqual({ feeds: [GOOD] });
  });

  it("does not start when the re-check finds no OK row", async () => {
    const api = demoApi();
    const { user, onStarted } = renderForm(api);
    await enter(user, 1, GOOD);
    await within(row(1)).findByText("OK, 20 items");
    CHECKS[GOOD].ok = false;
    CHECKS[GOOD].error = "This address did not answer";
    try {
      await user.click(make());
      expect(await within(row(1)).findByText("This address did not answer")).toBeInTheDocument();
    } finally {
      CHECKS[GOOD].ok = true;
      CHECKS[GOOD].error = null;
    }
    expect(api.calls.some((c) => c.url === "/api/runs")).toBe(false);
    expect(onStarted).not.toHaveBeenCalled();
  });

  it("a busy server (409) shows the friendly text and a retry that waits retry_after", async () => {
    const { api, user, onStarted } = renderForm(demoApi({ busy: 1 }));
    await enter(user, 1, GOOD);
    await within(row(1)).findByText("OK, 20 items");
    await user.click(make());
    const alert = await screen.findByRole("alert");
    expect(alert).toHaveTextContent("Someone else is making one right now. Try again in a minute.");
    const retry = within(alert).getByRole("button", { name: "Try again" });
    expect(retry).toBeDisabled();
    expect(alert).toHaveTextContent("You can try again in 2 s.");
    await act(async () => vi.advanceTimersByTime(1000));
    expect(retry).toBeDisabled();
    await act(async () => vi.advanceTimersByTime(1000));
    expect(retry).toBeEnabled();
    await user.click(retry);
    await vi.waitFor(() => expect(onStarted).toHaveBeenCalled());
    expect(api.calls.filter((c) => c.url === "/api/runs")).toHaveLength(2);
    expect(screen.queryByRole("alert")).toBeNull();
  });

  it("shows another error from the run start in plain words", async () => {
    const api = demoApi();
    const inner = api.fetch.getMockImplementation()!;
    api.fetch.mockImplementation(async (url: string, init?: RequestInit) =>
      url === "/api/runs"
        ? new Response(JSON.stringify({ detail: "the run could not start" }), { status: 500 })
        : inner(url, init),
    );
    const { user, onStarted } = renderForm(api);
    await enter(user, 1, GOOD);
    await within(row(1)).findByText("OK, 20 items");
    await user.click(make());
    expect(await screen.findByRole("alert")).toHaveTextContent("Could not start the run: the run could not start");
    expect(onStarted).not.toHaveBeenCalled();
  });

  it("drops a check's answer when the row was edited while it ran", async () => {
    const { api, user } = renderForm();
    await enter(user, 1, SLOW);
    within(row(1)).getByText("Checking…");
    await user.type(field(1), "x");
    await act(async () => {
      api.release();
      await vi.advanceTimersByTimeAsync(50);
    });
    expect(api.calls.filter((c) => c.url === "/api/feeds/check")).toHaveLength(1);
    expect(field(1)).toHaveValue(`${SLOW}x`);
    expect(within(row(1)).queryByText(/^OK/)).toBeNull();
    expect(make()).toBeDisabled();
  });

  it("does not check a blank row when the visitor leaves it", async () => {
    const { api, user } = renderForm();
    await user.click(field(1));
    await user.tab();
    await user.type(field(1), "   ");
    await user.tab();
    await act(async () => vi.advanceTimersByTimeAsync(50));
    expect(api.calls.some((c) => c.url === "/api/feeds/check")).toBe(false);
    expect(within(row(1)).queryByText("Checking…")).toBeNull();
  });

  it("an OK answer with no item count says 0 items", async () => {
    const api = demoApi();
    const inner = api.fetch.getMockImplementation()!;
    api.fetch.mockImplementation(async (url: string, init?: RequestInit) =>
      url === "/api/feeds/check" ? new Response(JSON.stringify({ ok: true }), { status: 200 }) : inner(url, init),
    );
    const { user } = renderForm(api);
    await enter(user, 1, GOOD);
    expect(await within(row(1)).findByText("OK, 0 items")).toBeInTheDocument();
    expect(row(1)).not.toHaveTextContent("undefined");
  });

  it("a preset with no feeds leaves the rows as they are", async () => {
    const api = demoApi();
    const inner = api.fetch.getMockImplementation()!;
    api.fetch.mockImplementation(async (url: string, init?: RequestInit) =>
      url === "/api/presets"
        ? new Response(JSON.stringify([{ id: "none", label: "Nothing", feeds: [] }]), { status: 200 })
        : inner(url, init),
    );
    const { user } = renderForm(api);
    await enter(user, 1, GOOD);
    await within(row(1)).findByText("OK, 20 items");
    await user.click(await screen.findByRole("button", { name: "Nothing" }));
    expect(screen.getAllByRole("group", { name: /^Feed \d$/ })).toHaveLength(1);
    expect(field(1)).toHaveValue(GOOD);
    expect(within(row(1)).getByText("OK, 20 items")).toBeInTheDocument();
  });

  it("a 409 with no retry_after or Retry-After waits 60 s before Try again", async () => {
    const api = demoApi();
    const inner = api.fetch.getMockImplementation()!;
    api.fetch.mockImplementation(async (url: string, init?: RequestInit) =>
      url === "/api/runs"
        ? new Response(JSON.stringify({ detail: "a run is already going" }), { status: 409 })
        : inner(url, init),
    );
    const { user } = renderForm(api);
    await enter(user, 1, GOOD);
    await within(row(1)).findByText("OK, 20 items");
    await user.click(make());
    const alert = await screen.findByRole("alert");
    expect(alert).toHaveTextContent("You can try again in 60 s.");
    const retry = within(alert).getByRole("button", { name: "Try again" });
    expect(retry).toBeDisabled();
    for (let n = 0; n < 59; n++) await act(async () => vi.advanceTimersByTime(1000));
    expect(retry).toBeDisabled();
    await act(async () => vi.advanceTimersByTime(1000));
    expect(retry).toBeEnabled();
  });

  it('"See an example" is a secondary button that calls onExample', async () => {
    const onExample = vi.fn();
    const { user } = renderForm(demoApi(), onExample);
    const example = screen.getByRole("button", { name: "See an example" });
    expect(example).toHaveClass("link-button");
    await user.click(example);
    expect(onExample).toHaveBeenCalledOnce();
  });
});

async function startApp(api = demoApi()) {
  vi.stubGlobal("fetch", api.fetch);
  sessionStorage.setItem(STORAGE_KEY, JSON.stringify({ preset: "demo", secret: CODE, remember: false }));
  const user = userEvent.setup({ advanceTimers: vi.advanceTimersByTime });
  render(<App />);
  await screen.findByRole("button", { name: "Make my newsletter" });
  return { api, user };
}

describe("the page in demo mode", () => {
  it("opens on the form, with the earlier drafts beside it", async () => {
    await startApp();
    const earlier = screen.getByRole("navigation", { name: "Earlier drafts" });
    expect(within(earlier).getByRole("heading", { name: "Earlier drafts" })).toBeVisible();
    expect(await within(earlier).findByRole("button", { name: /Open models week/ })).toBeInTheDocument();
    expect(screen.queryByText("Choose a draft to review.")).toBeNull();
  });

  it('"See an example" opens the seeded draft, not this browser\'s own', async () => {
    sessionStorage.setItem(MINE_KEY, JSON.stringify({ owner_token: "owner-1", runs: [], drafts: [MINE.id] }));
    const { api, user } = await startApp(demoApi({ drafts: [MINE, SEEDED] }));
    await user.click(screen.getByRole("button", { name: "See an example" }));
    expect(await screen.findByRole("heading", { level: 2, name: "Open models week" })).toBeInTheDocument();
    const opened = api.calls.filter((c) => c.url.startsWith("/api/drafts/")).map((c) => c.url);
    expect(opened).toEqual([`/api/drafts/${SEEDED.id}`]);
    await user.click(screen.getByRole("button", { name: "Make a newsletter" }));
    expect(screen.getByRole("button", { name: "Make my newsletter" })).toBeInTheDocument();
  });

  it("keeps the run's owner token and run id in sessionStorage and sends the token back", async () => {
    const { api, user } = await startApp(demoApi({ ownerToken: "owner-MARKER" }));
    await enter(user, 1, GOOD);
    await within(row(1)).findByText("OK, 20 items");
    await user.click(make());
    expect(await screen.findByText("Making your newsletter…")).toBeInTheDocument();
    expect(JSON.parse(sessionStorage.getItem(MINE_KEY)!)).toEqual({
      owner_token: "owner-MARKER",
      runs: ["run00000000000000000000000000001"],
      drafts: [],
    });
    expect(localStorage.getItem(MINE_KEY)).toBeNull();
    expect(document.body.innerHTML).not.toContain("owner-MARKER");
    await user.click(screen.getByRole("button", { name: "Refresh list" }));
    await vi.waitFor(() => expect(api.calls.filter((c) => c.url === "/api/drafts")).toHaveLength(2));
    expect(api.calls.filter((c) => c.url === "/api/drafts").at(-1)!.headers.get("X-Owner-Token")).toBe("owner-MARKER");
  });
});
