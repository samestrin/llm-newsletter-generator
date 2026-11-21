import { act, render, screen, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import App from "./App";
import { RUN_KEY, createApi, type ProgressStep, type Run } from "./api";
import { PROVIDERS, makeDraft } from "./fixtures";
import { STORAGE_KEY } from "./modelAccess";
import { MAX_FAILURES, Progress, elapsed } from "./Progress";

const CODE = "demo-MARKER-51b2";
const RUN_ID = "run00000000000000000000000000001";
const DRAFT = makeDraft({ id: "made0000000000000000000000000000", title: "Made just now" });
const T0 = Date.parse("2026-11-21T10:00:00Z");
const at = (seconds: number) => new Date(T0 + seconds * 1000).toISOString();

const LABELS = [
  ["feeds", "Fetching feeds"],
  ["articles", "Reading articles"],
  ["grouping", "Grouping and ranking stories"],
  ["claims", "Writing claims"],
  ["check", "Checking claims against sources"],
  ["editor", "Editing the issue"],
  ["done", "Done"],
] as const;

type Changes = Partial<Record<(typeof LABELS)[number][0], Partial<ProgressStep>>>;

/** The run as `GET /api/runs/{id}` answers it, every step waiting but those in `changes`. */
function snapshot(changes: Changes = {}, overrides: Partial<Run> = {}): Run {
  return {
    id: RUN_ID,
    status: "running",
    error: null,
    draft_id: null,
    feeds: ["https://good.example.com/feed.xml"],
    title: "Newsletter",
    created_at: at(0),
    started_at: at(0),
    finished_at: null,
    progress: LABELS.map(([id, label]) => ({
      id,
      label,
      state: "waiting",
      done: null,
      total: null,
      detail: null,
      started_at: null,
      ended_at: null,
      ...changes[id],
    })),
    ...overrides,
  };
}

const done = (detail: string | null = null, counts: Partial<ProgressStep> = {}): Partial<ProgressStep> => ({
  state: "done",
  detail,
  started_at: at(0),
  ended_at: at(1),
  ...counts,
});

const FEEDS_DETAIL = "3 of 4 loaded, 1 failed: https://down.example.com/feed.xml HTTP 503";
const FEEDS_DONE = done(FEEDS_DETAIL, { done: 4, total: 4 });

/** A run that goes through every step, one change per poll, then makes DRAFT. */
const SCRIPT: Run[] = [
  snapshot({}, { status: "queued", started_at: null }),
  snapshot({ feeds: { state: "running", done: 1, total: 4, detail: "1 of 4 loaded", started_at: at(0) } }),
  snapshot({
    feeds: FEEDS_DONE,
    articles: { state: "running", done: 12, total: 15, detail: "12 of 15 fetched", started_at: at(4) },
  }),
  snapshot({ feeds: FEEDS_DONE, articles: done("15 of 15 fetched"), grouping: { state: "running", started_at: at(6) } }),
  snapshot({
    feeds: FEEDS_DONE,
    articles: done(),
    grouping: done("8 stories from 15 items"),
    claims: { state: "running", done: 2, total: 8, detail: "story 3 of 8", started_at: at(8) },
  }),
  snapshot({
    feeds: FEEDS_DONE,
    articles: done(),
    grouping: done(),
    claims: done("20 claims from 8 stories"),
    check: { state: "running", done: 24, total: 40, detail: "24 of 40", started_at: at(10) },
  }),
  snapshot({
    feeds: FEEDS_DONE,
    articles: done(),
    grouping: done(),
    claims: done(),
    check: done("40 claims checked, 2 flagged"),
    editor: { state: "running", started_at: at(12) },
  }),
  snapshot(
    {
      feeds: FEEDS_DONE,
      articles: done(),
      grouping: done(),
      claims: done(),
      check: done(),
      editor: done("8 stories in the issue"),
      done: done(),
    },
    { status: "done", draft_id: DRAFT.id, finished_at: at(14) },
  ),
];

type Answer = Run | "fail" | "missing";

/**
 * A fake of a demo-mode server: `GET /api/runs/{id}` answers the next of `answers` at each
 * poll and keeps answering the last one; "fail" is a network failure, "missing" a 404.
 */
function fakeApi(answers: Answer[] = SCRIPT) {
  const calls: { method: string; url: string; headers: Headers }[] = [];
  const json = (status: number, body: unknown) =>
    new Response(JSON.stringify(body), { status, headers: { "Content-Type": "application/json" } });
  let polls = 0;
  const fetch = vi.fn(async (url: string, init: RequestInit = {}) => {
    const method = init.method ?? "GET";
    const headers = new Headers(init.headers);
    calls.push({ method, url, headers });
    if (method === "GET" && url === "/api/providers") return json(200, PROVIDERS);
    if (headers.get("X-Demo-Code") !== CODE) return json(401, { detail: "X-Demo-Code was not accepted" });
    if (method === "GET" && url === `/api/runs/${RUN_ID}`) {
      const answer = answers[Math.min(polls++, answers.length - 1)];
      if (answer === "fail") throw new TypeError("Failed to fetch");
      if (answer === "missing") return json(404, { detail: "no such run" });
      return json(200, answer);
    }
    if (method === "GET" && url === "/api/presets") return json(200, []);
    if (method === "GET" && url === "/api/drafts") {
      const { text: _t, report: _r, approval: _a, ...summary } = DRAFT;
      return json(200, [{ ...summary, flags: DRAFT.flags.length }]);
    }
    if (method === "GET" && url === `/api/drafts/${DRAFT.id}`) return json(200, DRAFT);
    return json(404, { detail: "not found" });
  });
  const polled = () => calls.filter((c) => c.url === `/api/runs/${RUN_ID}`).length;
  return { fetch, calls, polled };
}

function renderProgress(api = fakeApi()) {
  vi.stubGlobal("fetch", api.fetch);
  const onDone = vi.fn();
  const onStartOver = vi.fn();
  const view = render(
    <Progress
      api={createApi({ model: { preset: "demo", secret: CODE } })}
      runId={RUN_ID}
      onDone={onDone}
      onStartOver={onStartOver}
    />,
  );
  return { api, onDone, onStartOver, view };
}

/** Let `ms` pass on the page's clock, answering every poll that falls due on the way. */
const pass = (ms: number) => act(async () => void (await vi.advanceTimersByTimeAsync(ms)));

const steps = () => within(screen.getByRole("list", { name: "Steps" })).getAllByRole("listitem");
const step = (label: string) => steps().find((li) => li.querySelector(".step-label")?.textContent === label)!;
const stateOf = (label: string) => step(label).querySelector(".step-state")!.textContent;
const announced = (container: HTMLElement) => container.querySelector('[aria-live="polite"]')!.textContent;

beforeEach(() => {
  vi.unstubAllGlobals();
  sessionStorage.clear();
  localStorage.clear();
  vi.useFakeTimers();
  vi.setSystemTime(T0);
});
afterEach(() => {
  vi.useRealTimers();
  vi.unstubAllGlobals();
});

describe("elapsed time", () => {
  it.each([
    [0, "0 s"],
    [45_000, "45 s"],
    [80_000, "1 min 20 s"],
    [120_000, "2 min"],
    [3_725_000, "1 h 2 min"],
    [-5_000, "0 s"],
  ])("%i ms reads %s", (ms, text) => {
    expect(elapsed(ms)).toBe(text);
  });
});

describe("live progress", () => {
  it("shows every step and its state in order as the run moves, one poll every 2 s, then opens the draft", async () => {
    const { api, onDone, view } = renderProgress();
    await pass(0);
    expect(screen.getByRole("heading", { name: "Making your newsletter…" })).toBeInTheDocument();
    expect(steps().map((li) => li.querySelector(".step-label")!.textContent)).toEqual(LABELS.map(([, l]) => l));
    expect(steps().map((li) => li.querySelector(".step-state")!.textContent)).toEqual(Array(7).fill("Waiting"));
    expect(api.polled()).toBe(1);

    await pass(2000);
    expect(api.polled()).toBe(2);
    expect(stateOf("Fetching feeds")).toBe("Running");
    expect(screen.getByRole("progressbar", { name: "Fetching feeds: 1 of 4" })).toHaveAttribute("value", "1");
    expect(step("Fetching feeds")).toHaveTextContent("1 of 4 loaded");
    expect(announced(view.container)).toBe("Fetching feeds: running");

    await pass(2000);
    expect(stateOf("Fetching feeds")).toBe("Done");
    expect(step("Fetching feeds")).toHaveTextContent("3 of 4 loaded, 1 failed");
    const failed = within(step("Fetching feeds")).getByRole("list", { name: "Feeds that failed" });
    expect(within(failed).getAllByRole("listitem").map((li) => li.textContent)).toEqual([
      "https://down.example.com/feed.xml HTTP 503",
    ]);
    expect(stateOf("Reading articles")).toBe("Running");
    expect(screen.getByRole("progressbar", { name: "Reading articles: 12 of 15" })).toHaveAttribute("max", "15");
    expect(step("Reading articles")).toHaveTextContent("12 of 15 fetched");

    await pass(2000);
    expect(stateOf("Reading articles")).toBe("Done");
    expect(stateOf("Grouping and ranking stories")).toBe("Running");
    expect(within(step("Grouping and ranking stories")).queryByRole("progressbar")).toBeNull();

    await pass(2000);
    expect(stateOf("Grouping and ranking stories")).toBe("Done");
    expect(stateOf("Writing claims")).toBe("Running");
    expect(step("Writing claims")).toHaveTextContent("story 3 of 8");
    expect(screen.getByRole("progressbar", { name: "Writing claims: 2 of 8" })).toBeInTheDocument();

    await pass(2000);
    expect(stateOf("Checking claims against sources")).toBe("Running");
    expect(screen.getByRole("progressbar", { name: "Checking claims against sources: 24 of 40" })).toBeInTheDocument();
    expect(step("Checking claims against sources")).toHaveTextContent("24 of 40");

    await pass(2000);
    expect(stateOf("Checking claims against sources")).toBe("Done");
    expect(stateOf("Editing the issue")).toBe("Running");
    expect(announced(view.container)).toBe("Editing the issue: running");
    expect(onDone).not.toHaveBeenCalled();

    await pass(2000);
    expect(onDone).toHaveBeenCalledWith(DRAFT.id);
    await pass(10_000);
    expect(api.polled()).toBe(8);
  });

  it("shows how long a running step has run, so a long step is visibly alive", async () => {
    renderProgress(fakeApi([SCRIPT[4]]));
    await pass(0);
    // Writing claims started 8 s after T0; the page's clock is at T0.
    expect(step("Writing claims")).toHaveTextContent("0 s");
    await pass(88_000);
    expect(step("Writing claims")).toHaveTextContent("1 min 20 s");
    expect(within(step("Fetching feeds")).queryByText(/\d+ s$/)).toBeNull();
  });

  it('says "Still working on <step>…" once no step has changed for 30 s, and stops saying it on a change', async () => {
    const api = fakeApi([...Array(16).fill(SCRIPT[4]), SCRIPT[5]]);
    renderProgress(api);
    await pass(0);
    await pass(28_000);
    expect(screen.queryByText(/Still working/)).toBeNull();
    await pass(2000);
    expect(screen.getByText("Still working on Writing claims…")).toBeInTheDocument();
    await pass(2000);
    expect(api.polled()).toBe(17);
    expect(screen.queryByText(/Still working/)).toBeNull();
    expect(stateOf("Checking claims against sources")).toBe("Running");
  });

  it("states each step by text and icon: waiting, running, done, failed and skipped", async () => {
    renderProgress(
      fakeApi([
        snapshot(
          {
            feeds: FEEDS_DONE,
            articles: done(),
            grouping: { state: "failed", detail: "0 stories from 15 items", started_at: at(1), ended_at: at(2) },
            claims: { state: "skipped" },
          },
          { status: "running" },
        ),
      ]),
    );
    await pass(0);
    expect(stateOf("Fetching feeds")).toBe("Done");
    expect(stateOf("Grouping and ranking stories")).toBe("Failed");
    expect(stateOf("Writing claims")).toBe("Skipped");
    expect(stateOf("Done")).toBe("Waiting");
    const icons = steps().map((li) => li.querySelector(".step-icon")!);
    for (const icon of icons) expect(icon).toHaveAttribute("aria-hidden", "true");
    expect(new Set(icons.map((i) => i.textContent)).size).toBe(4);
    expect(step("Grouping and ranking stories")).toHaveClass("step-failed");
  });

  it("on an error run shows the editor's plain message and Start over, and stops polling", async () => {
    sessionStorage.setItem(RUN_KEY, RUN_ID);
    const failed = snapshot(
      { feeds: FEEDS_DONE, articles: { state: "failed", started_at: at(1), ended_at: at(2) } },
      {
        status: "error",
        error: "RuntimeError: provider said 500",
        editor_message: "The model provider stopped answering. Try again in a few minutes.",
      },
    );
    const { api, onStartOver } = renderProgress(fakeApi([SCRIPT[1], failed]));
    const user = userEvent.setup({ advanceTimers: vi.advanceTimersByTime });
    await pass(2000);
    const alert = screen.getByRole("alert");
    expect(alert).toHaveTextContent("The model provider stopped answering. Try again in a few minutes.");
    expect(alert).not.toHaveTextContent("RuntimeError");
    expect(stateOf("Reading articles")).toBe("Failed");
    expect(sessionStorage.getItem(RUN_KEY)).toBeNull();
    await pass(10_000);
    expect(api.polled()).toBe(2);
    await user.click(within(alert).getByRole("button", { name: "Start over" }));
    expect(onStartOver).toHaveBeenCalledTimes(1);
  });

  it("on an error run with no editor message shows the run's own message", async () => {
    renderProgress(fakeApi([snapshot({}, { status: "error", error: "the run took longer than 900 seconds" })]));
    await pass(0);
    expect(screen.getByRole("alert")).toHaveTextContent("The run stopped: the run took longer than 900 seconds");
  });

  it('says "lost contact, retrying" when a poll fails, and carries on when it answers again', async () => {
    const api = fakeApi([SCRIPT[1], "fail", "fail", SCRIPT[2]]);
    renderProgress(api);
    await pass(0);
    await pass(2000);
    expect(screen.getByText("Lost contact, retrying…")).toBeInTheDocument();
    expect(stateOf("Fetching feeds")).toBe("Running");
    await pass(2000);
    expect(screen.getByText("Lost contact, retrying…")).toBeInTheDocument();
    await pass(2000);
    expect(screen.queryByText(/Lost contact/)).toBeNull();
    expect(stateOf("Reading articles")).toBe("Running");
  });

  it("gives up after a cap of failed polls and says so, with Try again and Start over", async () => {
    sessionStorage.setItem(RUN_KEY, RUN_ID);
    const api = fakeApi([...Array(MAX_FAILURES).fill("fail"), SCRIPT[3]]);
    const { onStartOver } = renderProgress(api);
    const user = userEvent.setup({ advanceTimers: vi.advanceTimersByTime });
    await pass(2000 * MAX_FAILURES);
    expect(api.polled()).toBe(MAX_FAILURES);
    const alert = screen.getByRole("alert");
    expect(alert).toHaveTextContent(`Lost contact with the server after ${MAX_FAILURES} tries.`);
    await pass(20_000);
    expect(api.polled()).toBe(MAX_FAILURES);
    // Not over: the run may still finish, so a reload can still resume it.
    expect(sessionStorage.getItem(RUN_KEY)).toBe(RUN_ID);
    await user.click(within(alert).getByRole("button", { name: "Try again" }));
    await pass(0);
    expect(api.polled()).toBe(MAX_FAILURES + 1);
    expect(screen.queryByRole("alert")).toBeNull();
    expect(stateOf("Grouping and ranking stories")).toBe("Running");
    await user.click(screen.getByRole("button", { name: "Start over" }));
    expect(onStartOver).toHaveBeenCalledTimes(1);
    expect(sessionStorage.getItem(RUN_KEY)).toBeNull();
  });

  it("says a missing run is gone and clears the stored entry", async () => {
    sessionStorage.setItem(RUN_KEY, RUN_ID);
    const api = fakeApi(["missing"]);
    renderProgress(api);
    await pass(0);
    expect(screen.getByRole("alert")).toHaveTextContent("This run could not be found.");
    expect(sessionStorage.getItem(RUN_KEY)).toBeNull();
    await pass(10_000);
    expect(api.polled()).toBe(1);
  });

  it("clears the done entry", async () => {
    sessionStorage.setItem(RUN_KEY, RUN_ID);
    renderProgress(fakeApi([SCRIPT.at(-1)!]));
    await pass(0);
    expect(sessionStorage.getItem(RUN_KEY)).toBeNull();
  });

  it("clears its timers when the page leaves", async () => {
    const { api, view } = renderProgress(fakeApi([SCRIPT[4]]));
    await pass(2000);
    expect(api.polled()).toBe(2);
    view.unmount();
    expect(vi.getTimerCount()).toBe(0);
    await pass(10_000);
    expect(api.polled()).toBe(2);
  });

  it("stops at once when the page leaves with a poll still out", async () => {
    let answer: (r: Response) => void = () => {};
    const fetch = vi.fn(() => new Promise<Response>((resolve) => (answer = resolve)));
    vi.stubGlobal("fetch", fetch);
    const view = render(
      <Progress api={createApi({ model: { preset: "demo", secret: CODE } })} runId={RUN_ID} onDone={vi.fn()} onStartOver={vi.fn()} />,
    );
    await pass(0);
    view.unmount();
    answer(new Response(JSON.stringify(SCRIPT[1]), { status: 200 }));
    await pass(10_000);
    expect(fetch).toHaveBeenCalledTimes(1);
    expect(vi.getTimerCount()).toBe(0);
  });
});

describe("the run in the page", () => {
  /** The visitor's demo code, kept for this tab as the key panel does. */
  const keep = () => sessionStorage.setItem(STORAGE_KEY, JSON.stringify({ preset: "demo", secret: CODE, remember: false }));

  it("after a reload mid-run resumes polling the stored run id and shows the same steps, then opens the draft", async () => {
    keep();
    sessionStorage.setItem(RUN_KEY, RUN_ID);
    const api = fakeApi(SCRIPT.slice(4));
    vi.stubGlobal("fetch", api.fetch);
    render(<App />);
    await pass(0);
    expect(screen.getByRole("heading", { name: "Making your newsletter…" })).toBeInTheDocument();
    expect(stateOf("Writing claims")).toBe("Running");
    expect(step("Writing claims")).toHaveTextContent("story 3 of 8");
    expect(api.calls.find((c) => c.url === `/api/runs/${RUN_ID}`)!.headers.get("X-Demo-Code")).toBe(CODE);
    await pass(6000);
    expect(await screen.findByRole("heading", { level: 2, name: "Made just now" })).toBeInTheDocument();
    expect(sessionStorage.getItem(RUN_KEY)).toBeNull();
  });

  it("waits for the key before it resumes the stored run", async () => {
    sessionStorage.setItem(RUN_KEY, RUN_ID);
    const api = fakeApi([SCRIPT[4]]);
    vi.stubGlobal("fetch", api.fetch);
    render(<App />);
    await pass(0);
    expect(api.polled()).toBe(0);
    const user = userEvent.setup({ advanceTimers: vi.advanceTimersByTime });
    const panel = screen.getByRole("region", { name: "Model key" });
    await user.selectOptions(within(panel).getByLabelText("Provider"), "demo");
    await user.type(within(panel).getByLabelText("Demo code", { selector: "input" }), CODE);
    await user.click(within(panel).getByRole("button", { name: "Save" }));
    await pass(0);
    expect(api.polled()).toBe(1);
    expect(stateOf("Writing claims")).toBe("Running");
  });

  it("keeps only the run id in the stored entry, never the code, and Start over clears it", async () => {
    keep();
    sessionStorage.setItem(RUN_KEY, RUN_ID);
    vi.stubGlobal("fetch", fakeApi([snapshot({}, { status: "error", error: "it broke" })]).fetch);
    const user = userEvent.setup({ advanceTimers: vi.advanceTimersByTime });
    render(<App />);
    await pass(0);
    await user.click(within(screen.getByRole("alert")).getByRole("button", { name: "Start over" }));
    expect(sessionStorage.getItem(RUN_KEY)).toBeNull();
    expect(screen.getByRole("button", { name: "Make my newsletter" })).toBeInTheDocument();
  });

  it("starting a run from the form stores its id only and shows its progress", async () => {
    keep();
    const api = fakeApi([SCRIPT[1]]);
    const fetch = vi.fn(async (url: string, init: RequestInit = {}) => {
      if (init.method === "POST" && url === "/api/feeds/check") {
        return new Response(JSON.stringify({ ok: true, title: "Good", items: 3, error: null }), { status: 200 });
      }
      if (init.method === "POST" && url === "/api/runs") {
        return new Response(JSON.stringify({ ...SCRIPT[0], owner_token: "owner-MARKER" }), { status: 202 });
      }
      return api.fetch(url, init);
    });
    vi.stubGlobal("fetch", fetch);
    const user = userEvent.setup({ advanceTimers: vi.advanceTimersByTime });
    render(<App />);
    await pass(0);
    const address = screen.getByLabelText("Feed 1 address");
    await user.type(address, "https://good.example.com/feed.xml");
    await user.tab();
    await pass(0);
    await user.click(screen.getByRole("button", { name: "Make my newsletter" }));
    await pass(0);
    expect(screen.getByRole("heading", { name: "Making your newsletter…" })).toBeInTheDocument();
    expect(sessionStorage.getItem(RUN_KEY)).toBe(RUN_ID);
    expect(stateOf("Fetching feeds")).toBe("Running");
    expect(sessionStorage.getItem(RUN_KEY)).not.toContain(CODE);
  });
});
