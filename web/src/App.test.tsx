import { render, screen, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { readdirSync, readFileSync } from "node:fs";
import { join } from "node:path";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import App from "./App";
import type { Draft } from "./api";
import { MARKUP, makeDraft } from "./fixtures";

/** A fake API in fetch's place: one draft, the same rules for edit and approve. */
function fakeApi(start: Draft) {
  let draft = { ...start };
  const calls: { method: string; url: string; body?: unknown }[] = [];
  const json = (status: number, body: unknown) =>
    new Response(JSON.stringify(body), { status, headers: { "Content-Type": "application/json" } });
  const fetch = vi.fn(async (url: string, init: RequestInit = {}) => {
    const method = init.method ?? "GET";
    const body = init.body ? JSON.parse(String(init.body)) : undefined;
    calls.push({ method, url, body });
    if (new Headers(init.headers).get("Authorization") !== "Bearer good-token") {
      return json(401, { detail: "missing or wrong token" });
    }
    if (method === "GET" && url === "/api/drafts") {
      const { text: _text, report: _report, approval: _approval, ...summary } = draft;
      return json(200, [{ ...summary, flags: draft.flags.length }]);
    }
    if (method === "GET" && url === `/api/drafts/${draft.id}`) return json(200, draft);
    if (method === "PUT" && url === `/api/drafts/${draft.id}`) {
      draft = { ...draft, text: body.text, approval: null, approved: false };
      return json(200, draft);
    }
    if (method === "POST" && url === `/api/drafts/${draft.id}/approve`) {
      if (body.sha256 !== (await sha(draft.text))) {
        return json(409, { detail: "the draft changed since it was read; reload it and approve again" });
      }
      draft = { ...draft, approval: { sha256: body.sha256, approved_at: "now" }, approved: true };
      return json(200, draft);
    }
    return json(404, { detail: "not found" });
  });
  return {
    fetch,
    calls,
    change(text: string) {
      draft = { ...draft, text, approval: null, approved: false };
    },
  };
}

async function sha(text: string) {
  const digest = await crypto.subtle.digest("SHA-256", new TextEncoder().encode(text));
  return [...new Uint8Array(digest)].map((b) => b.toString(16).padStart(2, "0")).join("");
}

async function openDraft(api = fakeApi(makeDraft())) {
  vi.stubGlobal("fetch", api.fetch);
  const user = userEvent.setup();
  const view = render(<App />);
  await user.type(screen.getByLabelText("API token"), "good-token");
  await user.click(screen.getByRole("button", { name: "Connect" }));
  await user.click(await screen.findByRole("button", { name: /AI Weekly/ }));
  await screen.findByRole("heading", { level: 2, name: "AI Weekly" });
  return { api, user, view };
}

beforeEach(() => vi.unstubAllGlobals());
afterEach(() => vi.unstubAllGlobals());

describe("connecting", () => {
  it("lists the drafts with their flag count and approval", async () => {
    vi.stubGlobal("fetch", fakeApi(makeDraft()).fetch);
    const user = userEvent.setup();
    render(<App />);
    await user.type(screen.getByLabelText("API token"), "good-token");
    await user.click(screen.getByRole("button", { name: "Connect" }));
    const item = await screen.findByRole("button", { name: /AI Weekly/ });
    expect(item).toHaveTextContent("2026-11-03");
    expect(item).toHaveTextContent("2 flags");
    expect(item).toHaveTextContent("Not approved");
  });

  it("says so when the token is wrong", async () => {
    vi.stubGlobal("fetch", fakeApi(makeDraft()).fetch);
    const user = userEvent.setup();
    render(<App />);
    await user.type(screen.getByLabelText("API token"), "bad");
    await user.click(screen.getByRole("button", { name: "Connect" }));
    expect(await screen.findByRole("alert")).toHaveTextContent("missing or wrong token");
    expect(screen.getByLabelText("API token")).toBeInTheDocument();
  });
});

describe("the draft text", () => {
  it("is shown as text, never as HTML", async () => {
    const { view } = await openDraft();
    const text = screen.getByTestId("draft-text");
    expect(text.textContent).toContain(MARKUP);
    expect(view.container.querySelector("img, b, script")).toBeNull();
  });

  it("keeps markup as text in the editor too", async () => {
    const { user, view } = await openDraft();
    await user.click(screen.getByRole("button", { name: "Edit" }));
    expect(screen.getByLabelText("Draft text")).toHaveValue(makeDraft().text);
    expect(view.container.querySelector("img, b, script")).toBeNull();
  });

  it("is never written as HTML anywhere in the page's source", () => {
    const dir = join(__dirname);
    const sources = readdirSync(dir).filter((f) => /\.tsx?$/.test(f) && !f.includes(".test."));
    expect(sources.length).toBeGreaterThan(0);
    for (const file of sources) {
      const source = readFileSync(join(dir, file), "utf-8");
      expect(source, file).not.toMatch(/dangerouslySetInnerHTML|innerHTML|outerHTML|insertAdjacentHTML/);
    }
  });
});

describe("flags", () => {
  it("shows each flagged claim with its status, reason and source", async () => {
    await openDraft();
    const flags = screen.getByRole("region", { name: "Flags (2)" });
    const items = within(flags).getAllByRole("listitem");
    expect(items[0]).toHaveTextContent("The model scored 99% on every benchmark.");
    expect(items[0]).toHaveTextContent("unsupported");
    expect(items[0]).toHaveTextContent("The source gives 91%.");
    expect(within(items[0]).getByRole("link", { name: "https://example.com/story-1" })).toHaveAttribute(
      "href",
      "https://example.com/story-1",
    );
  });

  it("marks a flagged claim by text and an icon the screen reader skips", async () => {
    await openDraft();
    const items = within(screen.getByRole("region", { name: "Flags (2)" })).getAllByRole("listitem");
    expect(items[0]).toHaveTextContent("Flagged: unsupported");
    expect(within(items[0]).getByTestId("flag-icon")).toHaveAttribute("aria-hidden", "true");
  });

  it("does not link a source that is not http or https", async () => {
    await openDraft();
    const items = within(screen.getByRole("region", { name: "Flags (2)" })).getAllByRole("listitem");
    expect(items[1]).toHaveTextContent("javascript:alert(3)");
    expect(within(items[1]).queryByRole("link")).toBeNull();
  });

  it("shows a source that is not a URL as text, without breaking the page", async () => {
    const flag = { ...makeDraft().flags[0], id: "s4-c1", source_url: "not a url" };
    await openDraft(fakeApi(makeDraft({ flags: [flag] })));
    const item = within(screen.getByRole("region", { name: "Flags (1)" })).getByRole("listitem");
    expect(item).toHaveTextContent("Source: not a url");
    expect(within(item).queryByRole("link")).toBeNull();
    expect(screen.getByRole("region", { name: "Run report" })).toBeInTheDocument();
  });

  it("says when there are none", async () => {
    await openDraft(fakeApi(makeDraft({ flags: [] })));
    expect(screen.getByRole("region", { name: "Flags (0)" })).toHaveTextContent("No flagged claims.");
  });
});

describe("the run report", () => {
  it("shows what failed, was dropped or kept neutral, and the cost", async () => {
    await openDraft();
    const report = screen.getByRole("region", { name: "Run report" });
    expect(report).toHaveTextContent("Problems found");
    expect(report).toHaveTextContent("https://down.example.com/feed.xml");
    expect(report).toHaveTextContent("HTTP 503");
    expect(report).toHaveTextContent("Empty story");
    expect(report).toHaveTextContent("Dropped claim text");
    expect(report).toHaveTextContent("intro cited an unknown claim");
    expect(report).toHaveTextContent("edit");
    expect(report).toHaveTextContent("why-s1");
    expect(report).toHaveTextContent("the rewrite changed a number");
    expect(report).toHaveTextContent("fast-model");
    expect(report).toHaveTextContent("$0.0042");
    expect(report).toHaveTextContent("unknown");
    expect(report).toHaveTextContent("1 step cost unknown");
  });

  it("says when nothing went wrong", async () => {
    const clean = {
      ok: true,
      failed_sources: [],
      steps: [],
      total_cost: 0,
      cost_complete: true,
      voice: { enabled: false, skipped: "YOURVOICE_API_KEY is not set", ok: true },
    };
    await openDraft(fakeApi(makeDraft({ flags: [], report: clean })));
    const report = screen.getByRole("region", { name: "Run report" });
    expect(report).toHaveTextContent("Nothing failed");
    expect(report).toHaveTextContent("YOURVOICE_API_KEY is not set");
  });

  it("copes with a report missing its fields", async () => {
    await openDraft(fakeApi(makeDraft({ report: {} })));
    expect(screen.getByRole("region", { name: "Run report" })).toBeInTheDocument();
  });
});

describe("edit and approve", () => {
  it("saves an edit, which clears the approval", async () => {
    const api = fakeApi(makeDraft({ approval: { sha256: "x", approved_at: "then" }, approved: true }));
    const { user } = await openDraft(api);
    expect(screen.getByTestId("approval")).toHaveTextContent("Approved");
    await user.click(screen.getByRole("button", { name: "Edit" }));
    const box = screen.getByLabelText("Draft text");
    await user.clear(box);
    await user.type(box, "Fixed text");
    await user.click(screen.getByRole("button", { name: "Save" }));
    expect(await screen.findByTestId("draft-text")).toHaveTextContent("Fixed text");
    expect(api.calls).toContainEqual({
      method: "PUT",
      url: `/api/drafts/${makeDraft().id}`,
      body: { text: "Fixed text" },
    });
    expect(screen.getByTestId("approval")).toHaveTextContent("Not approved");
  });

  it("cannot approve while an edit is unsaved", async () => {
    const { user } = await openDraft();
    await user.click(screen.getByRole("button", { name: "Edit" }));
    await user.type(screen.getByLabelText("Draft text"), " more");
    expect(screen.getByRole("button", { name: "Approve" })).toBeDisabled();
    await user.click(screen.getByRole("button", { name: "Cancel" }));
    expect(screen.getByTestId("draft-text").textContent).toBe(makeDraft().text);
    expect(screen.getByRole("button", { name: "Approve" })).toBeEnabled();
  });

  it("approves the text that was read", async () => {
    const { api, user } = await openDraft();
    await user.click(screen.getByRole("button", { name: "Approve" }));
    expect(await screen.findByTestId("approval")).toHaveTextContent(/^Approved/);
    const call = api.calls.find((c) => c.method === "POST");
    expect(call?.body).toEqual({ sha256: await sha(makeDraft().text) });
    expect(screen.queryByRole("button", { name: "Approve" })).toBeNull();
  });

  it("refuses to show approved when the draft changed under the reviewer", async () => {
    const { api, user } = await openDraft();
    api.change("someone else's text");
    await user.click(screen.getByRole("button", { name: "Approve" }));
    expect(await screen.findByRole("alert")).toHaveTextContent("the draft changed since it was read");
    expect(screen.getByTestId("approval")).toHaveTextContent("Not approved");
  });

  it("reloads the draft after a conflict, then approves the new text", async () => {
    const { api, user } = await openDraft();
    api.change("someone else's text");
    await user.click(screen.getByRole("button", { name: "Approve" }));
    await screen.findByRole("alert");
    await user.click(screen.getByRole("button", { name: "Reload" }));
    expect(await screen.findByTestId("draft-text")).toHaveTextContent("someone else's text");
    expect(screen.queryByRole("alert")).toBeNull();
    await user.click(screen.getByRole("button", { name: "Approve" }));
    expect(await screen.findByTestId("approval")).toHaveTextContent(/^Approved/);
    expect(api.calls.filter((c) => c.method === "POST").at(-1)?.body).toEqual({
      sha256: await sha("someone else's text"),
    });
  });

  it("shows a failed save and keeps the edit", async () => {
    const api = fakeApi(makeDraft());
    const { user } = await openDraft(api);
    await user.click(screen.getByRole("button", { name: "Edit" }));
    await user.type(screen.getByLabelText("Draft text"), "!");
    api.fetch.mockResolvedValueOnce(
      new Response(JSON.stringify({ detail: "this origin may not write" }), { status: 403 }),
    );
    await user.click(screen.getByRole("button", { name: "Save" }));
    expect(await screen.findByRole("alert")).toHaveTextContent("this origin may not write");
    expect(screen.getByLabelText("Draft text")).toHaveValue(`${makeDraft().text}!`);
  });
});
