import { render, screen, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import App from "./App";
import type { Draft } from "./api";
import { PROVIDERS, makeDraft } from "./fixtures";
import { STORAGE_KEY } from "./modelAccess";

const KEY = "sk-or-MARKER-7f3a9c";
const CODE = "demo-MARKER-51b2";

type Call = { method: string; url: string; headers: Headers; body: string };

/**
 * A fake of a server in demo mode: it lists providers, and answers draft calls only for the
 * good OpenRouter key or the good demo code, with a 401 that names the header, never the value.
 */
function keyApi(draft: Draft = makeDraft()) {
  const calls: Call[] = [];
  const json = (status: number, body: unknown) =>
    new Response(JSON.stringify(body), { status, headers: { "Content-Type": "application/json" } });
  let current = { ...draft };
  const fetch = vi.fn(async (url: string, init: RequestInit = {}) => {
    const method = init.method ?? "GET";
    const headers = new Headers(init.headers);
    calls.push({ method, url, headers, body: String(init.body ?? "") });
    if (method === "GET" && url === "/api/providers") return json(200, PROVIDERS);
    const preset = headers.get("X-Model-Preset");
    const ok =
      (preset === "openrouter" && headers.get("X-Model-Key") === KEY) ||
      (preset === "demo" && headers.get("X-Demo-Code") === CODE);
    if (!ok) {
      return json(401, {
        detail: preset === "demo" ? "X-Demo-Code was not accepted" : "X-Model-Key was refused by the provider",
      });
    }
    if (method === "GET" && url === "/api/drafts") {
      const { text: _t, report: _r, approval: _a, ...summary } = current;
      return json(200, [{ ...summary, flags: current.flags.length }]);
    }
    if (method === "GET" && url === `/api/drafts/${current.id}`) return json(200, current);
    if (method === "PUT" && url === `/api/drafts/${current.id}`) {
      current = { ...current, text: JSON.parse(String(init.body)).text, approval: null, approved: false };
      return json(200, current);
    }
    return json(404, { detail: "not found" });
  });
  return { fetch, calls };
}

function stored(preset: string, secret: string, remember: boolean) {
  (remember ? localStorage : sessionStorage).setItem(STORAGE_KEY, JSON.stringify({ preset, secret, remember }));
}

async function start(api = keyApi()) {
  vi.stubGlobal("fetch", api.fetch);
  const user = userEvent.setup();
  const view = render(<App />);
  return { api, user, view };
}

async function saveKey(user: ReturnType<typeof userEvent.setup>, preset: string, secret: string) {
  const panel = await screen.findByRole("region", { name: "Model key" });
  await user.selectOptions(within(panel).getByLabelText("Provider"), preset);
  const field = within(panel).getByLabelText(preset === "demo" ? "Demo code" : "API key", { selector: "input" });
  await user.clear(field);
  await user.type(field, secret);
  await user.click(within(panel).getByRole("button", { name: "Save" }));
}

beforeEach(() => {
  vi.unstubAllGlobals();
  sessionStorage.clear();
  localStorage.clear();
});
afterEach(() => vi.unstubAllGlobals());

describe("the key panel on a demo-mode server", () => {
  it("opens on the first visit, with the provider list, and no token field", async () => {
    await start();
    const panel = await screen.findByRole("region", { name: "Model key" });
    const options = within(within(panel).getByLabelText("Provider")).getAllByRole("option");
    expect(options.map((o) => o.textContent)).toEqual(["Demo code", "OpenAI", "OpenRouter", "Alibaba", "Synthetic"]);
    expect(screen.queryByLabelText("API token")).toBeNull();
  });

  it("saves a key, lists the drafts with it and closes the panel", async () => {
    const { user } = await start();
    await saveKey(user, "openrouter", KEY);
    expect(await screen.findByRole("button", { name: /AI Weekly/ })).toBeInTheDocument();
    expect(screen.queryByRole("region", { name: "Model key" })).toBeNull();
    expect(JSON.parse(sessionStorage.getItem(STORAGE_KEY)!)).toMatchObject({ preset: "openrouter", secret: KEY });
    expect(localStorage.getItem(STORAGE_KEY)).toBeNull();
  });

  it("with Remember, keeps the key in local storage only", async () => {
    const { user } = await start();
    const panel = await screen.findByRole("region", { name: "Model key" });
    await user.click(within(panel).getByLabelText("Remember on this device"));
    await saveKey(user, "demo", CODE);
    await screen.findByRole("button", { name: /AI Weekly/ });
    expect(JSON.parse(localStorage.getItem(STORAGE_KEY)!)).toMatchObject({ preset: "demo", secret: CODE, remember: true });
    expect(sessionStorage.getItem(STORAGE_KEY)).toBeNull();
  });

  it("uses saved credentials without opening the panel", async () => {
    stored("demo", CODE, true);
    const { api } = await start();
    expect(await screen.findByRole("button", { name: /AI Weekly/ })).toBeInTheDocument();
    expect(screen.queryByRole("region", { name: "Model key" })).toBeNull();
    const list = api.calls.find((c) => c.url === "/api/drafts")!;
    expect(list.headers.get("X-Demo-Code")).toBe(CODE);
  });

  it("sends the secret only in its headers, and never renders or logs it", async () => {
    const spies = (["log", "info", "warn", "error", "debug"] as const).map((level) =>
      vi.spyOn(console, level).mockImplementation(() => {}),
    );
    try {
      const { api, user, view } = await start();
      await saveKey(user, "openrouter", KEY);
      await user.click(await screen.findByRole("button", { name: /AI Weekly/ }));
      await screen.findByRole("heading", { level: 2, name: "AI Weekly" });
      await user.click(screen.getByRole("button", { name: "Edit" }));
      await user.type(screen.getByLabelText("Draft text"), "!");
      await user.click(screen.getByRole("button", { name: "Save" }));
      await user.click(await screen.findByRole("button", { name: "Done editing" }));
      expect(await screen.findByTestId("draft-text")).toHaveTextContent("Second paragraph.!");
      await user.click(screen.getByRole("button", { name: "Key" }));
      await screen.findByRole("region", { name: "Model key" });

      const withKey = api.calls.filter((c) => c.url !== "/api/providers");
      // The form asks for the presets once the drafts are listed: it carries the key too.
      expect(withKey.map((c) => `${c.method} ${c.url}`)).toEqual([
        "GET /api/drafts",
        "GET /api/presets",
        "GET /api/drafts/0123456789abcdef0123456789abcdef",
        "PUT /api/drafts/0123456789abcdef0123456789abcdef",
      ]);
      for (const call of api.calls) {
        expect(call.url).not.toContain(KEY);
        expect(call.body).not.toContain(KEY);
        const other = [...call.headers].filter(([name]) => name !== "x-model-key");
        expect(JSON.stringify(other)).not.toContain(KEY);
      }
      for (const call of withKey) expect(call.headers.get("X-Model-Key")).toBe(KEY);
      expect(view.container.innerHTML).not.toContain(KEY);
      expect(document.body.innerHTML).not.toContain(KEY);
      for (const spy of spies) expect(JSON.stringify(spy.mock.calls)).not.toContain(KEY);
    } finally {
      for (const spy of spies) spy.mockRestore();
    }
  });

  it("opens from a Key button on the draft list and on a draft", async () => {
    stored("openrouter", KEY, false);
    const { user } = await start();
    await screen.findByRole("button", { name: /AI Weekly/ });
    await user.click(screen.getByRole("button", { name: "Key" }));
    const panel = screen.getByRole("region", { name: "Model key" });
    expect(within(panel).getByRole("heading", { name: "Model key" })).toHaveFocus();
    expect(within(panel).getByLabelText("Provider")).toHaveValue("openrouter");
    await user.click(within(panel).getByRole("button", { name: "Close" }));
    expect(screen.queryByRole("region", { name: "Model key" })).toBeNull();
    await user.click(screen.getByRole("button", { name: /AI Weekly/ }));
    await screen.findByRole("heading", { level: 2, name: "AI Weekly" });
    await user.click(screen.getByRole("button", { name: "Key" }));
    expect(screen.getByRole("region", { name: "Model key" })).toBeInTheDocument();
    expect(screen.getByRole("heading", { level: 2, name: "AI Weekly" })).toBeInTheDocument();
  });

  it("a 401 shows the API's header-only message and reopens the panel", async () => {
    stored("openrouter", "sk-wrong", false);
    await start();
    const panel = await screen.findByRole("region", { name: "Model key" });
    expect(within(panel).getByRole("alert")).toHaveTextContent("X-Model-Key was refused by the provider");
  });

  it("a 401 on a later call reopens the panel over the draft", async () => {
    stored("openrouter", KEY, false);
    const api = keyApi();
    const { user } = await start(api);
    await user.click(await screen.findByRole("button", { name: /AI Weekly/ }));
    await screen.findByRole("heading", { level: 2, name: "AI Weekly" });
    api.fetch.mockResolvedValueOnce(
      new Response(JSON.stringify({ detail: "X-Model-Key was refused by the provider" }), { status: 401 }),
    );
    await user.click(screen.getByRole("button", { name: "Edit" }));
    await user.type(screen.getByLabelText("Draft text"), "!");
    await user.click(screen.getByRole("button", { name: "Save" }));
    const panel = await screen.findByRole("region", { name: "Model key" });
    expect(within(panel).getByRole("alert")).toHaveTextContent("X-Model-Key was refused by the provider");
  });

  it("a new key after a 401 lists the drafts and drops the message", async () => {
    stored("openrouter", "sk-wrong", false);
    const { user } = await start();
    await screen.findByRole("alert");
    await saveKey(user, "openrouter", KEY);
    expect(await screen.findByRole("button", { name: /AI Weekly/ })).toBeInTheDocument();
    expect(screen.queryByRole("region", { name: "Model key" })).toBeNull();
    expect(screen.queryByRole("alert")).toBeNull();
  });

  it("Clear wipes both stores and leaves the panel open", async () => {
    stored("openrouter", KEY, true);
    sessionStorage.setItem(STORAGE_KEY, JSON.stringify({ preset: "demo", secret: CODE }));
    const { user } = await start();
    await screen.findByRole("button", { name: /AI Weekly/ });
    await user.click(screen.getByRole("button", { name: "Key" }));
    await user.click(screen.getByRole("button", { name: "Clear" }));
    expect(sessionStorage.getItem(STORAGE_KEY)).toBeNull();
    expect(localStorage.getItem(STORAGE_KEY)).toBeNull();
    expect(screen.getByRole("region", { name: "Model key" })).toBeInTheDocument();
    expect(screen.queryByRole("navigation", { name: "Drafts" })).toBeNull();
  });

  it("offers Try again when the list fails for another reason, with the saved key", async () => {
    stored("openrouter", KEY, false);
    const api = keyApi();
    const answer = (status: number, body: unknown) => new Response(JSON.stringify(body), { status });
    api.fetch.mockImplementationOnce(async () => answer(200, PROVIDERS));
    api.fetch.mockImplementationOnce(async () => answer(503, { detail: "the API is busy" }));
    const { user } = await start(api);
    expect(await screen.findByRole("alert")).toHaveTextContent("Could not load the drafts: the API is busy");
    expect(screen.queryByRole("region", { name: "Model key" })).toBeNull();
    await user.click(screen.getByRole("button", { name: "Try again" }));
    expect(await screen.findByRole("button", { name: /AI Weekly/ })).toBeInTheDocument();
    expect(api.calls.filter((c) => c.url === "/api/drafts").at(-1)!.headers.get("X-Model-Key")).toBe(KEY);
  });

  it("says how to start when the panel is closed with no key", async () => {
    const { user } = await start();
    await user.click(within(await screen.findByRole("region", { name: "Model key" })).getByRole("button", { name: "Close" }));
    expect(screen.getByText("Add a key or a demo code with the Key button to start.")).toBeInTheDocument();
  });
});

describe("a server-key server", () => {
  it("keeps the token field and shows no key panel or Key button", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn(async () => new Response(JSON.stringify({ detail: "not found" }), { status: 404 })),
    );
    render(<App />);
    expect(await screen.findByLabelText("API token")).toBeInTheDocument();
    expect(screen.queryByRole("region", { name: "Model key" })).toBeNull();
    expect(screen.queryByRole("button", { name: "Key" })).toBeNull();
  });
});
