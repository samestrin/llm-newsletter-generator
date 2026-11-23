import { afterEach, describe, expect, it, vi } from "vitest";
import { ApiError, createApi, listProviders, sha256Hex } from "./api";
import { PROVIDERS, makeDraft } from "./fixtures";

function reply(status: number, body: unknown) {
  return new Response(JSON.stringify(body), {
    status,
    headers: { "Content-Type": "application/json" },
  });
}

afterEach(() => vi.unstubAllGlobals());

describe("sha256Hex", () => {
  it("hashes UTF-8 text like the API's text_sha256", async () => {
    expect(await sha256Hex("abc")).toBe(
      "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad",
    );
    expect(await sha256Hex("é")).toBe(
      "4a99557e4033c3539de2eb65472017cad5f9557f7a0625a09f1c3f6e2ba69c4c",
    );
  });

  it("refuses with a clear reason when the page has no WebCrypto", async () => {
    vi.stubGlobal("crypto", {});
    const error = await sha256Hex("abc").catch((e) => e);
    expect(error).toBeInstanceOf(ApiError);
    expect(error.message).toBe(
      "approving needs a secure page: open it on https, localhost or 127.0.0.1",
    );
  });
});

describe("createApi", () => {
  it("sends the token on every read", async () => {
    const fetch = vi.fn().mockResolvedValue(reply(200, []));
    vi.stubGlobal("fetch", fetch);
    await createApi({ token: "s3cret" }).listDrafts();
    const [url, init] = fetch.mock.calls[0];
    expect(url).toBe("/api/drafts");
    expect(new Headers(init.headers).get("Authorization")).toBe("Bearer s3cret");
  });

  it("saves the text as a JSON object", async () => {
    const draft = makeDraft();
    const fetch = vi.fn().mockResolvedValue(reply(200, draft));
    vi.stubGlobal("fetch", fetch);
    await createApi({ token: "t" }).saveText(draft.id, "new text");
    const [url, init] = fetch.mock.calls[0];
    expect(url).toBe(`/api/drafts/${draft.id}`);
    expect(init.method).toBe("PUT");
    expect(new Headers(init.headers).get("Content-Type")).toBe("application/json");
    expect(JSON.parse(init.body)).toEqual({ text: "new text" });
  });

  it("approves with the sha256 of the text the reviewer read", async () => {
    const draft = makeDraft({ text: "abc" });
    const fetch = vi.fn().mockResolvedValue(reply(200, { ...draft, approved: true }));
    vi.stubGlobal("fetch", fetch);
    await createApi({ token: "t" }).approve(draft);
    const [url, init] = fetch.mock.calls[0];
    expect(url).toBe(`/api/drafts/${draft.id}/approve`);
    expect(init.method).toBe("POST");
    expect(JSON.parse(init.body)).toEqual({
      sha256: "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad",
    });
  });

  it("reads the email preview with a GET and no body", async () => {
    const draft = makeDraft();
    const fetch = vi.fn().mockResolvedValue(reply(200, { html: "<p>hi</p>" }));
    vi.stubGlobal("fetch", fetch);
    expect(await createApi({ token: "t" }).email(draft.id)).toEqual({ html: "<p>hi</p>" });
    const [url, init] = fetch.mock.calls[0];
    expect(url).toBe(`/api/drafts/${draft.id}/email`);
    expect(init.method).toBe("GET");
    expect(init.body).toBeUndefined();
  });

  it("raises the server's reason with the status", async () => {
    vi.stubGlobal("fetch", vi.fn().mockResolvedValue(reply(409, { detail: "the draft changed" })));
    const error = await createApi({ token: "t" }).approve(makeDraft()).catch((e) => e);
    expect(error).toBeInstanceOf(ApiError);
    expect(error.status).toBe(409);
    expect(error.message).toBe("the draft changed");
  });

  it("names the status when the body has no reason", async () => {
    vi.stubGlobal("fetch", vi.fn().mockResolvedValue(new Response("<html>", { status: 502 })));
    const error = await createApi({ token: "t" }).listDrafts().catch((e) => e);
    expect(error.status).toBe(502);
    expect(error.message).toBe("the API answered 502");
  });

  it("reports a network failure as an error, not a crash", async () => {
    vi.stubGlobal("fetch", vi.fn().mockRejectedValue(new TypeError("Failed to fetch")));
    const error = await createApi({ token: "t" }).listDrafts().catch((e) => e);
    expect(error).toBeInstanceOf(ApiError);
    expect(error.status).toBe(0);
    expect(error.message).toBe("could not reach the API");
  });
});

describe("createApi with a visitor's key or demo code", () => {
  const KEY = "sk-or-MARKER-31337";

  it("sends the key only as X-Model-Preset and X-Model-Key, never in the URL or body", async () => {
    const draft = makeDraft();
    const fetch = vi.fn().mockResolvedValue(reply(200, draft));
    vi.stubGlobal("fetch", fetch);
    const api = createApi({ model: { preset: "openrouter", secret: KEY } });
    await api.listDrafts();
    await api.saveText(draft.id, "new text");
    for (const [url, init] of fetch.mock.calls) {
      const headers = new Headers(init.headers);
      expect(headers.get("X-Model-Preset")).toBe("openrouter");
      expect(headers.get("X-Model-Key")).toBe(KEY);
      expect(headers.get("X-Demo-Code")).toBeNull();
      expect(headers.get("Authorization")).toBeNull();
      expect(String(url)).not.toContain(KEY);
      expect(String(init.body ?? "")).not.toContain(KEY);
    }
  });

  it("sends a demo code as X-Demo-Code", async () => {
    const fetch = vi.fn().mockResolvedValue(reply(200, []));
    vi.stubGlobal("fetch", fetch);
    await createApi({ model: { preset: "demo", secret: "code-9" } }).listDrafts();
    const headers = new Headers(fetch.mock.calls[0][1].headers);
    expect(headers.get("X-Model-Preset")).toBe("demo");
    expect(headers.get("X-Demo-Code")).toBe("code-9");
    expect(headers.get("X-Model-Key")).toBeNull();
  });

  it("tells the page about a 401 with the API's header-only message", async () => {
    vi.stubGlobal("fetch", vi.fn().mockResolvedValue(reply(401, { detail: "X-Model-Key was refused" })));
    const onUnauthorized = vi.fn();
    const error = await createApi({ model: { preset: "openai", secret: KEY } }, "", onUnauthorized)
      .listDrafts()
      .catch((e) => e);
    expect(error.status).toBe(401);
    expect(error.message).toBe("X-Model-Key was refused");
    expect(onUnauthorized).toHaveBeenCalledWith(error);
  });

  it("does not tell the page about other failures", async () => {
    vi.stubGlobal("fetch", vi.fn().mockResolvedValue(reply(409, { detail: "busy" })));
    const onUnauthorized = vi.fn();
    await createApi({ token: "t" }, "", onUnauthorized).listDrafts().catch(() => {});
    expect(onUnauthorized).not.toHaveBeenCalled();
  });

  it("never shows a message that echoes the secret", async () => {
    vi.stubGlobal("fetch", vi.fn().mockResolvedValue(reply(401, { detail: `bad key ${KEY}` })));
    const error = await createApi({ model: { preset: "openai", secret: KEY } })
      .listDrafts()
      .catch((e) => e);
    expect(error.message).toBe("the API answered 401");
  });
});

describe("listProviders", () => {
  it("returns the providers, asked with no credentials", async () => {
    const fetch = vi.fn().mockResolvedValue(reply(200, PROVIDERS));
    vi.stubGlobal("fetch", fetch);
    expect(await listProviders("http://api")).toEqual(PROVIDERS);
    const [url, init] = fetch.mock.calls[0];
    expect(url).toBe("http://api/api/providers");
    const headers = new Headers(init?.headers);
    expect(headers.get("Authorization")).toBeNull();
    expect(headers.get("X-Model-Preset")).toBeNull();
  });

  it("is null (server-key mode) for a 404, a 401, an empty list, a non-list or no answer", async () => {
    for (const answer of [reply(404, { detail: "not found" }), reply(401, {}), reply(200, []), reply(200, {})]) {
      vi.stubGlobal("fetch", vi.fn().mockResolvedValue(answer));
      expect(await listProviders()).toBeNull();
    }
    vi.stubGlobal("fetch", vi.fn().mockRejectedValue(new TypeError("Failed to fetch")));
    expect(await listProviders()).toBeNull();
  });
});
