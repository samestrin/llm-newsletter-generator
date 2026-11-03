import { afterEach, describe, expect, it, vi } from "vitest";
import { ApiError, createApi, sha256Hex } from "./api";
import { makeDraft } from "./fixtures";

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
});

describe("createApi", () => {
  it("sends the token on every read", async () => {
    const fetch = vi.fn().mockResolvedValue(reply(200, []));
    vi.stubGlobal("fetch", fetch);
    await createApi("s3cret").listDrafts();
    const [url, init] = fetch.mock.calls[0];
    expect(url).toBe("/api/drafts");
    expect(new Headers(init.headers).get("Authorization")).toBe("Bearer s3cret");
  });

  it("saves the text as a JSON object", async () => {
    const draft = makeDraft();
    const fetch = vi.fn().mockResolvedValue(reply(200, draft));
    vi.stubGlobal("fetch", fetch);
    await createApi("t").saveText(draft.id, "new text");
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
    await createApi("t").approve(draft);
    const [url, init] = fetch.mock.calls[0];
    expect(url).toBe(`/api/drafts/${draft.id}/approve`);
    expect(init.method).toBe("POST");
    expect(JSON.parse(init.body)).toEqual({
      sha256: "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad",
    });
  });

  it("raises the server's reason with the status", async () => {
    vi.stubGlobal("fetch", vi.fn().mockResolvedValue(reply(409, { detail: "the draft changed" })));
    const error = await createApi("t").approve(makeDraft()).catch((e) => e);
    expect(error).toBeInstanceOf(ApiError);
    expect(error.status).toBe(409);
    expect(error.message).toBe("the draft changed");
  });

  it("names the status when the body has no reason", async () => {
    vi.stubGlobal("fetch", vi.fn().mockResolvedValue(new Response("<html>", { status: 502 })));
    const error = await createApi("t").listDrafts().catch((e) => e);
    expect(error.status).toBe(502);
    expect(error.message).toBe("the API answered 502");
  });

  it("reports a network failure as an error, not a crash", async () => {
    vi.stubGlobal("fetch", vi.fn().mockRejectedValue(new TypeError("Failed to fetch")));
    const error = await createApi("t").listDrafts().catch((e) => e);
    expect(error).toBeInstanceOf(ApiError);
    expect(error.status).toBe(0);
    expect(error.message).toBe("could not reach the API");
  });
});
