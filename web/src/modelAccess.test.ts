import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import {
  STORAGE_KEY,
  clearCredentials,
  loadCredentials,
  modelHeaders,
  saveCredentials,
} from "./modelAccess";

beforeEach(() => {
  sessionStorage.clear();
  localStorage.clear();
});
afterEach(() => vi.restoreAllMocks());

describe("saveCredentials", () => {
  it("keeps the secret in session storage only, without remember", () => {
    saveCredentials({ preset: "openai", secret: "sk-1" }, false);
    expect(JSON.parse(sessionStorage.getItem(STORAGE_KEY)!)).toEqual({
      preset: "openai",
      secret: "sk-1",
      remember: false,
    });
    expect(localStorage.getItem(STORAGE_KEY)).toBeNull();
  });

  it("keeps it in local storage only, with remember", () => {
    saveCredentials({ preset: "openai", secret: "sk-1" }, false);
    saveCredentials({ preset: "demo", secret: "code-1" }, true);
    expect(JSON.parse(localStorage.getItem(STORAGE_KEY)!)).toEqual({
      preset: "demo",
      secret: "code-1",
      remember: true,
    });
    expect(sessionStorage.getItem(STORAGE_KEY)).toBeNull();
  });

  it("moves back to session storage when remember is unticked", () => {
    saveCredentials({ preset: "demo", secret: "code-1" }, true);
    saveCredentials({ preset: "demo", secret: "code-1" }, false);
    expect(localStorage.getItem(STORAGE_KEY)).toBeNull();
    expect(sessionStorage.getItem(STORAGE_KEY)).not.toBeNull();
  });

  it("does not throw when storage is blocked", () => {
    vi.spyOn(Storage.prototype, "setItem").mockImplementation(() => {
      throw new DOMException("blocked", "SecurityError");
    });
    expect(() => saveCredentials({ preset: "openai", secret: "sk-1" }, true)).not.toThrow();
  });
});

describe("loadCredentials", () => {
  it("reads either store, session first", () => {
    localStorage.setItem(STORAGE_KEY, JSON.stringify({ preset: "openai", secret: "local", remember: true }));
    expect(loadCredentials()).toEqual({ preset: "openai", secret: "local", remember: true });
    sessionStorage.setItem(STORAGE_KEY, JSON.stringify({ preset: "demo", secret: "session" }));
    expect(loadCredentials()).toEqual({ preset: "demo", secret: "session", remember: false });
  });

  it("is null for nothing, malformed JSON or an empty secret", () => {
    expect(loadCredentials()).toBeNull();
    sessionStorage.setItem(STORAGE_KEY, "{not json");
    expect(loadCredentials()).toBeNull();
    sessionStorage.setItem(STORAGE_KEY, JSON.stringify({ preset: "openai", secret: "" }));
    expect(loadCredentials()).toBeNull();
    sessionStorage.setItem(STORAGE_KEY, JSON.stringify({ preset: 3, secret: "x" }));
    expect(loadCredentials()).toBeNull();
  });

  it("is null when storage is blocked", () => {
    vi.spyOn(Storage.prototype, "getItem").mockImplementation(() => {
      throw new DOMException("blocked", "SecurityError");
    });
    expect(loadCredentials()).toBeNull();
  });
});

describe("clearCredentials", () => {
  it("wipes both stores", () => {
    sessionStorage.setItem(STORAGE_KEY, "a");
    localStorage.setItem(STORAGE_KEY, "b");
    clearCredentials();
    expect(sessionStorage.getItem(STORAGE_KEY)).toBeNull();
    expect(localStorage.getItem(STORAGE_KEY)).toBeNull();
  });

  it("does not throw when storage is blocked", () => {
    vi.spyOn(Storage.prototype, "removeItem").mockImplementation(() => {
      throw new DOMException("blocked", "SecurityError");
    });
    expect(() => clearCredentials()).not.toThrow();
  });
});

describe("modelHeaders", () => {
  it("sends a visitor key as X-Model-Key with its preset", () => {
    expect(modelHeaders({ preset: "openrouter", secret: "sk-or" })).toEqual({
      "X-Model-Preset": "openrouter",
      "X-Model-Key": "sk-or",
    });
  });

  it("sends a demo code as X-Demo-Code, never as a key", () => {
    expect(modelHeaders({ preset: "demo", secret: "code-1" })).toEqual({
      "X-Model-Preset": "demo",
      "X-Demo-Code": "code-1",
    });
  });
});
