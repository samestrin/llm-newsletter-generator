// The visitor's model credentials in the browser: a provider id and their key or demo
// code. Session storage by default; local storage only when they tick "Remember on this
// device". The secret goes to the API only as a request header (modelHeaders) and is
// never put in a URL, a body, a log line or the page.

export interface ModelCredentials {
  /** A provider id from `GET /api/providers`; `demo` means the secret is a demo code. */
  preset: string;
  secret: string;
}

export interface StoredCredentials extends ModelCredentials {
  remember: boolean;
}

export const STORAGE_KEY = "newsletter.model";

function parse(raw: string | null): StoredCredentials | null {
  if (raw === null) return null;
  try {
    const value = JSON.parse(raw);
    if (value && typeof value.preset === "string" && value.preset && typeof value.secret === "string" && value.secret) {
      return { preset: value.preset, secret: value.secret, remember: value.remember === true };
    }
  } catch {
    // A malformed stored value is no credentials.
  }
  return null;
}

/** The saved credentials, session storage first; `null` when none are usable. */
export function loadCredentials(): StoredCredentials | null {
  try {
    return parse(sessionStorage.getItem(STORAGE_KEY)) ?? parse(localStorage.getItem(STORAGE_KEY));
  } catch {
    // Storage can be blocked (private browsing); the page then holds none.
    return null;
  }
}

/** Save into one store only: local with `remember`, session without. */
export function saveCredentials(credentials: ModelCredentials, remember: boolean): void {
  const value: StoredCredentials = { ...credentials, remember };
  try {
    const keep = remember ? localStorage : sessionStorage;
    const drop = remember ? sessionStorage : localStorage;
    keep.setItem(STORAGE_KEY, JSON.stringify(value));
    drop.removeItem(STORAGE_KEY);
  } catch {
    // Blocked storage: the credentials still live in the page's state for this visit.
  }
}

/** Wipe the credentials from both stores. */
export function clearCredentials(): void {
  try {
    sessionStorage.removeItem(STORAGE_KEY);
    localStorage.removeItem(STORAGE_KEY);
  } catch {
    // Nothing was stored if storage is blocked.
  }
}

/** The request headers for a model call: the secret is a header value, nothing else. */
export function modelHeaders(credentials: ModelCredentials): Record<string, string> {
  return credentials.preset === "demo"
    ? { "X-Model-Preset": "demo", "X-Demo-Code": credentials.secret }
    : { "X-Model-Preset": credentials.preset, "X-Model-Key": credentials.secret };
}
