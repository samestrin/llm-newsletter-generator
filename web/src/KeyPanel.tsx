import { useEffect, useId, useRef, useState, type FormEvent } from "react";
import type { Provider } from "./api";
import type { ModelCredentials, StoredCredentials } from "./modelAccess";

/**
 * The key panel: a provider list fed by `GET /api/providers`, the visitor's own key or a
 * demo code, a remember box, Clear, Save and Close. The page stores the secret and sends
 * it only as a request header. The field is uncontrolled and never prefilled, so the
 * secret is never written into the page: React would copy a controlled value into the
 * input's `value` attribute.
 */
export function KeyPanel({
  providers,
  stored,
  error = null,
  onSave,
  onClear,
  onClose,
}: {
  providers: Provider[];
  stored: StoredCredentials | null;
  /** The API's message for a refused key or code; it names the header, never the value. */
  error?: string | null;
  onSave: (credentials: ModelCredentials, remember: boolean) => void;
  onClear: () => void;
  onClose: () => void;
}) {
  const id = useId();
  const field = useRef<HTMLInputElement>(null);
  const heading = useRef<HTMLHeadingElement>(null);
  const [providerId, setProviderId] = useState(
    providers.some((p) => p.id === stored?.preset) ? stored!.preset : (providers[0]?.id ?? ""),
  );
  const [hasSecret, setHasSecret] = useState(false);
  const [remember, setRemember] = useState(stored?.remember ?? false);
  const provider = providers.find((p) => p.id === providerId);
  const savedFor = stored && providers.find((p) => p.id === stored.preset);

  // The panel opens on purpose (first visit, the Key button, a refused key): go to it.
  useEffect(() => heading.current?.focus(), []);

  function onSubmit(e: FormEvent) {
    e.preventDefault();
    const secret = field.current?.value.trim() ?? "";
    if (!secret || !provider) return;
    onSave({ preset: provider.id, secret }, remember);
  }

  function onClearClick() {
    if (field.current) field.current.value = "";
    setHasSecret(false);
    setRemember(false);
    onClear();
  }

  return (
    <section aria-labelledby={`${id}-heading`} className="key-panel">
      <h2 id={`${id}-heading`} ref={heading} tabIndex={-1}>
        Model key
      </h2>
      <p>
        Every newsletter is written by a model. Bring your own key for a provider, or use a demo code. The key stays
        in this browser and is sent only with your requests.
      </p>
      {error && (
        <p role="alert" className="state state-error">
          {error}
        </p>
      )}
      {savedFor && <p className="muted">A key for {savedFor.label} is saved in this browser.</p>}
      <form onSubmit={onSubmit}>
        <label htmlFor={`${id}-provider`}>Provider</label>
        <select id={`${id}-provider`} value={providerId} onChange={(e) => setProviderId(e.target.value)}>
          {providers.map((p) => (
            <option key={p.id} value={p.id}>
              {p.label}
            </option>
          ))}
        </select>
        <label htmlFor={`${id}-secret`}>{provider?.kind === "demo" ? "Demo code" : "API key"}</label>
        <input
          ref={field}
          id={`${id}-secret`}
          type="password"
          autoComplete="off"
          spellCheck={false}
          onChange={(e) => setHasSecret(e.target.value.trim() !== "")}
        />
        <label className="remember-field">
          <input type="checkbox" checked={remember} onChange={(e) => setRemember(e.target.checked)} /> Remember on
          this device
        </label>
        {provider && !provider.embedding_model && (
          <p className="key-note">{provider.label} has no embedding model: stories will not be grouped by meaning.</p>
        )}
        <div className="actions">
          <button type="submit" className="primary" disabled={!hasSecret || !provider}>
            Save
          </button>
          <button type="button" onClick={onClearClick}>
            Clear
          </button>
          <button type="button" onClick={onClose}>
            Close
          </button>
        </div>
      </form>
    </section>
  );
}
