import { render, screen, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it, vi } from "vitest";
import { KeyPanel } from "./KeyPanel";
import { PROVIDERS } from "./fixtures";
import type { StoredCredentials } from "./modelAccess";

function setup(stored: StoredCredentials | null = null, error: string | null = null) {
  const onSave = vi.fn();
  const onClear = vi.fn();
  const onClose = vi.fn();
  const user = userEvent.setup();
  const view = render(
    <KeyPanel
      providers={PROVIDERS}
      stored={stored}
      error={error}
      onSave={onSave}
      onClear={onClear}
      onClose={onClose}
    />,
  );
  return { onSave, onClear, onClose, user, view };
}

describe("KeyPanel", () => {
  it("lists the providers GET /api/providers gave it, by label", () => {
    setup();
    const options = within(screen.getByLabelText("Provider")).getAllByRole("option");
    expect(options.map((o) => o.textContent)).toEqual([
      "Demo code",
      "OpenAI",
      "OpenRouter",
      "Alibaba",
      "Synthetic",
    ]);
  });

  it("shows the notice that the key stays in this browser", () => {
    setup();
    expect(screen.getByRole("region", { name: "Model key" })).toHaveTextContent(
      "The key stays in this browser and is sent only with your requests.",
    );
  });

  it("asks for a demo code for the demo entry and an API key otherwise", async () => {
    const { user } = setup();
    expect(screen.getByLabelText("Demo code", { selector: "input" })).toBeInTheDocument();
    expect(screen.queryByLabelText("API key")).toBeNull();
    await user.selectOptions(screen.getByLabelText("Provider"), "openai");
    expect(screen.getByLabelText("API key")).toBeInTheDocument();
    expect(screen.queryByLabelText("Demo code", { selector: "input" })).toBeNull();
  });

  it("keeps Save disabled until there is a secret", async () => {
    const { user, onSave } = setup();
    const save = screen.getByRole("button", { name: "Save" });
    expect(save).toBeDisabled();
    await user.type(screen.getByLabelText("Demo code", { selector: "input" }), "   ");
    expect(save).toBeDisabled();
    await user.type(screen.getByLabelText("Demo code", { selector: "input" }), "x");
    expect(save).toBeEnabled();
    expect(onSave).not.toHaveBeenCalled();
  });

  it("saves the trimmed secret with the provider and the remember box", async () => {
    const { user, onSave } = setup();
    await user.selectOptions(screen.getByLabelText("Provider"), "openrouter");
    await user.type(screen.getByLabelText("API key"), "  sk-or-1  ");
    await user.click(screen.getByLabelText("Remember on this device"));
    await user.click(screen.getByRole("button", { name: "Save" }));
    expect(onSave).toHaveBeenCalledWith({ preset: "openrouter", secret: "sk-or-1" }, true);
  });

  it("saves without remember by default", async () => {
    const { user, onSave } = setup();
    await user.type(screen.getByLabelText("Demo code", { selector: "input" }), "code-1");
    await user.click(screen.getByRole("button", { name: "Save" }));
    expect(onSave).toHaveBeenCalledWith({ preset: "demo", secret: "code-1" }, false);
  });

  it("says when the chosen provider has no embedding model", async () => {
    const { user } = setup();
    const note = "Synthetic has no embedding model: stories will not be grouped by meaning.";
    expect(screen.queryByText(note)).toBeNull();
    await user.selectOptions(screen.getByLabelText("Provider"), "synthetic");
    expect(screen.getByText(note)).toBeInTheDocument();
    await user.selectOptions(screen.getByLabelText("Provider"), "openai");
    expect(screen.queryByText(note)).toBeNull();
  });

  it("starts from the stored provider and remember box, never rendering the stored secret", () => {
    const secret = "sk-stored-SECRET-42";
    const { view } = setup({ preset: "synthetic", secret, remember: true });
    expect(screen.getByLabelText("Provider")).toHaveValue("synthetic");
    expect(screen.getByLabelText("Remember on this device")).toBeChecked();
    expect(screen.getByText("A key for Synthetic is saved in this browser.")).toBeInTheDocument();
    const field = screen.getByLabelText("API key");
    expect(field).toHaveAttribute("type", "password");
    expect(field).toHaveValue("");
    expect(view.container.innerHTML).not.toContain(secret);
  });

  it("never writes a typed secret into the DOM", async () => {
    const secret = "sk-typed-SECRET-77";
    const { user, view } = setup();
    await user.selectOptions(screen.getByLabelText("Provider"), "openai");
    await user.type(screen.getByLabelText("API key"), secret);
    expect(view.container.innerHTML).not.toContain(secret);
    expect(view.container.textContent).not.toContain(secret);
  });

  it("Clear empties the field, unticks remember and reports the clear", async () => {
    const { user, onClear } = setup({ preset: "openai", secret: "sk-1", remember: true });
    await user.type(screen.getByLabelText("API key"), "sk-2");
    await user.click(screen.getByRole("button", { name: "Clear" }));
    expect(onClear).toHaveBeenCalledTimes(1);
    expect(screen.getByLabelText("API key")).toHaveValue("");
    expect(screen.getByLabelText("Remember on this device")).not.toBeChecked();
    expect(screen.getByRole("button", { name: "Save" })).toBeDisabled();
  });

  it("closes without saving", async () => {
    const { user, onClose, onSave } = setup();
    await user.click(screen.getByRole("button", { name: "Close" }));
    expect(onClose).toHaveBeenCalledTimes(1);
    expect(onSave).not.toHaveBeenCalled();
  });

  it("shows the API's message when the key was refused", () => {
    setup(null, "X-Model-Key was refused by the provider");
    expect(screen.getByRole("alert")).toHaveTextContent("X-Model-Key was refused by the provider");
  });
});
