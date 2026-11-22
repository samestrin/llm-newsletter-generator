import { render, screen, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it } from "vitest";
import type { Report } from "./api";
import { makeDraft } from "./fixtures";
import { RunReport } from "./Report";

const fixture = makeDraft().report;

function show(report: Report, focus?: string) {
  const user = userEvent.setup();
  render(<RunReport report={report} focus={focus} />);
  return { user, summary: screen.getByRole("list", { name: "Run summary" }) };
}

describe("the run summary", () => {
  it("shows the focus, each model once, the number of steps and the total cost", () => {
    const report: Report = {
      ...fixture,
      steps: [
        ...fixture.steps!,
        { step: "edit", tier: "reasoning", model: "big-model", prompt_tokens: 10, completion_tokens: 5, cost: 0.01 },
      ],
      total_cost: 0.0142,
      cost_complete: true,
    };
    const { summary } = show(report, "chips");
    expect(screen.getByRole("region", { name: "Run report" })).toHaveTextContent("Focus: chips");
    expect(summary).toHaveTextContent("Models used: fast-model, big-model");
    expect(summary).toHaveTextContent("Steps: 3");
    expect(summary).toHaveTextContent("Total cost: $0.0142");
    expect(summary).not.toHaveTextContent("unknown");
  });

  it("says some costs are unknown when cost_complete is false", () => {
    const { summary } = show(fixture);
    expect(summary).toHaveTextContent("Total cost: $0.0042, some costs unknown");
  });

  it("says some costs are unknown on an older report with a null step cost and no cost_complete", () => {
    const { cost_complete: _complete, ...older } = fixture;
    const { summary } = show(older);
    expect(summary).toHaveTextContent("some costs unknown");
  });

  it("says the total is unknown when the report has none", () => {
    const { summary } = show({ steps: [{ step: "edit", model: "m", cost: null }], total_cost: null as unknown as number });
    expect(summary).toHaveTextContent("Total cost: unknown");
  });

  it("says so when no model was called", () => {
    const { summary } = show({ steps: [], total_cost: 0, cost_complete: true });
    expect(summary).toHaveTextContent("Models used: none");
    expect(summary).toHaveTextContent("Steps: 0");
  });

  it("shows the editor's plain message in place of the table and the raw error", () => {
    const message = "The editor ran out of time, so this issue has no intro, themes or lead.";
    show({ ...fixture, editor_error: "ReadTimeout('timed out')", editor_message: message });
    const report = screen.getByRole("region", { name: "Run report" });
    expect(report).toHaveTextContent(message);
    expect(report).not.toHaveTextContent("ReadTimeout");
    expect(screen.queryByRole("table")).toBeNull();
  });

  it("says the editor failed on an older report with only the raw error", () => {
    show({ ...fixture, editor_error: "ReadTimeout('timed out')" });
    const report = screen.getByRole("region", { name: "Run report" });
    expect(report).toHaveTextContent("The editor failed");
    expect(report).not.toHaveTextContent("ReadTimeout");
  });
});

describe("the run report's details", () => {
  it("keeps the table hidden until Show details is pressed, and hides it again", async () => {
    const { user } = show(fixture);
    const toggle = screen.getByRole("button", { name: "Show details" });
    expect(toggle).toHaveAttribute("aria-expanded", "false");
    expect(screen.queryByRole("table")).toBeNull();

    await user.click(toggle);
    expect(toggle).toHaveAttribute("aria-expanded", "true");
    expect(toggle).toHaveAccessibleName("Hide details");
    const table = screen.getByRole("region", { name: "Model calls table" });
    const rows = within(table).getAllByRole("row").slice(1);
    expect(rows).toHaveLength(2);
    expect(rows[1]).toHaveTextContent("unknown");

    await user.click(toggle);
    expect(screen.queryByRole("table")).toBeNull();
  });

  it("shows the editor's raw error among the details", async () => {
    const { user } = show({ ...fixture, editor_error: "ReadTimeout('timed out')", editor_message: "Plain." });
    await user.click(screen.getByRole("button", { name: "Show details" }));
    expect(screen.getByRole("region", { name: "Run report" })).toHaveTextContent("Editor failed: ReadTimeout");
  });
});
