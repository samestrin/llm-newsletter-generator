import { render, screen, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { useState } from "react";
import { describe, expect, it, vi } from "vitest";
import type { Api, Draft, Flag } from "./api";
import { DraftView } from "./DraftView";
import { makeDraft } from "./fixtures";

/** The Markdown `newsletter run` writes: a lead story, themes and flagged claims with source links. */
const ISSUE = [
  "# AI Weekly",
  "",
  "*2026-11-03*",
  "",
  "Two models and a rule this week. [1](<https://example.com/story-1>)",
  "",
  "**Themes**",
  "",
  "- Smaller models [1](<https://example.com/story-1>)",
  "",
  "## Lead story: [Tiny model wins](<https://example.com/story-1>)",
  "",
  "*Example Feed · 2026-11-03*",
  "",
  "**Why it matters:** Laptops can run it.",
  "",
  "- \\[1\\] **Flagged: unsupported, the source does not support this claim.** The model scored 99% on every benchmark. ([source](<https://example.com/story-1>))",
  "- \\[2\\] It runs on a laptop. ([source](<https://example.com/story-1>))",
  "",
].join("\n");

const FLAG: Flag = {
  id: "s1-c1",
  story_id: "s1",
  text: "The model scored 99% on every benchmark.",
  status: "unsupported",
  reason: "The source gives 91%.",
  source_url: "https://example.com/story-1",
  passage: "It scored 91% on the benchmark and runs on a laptop.",
  explanation: "The article says 91%; the draft said 99%.",
};

function fakeApi(start: Draft) {
  let draft = start;
  return {
    saveText: vi.fn(async (_id: string, text: string) => {
      draft = { ...draft, text, approval: null, approved: false };
      return draft;
    }),
    getDraft: vi.fn(async () => draft),
    approve: vi.fn(async () => {
      draft = {
        ...draft,
        approved: true,
        approval: { sha256: "x", approved_at: "now" },
      };
      return draft;
    }),
  } as unknown as Api & { saveText: ReturnType<typeof vi.fn> };
}

function Harness({ api, start }: { api: Api; start: Draft }) {
  const [draft, setDraft] = useState(start);
  return <DraftView api={api} draft={draft} onChange={setDraft} />;
}

function show(overrides: Partial<Draft> = {}) {
  const draft = makeDraft({ text: ISSUE, flags: [FLAG], ...overrides });
  const api = fakeApi(draft);
  const user = userEvent.setup();
  const view = render(<Harness api={api} start={draft} />);
  return {
    api,
    user,
    view,
    newsletter: () => screen.getByTestId("draft-text"),
  };
}

describe("the draft as a newsletter", () => {
  it("renders headings, lead, themes, stories and claim links from the Markdown", () => {
    const { newsletter } = show();
    const body = newsletter();
    expect(
      within(body).getByRole("heading", { name: "AI Weekly" }),
    ).toBeInTheDocument();
    const lead = within(body).getByRole("heading", {
      name: "Lead story: Tiny model wins",
    });
    expect(
      within(lead).getByRole("link", { name: "Tiny model wins" }),
    ).toHaveAttribute("href", "https://example.com/story-1");
    expect(body).toHaveTextContent("Themes");
    expect(body).toHaveTextContent("Why it matters: Laptops can run it.");
    expect(
      within(body).getAllByRole("link", { name: "source" })[0],
    ).toHaveAttribute("href", "https://example.com/story-1");
  });

  it("keeps the draft's headings below the page's own", () => {
    const { newsletter } = show();
    expect(
      within(newsletter()).queryByRole("heading", { level: 1 }),
    ).toBeNull();
    expect(
      within(newsletter()).queryByRole("heading", { level: 2 }),
    ).toBeNull();
    expect(
      within(newsletter()).getByRole("heading", {
        level: 3,
        name: "AI Weekly",
      }),
    ).toBeInTheDocument();
  });

  it("opens links in a new tab with noopener noreferrer", () => {
    const { newsletter } = show();
    for (const link of within(newsletter()).getAllByRole("link")) {
      expect(link).toHaveAttribute("target", "_blank");
      expect(link).toHaveAttribute("rel", "noopener noreferrer");
    }
  });

  it("shows a <script> and a javascript: link as inert text", () => {
    const text = [
      "# Hostile",
      "",
      "<script>alert(1)</script>",
      "",
      '<img src=x onerror="alert(2)"><b>bold</b>',
      "",
      "[click me](javascript:alert(3)) and [later](JavaScript:alert(4))",
    ].join("\n");
    const { view, newsletter } = show({ text });
    expect(view.container.querySelector("script, img, b, iframe")).toBeNull();
    expect(newsletter()).toHaveTextContent("<script>alert(1)</script>");
    expect(newsletter()).toHaveTextContent("click me");
    expect(within(newsletter()).queryByRole("link")).toBeNull();
    for (const a of view.container.querySelectorAll("a")) {
      expect(a.getAttribute("href") ?? "").toMatch(/^https?:/);
    }
  });

  it("links only http and https: not relative, mailto, data or protocol-relative URLs", () => {
    const text = [
      "[rel](/api/drafts) [mail](mailto:a@example.com) [data](data:text/html,hi)",
      "[proto](//evil.example.com) [ok](http://example.com/a) [ok2](https://example.com/b)",
      "",
      "<https://example.com/auto>",
    ].join("\n");
    const { newsletter } = show({ text });
    const links = within(newsletter()).getAllByRole("link");
    expect(links.map((a) => a.getAttribute("href"))).toEqual([
      "http://example.com/a",
      "https://example.com/b",
      "https://example.com/auto",
    ]);
    expect(newsletter()).toHaveTextContent("rel mail data proto");
  });

  it("says when the draft has no text", () => {
    show({ text: "" });
    expect(
      screen.getByText("This draft has no text. Edit it to write some."),
    ).toBeInTheDocument();
  });
});

describe("a flagged claim in plain words", () => {
  it("shows the explanation, passage and source link next to the claim", () => {
    const { newsletter } = show();
    const claim = within(newsletter())
      .getAllByRole("listitem")
      .find((li) => li.textContent?.includes(FLAG.text))!;
    const note = within(claim).getByRole("note");
    expect(note).toHaveTextContent("Flagged: unsupported");
    expect(within(note).getByTestId("flag-icon")).toHaveAttribute(
      "aria-hidden",
      "true",
    );
    expect(note).toHaveTextContent(FLAG.explanation!);
    expect(within(note).getByText(FLAG.passage!)).toBeInTheDocument();
    expect(
      within(note).getByRole("link", { name: FLAG.source_url }),
    ).toHaveAttribute("rel", "noopener noreferrer");
  });

  it("explains a flag once, at the claim, not again in the draft's own run report", () => {
    const text = `${ISSUE}\n## Run report\n\n**Claims flagged**\n\n- \`s1-c1\` unsupported: ${FLAG.text} (${FLAG.reason})\n`;
    const { newsletter } = show({ text });
    const notes = within(newsletter()).getAllByRole("note");
    expect(notes).toHaveLength(1);
    expect(notes[0].closest("li")).toHaveTextContent(
      "Flagged: unsupported, the source does not support this claim.",
    );
  });

  it("does not mark a claim that is not flagged", () => {
    const { newsletter } = show();
    const claim = within(newsletter())
      .getAllByRole("listitem")
      .find((li) => li.textContent?.includes("It runs on a laptop."))!;
    expect(within(claim).queryByRole("note")).toBeNull();
  });

  it("shows the model's reason for an old draft with no explanation", () => {
    const { passage: _p, explanation: _e, ...old } = FLAG;
    const { newsletter } = show({ flags: [old] });
    const note = within(newsletter()).getByRole("note");
    expect(note).toHaveTextContent(FLAG.reason);
    expect(note).not.toHaveTextContent("Source text");
  });

  it("lists every flag in the Flags section too, with its explanation and passage", () => {
    show();
    const item = within(
      screen.getByRole("region", { name: "Flags (1)" }),
    ).getByRole("listitem");
    expect(item).toHaveTextContent(FLAG.explanation!);
    expect(item).toHaveTextContent(FLAG.passage!);
  });
});

describe("edit, save, done editing", () => {
  it("shows the edited text once editing is done", async () => {
    const { api, user, newsletter } = show();
    await user.click(screen.getByRole("button", { name: "Edit" }));
    const box = screen.getByLabelText("Draft text");
    await user.clear(box);
    await user.type(box, "# Edited{Enter}{Enter}Fixed by the reviewer.");
    await user.click(screen.getByRole("button", { name: "Save" }));
    expect(api.saveText).toHaveBeenCalledWith(
      makeDraft().id,
      "# Edited\n\nFixed by the reviewer.",
    );
    expect(screen.getByLabelText("Draft text")).toBeInTheDocument();
    await user.click(screen.getByRole("button", { name: "Done editing" }));
    expect(
      within(newsletter()).getByRole("heading", { name: "Edited" }),
    ).toBeInTheDocument();
    expect(newsletter()).toHaveTextContent("Fixed by the reviewer.");
  });

  it("will not leave the editor with unsaved changes until they are saved or discarded", async () => {
    const { user, newsletter } = show();
    await user.click(screen.getByRole("button", { name: "Edit" }));
    await user.type(screen.getByLabelText("Draft text"), " more");
    expect(screen.getByRole("button", { name: "Done editing" })).toBeDisabled();
    expect(screen.getByRole("button", { name: "Approve" })).toBeDisabled();
    await user.click(screen.getByRole("button", { name: "Discard changes" }));
    expect(screen.getByLabelText("Draft text")).toHaveValue(ISSUE);
    await user.click(screen.getByRole("button", { name: "Done editing" }));
    expect(newsletter()).not.toHaveTextContent("more");
  });

  it("warns that saving an approved draft clears its approval", async () => {
    const { user } = show({
      approved: true,
      approval: { sha256: "x", approved_at: "then" },
    });
    await user.click(screen.getByRole("button", { name: "Edit" }));
    expect(
      screen.getByText(/Saving a change clears the approval/),
    ).toBeInTheDocument();
    await user.type(screen.getByLabelText("Draft text"), "!");
    await user.click(screen.getByRole("button", { name: "Save" }));
    expect(await screen.findByTestId("approval")).toHaveTextContent(
      "Not approved",
    );
  });
});

describe("the run's focus", () => {
  it("is shown at the top of the draft and in the report", () => {
    show({ focus: "practical AI tools for small businesses" });
    expect(screen.getByTestId("focus")).toHaveTextContent(
      "Focus: practical AI tools for small businesses",
    );
    expect(
      screen.getByRole("region", { name: "Run report" }),
    ).toHaveTextContent("Focus: practical AI tools for small businesses");
  });

  it("is read from the report when the draft has none", () => {
    show({ report: { ...makeDraft().report, focus: "chips" } });
    expect(screen.getByTestId("focus")).toHaveTextContent("Focus: chips");
  });

  it("is not shown when the run had none", () => {
    show();
    expect(screen.queryByTestId("focus")).toBeNull();
    expect(
      screen.getByRole("region", { name: "Run report" }),
    ).not.toHaveTextContent("Focus:");
  });
});

describe("what the run left out", () => {
  it("still lists failed sources and dropped items on the draft", () => {
    show();
    const report = screen.getByRole("region", { name: "Run report" });
    expect(report).toHaveTextContent(
      "https://down.example.com/feed.xml: HTTP 503",
    );
    expect(report).toHaveTextContent("Empty story");
    expect(report).toHaveTextContent("Dropped claim text");
  });
});
