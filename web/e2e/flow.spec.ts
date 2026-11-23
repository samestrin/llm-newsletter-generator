import AxeBuilder from "@axe-core/playwright";
import { expect, test, type Page, type Request } from "@playwright/test";

// The visitor's whole path through the page, against the stateful fake API: key, feeds,
// focus, the run's live progress, the draft with its flagged claim, edit, approve and the
// email preview. Each run step shows for one poll (2 s), so the first test takes about 20 s.
const API = "http://127.0.0.1:8787";
const KEY = "sk-or-e2e-key";
const FOCUS = "practical AI tools";
const GOOD = "https://good.example.com/feed.xml";
const ALSO = "https://also.example.com/rss";
const DOWN = "https://down.example.com/feed.xml";
const FLAKY = "https://flaky.example.com/feed.xml";

/** Every step, in order, and the counts it shows while it runs (`null`: none). */
const STEPS: [string, string | null][] = [
  ["Fetching feeds", "3 of 3"],
  ["Reading articles", "2 of 4"],
  ["Grouping and ranking stories", null],
  ["Writing claims", "1 of 3"],
  ["Checking claims against sources", "2 of 3"],
  ["Editing the issue", null],
  ["Done", null],
];

test.setTimeout(90_000);

test.beforeEach(async ({ request, page }) => {
  await request.post(`${API}/__reset`);
  await request.post(`${API}/__keys`);
  page.on("dialog", (dialog) => {
    throw new Error(`the page ran script: ${dialog.message()}`);
  });
});

/** Every serious or critical axe finding on the page as it is now, one line each. */
async function axeIssues(page: Page) {
  // Legacy mode runs axe in the page itself. The default opens a second page to put the
  // frames' results together, and that never opens once the sandboxed email preview is shown.
  const { violations } = await new AxeBuilder({ page }).setLegacyMode().analyze();
  return violations
    .filter((v) => v.impact === "serious" || v.impact === "critical")
    .map((v) => `${v.id} (${v.impact}): ${v.nodes.map((n) => n.target.join(" ")).join(", ")}`);
}

/** The page passes axe and nothing scrolls sideways, at whatever width the project has. */
async function checkPage(page: Page) {
  expect(await axeIssues(page)).toEqual([]);
  expect(
    await page.evaluate(() => document.documentElement.scrollWidth <= document.documentElement.clientWidth),
  ).toBe(true);
}

const row = (page: Page, n: number) => page.getByRole("group", { name: `Feed ${n}` });

async function addFeed(page: Page, n: number, url: string) {
  if (n > 1) await page.getByRole("button", { name: "Add a feed" }).click();
  const address = row(page, n).getByLabel(`Feed ${n} address`);
  await address.fill(url);
  await address.blur();
}

/** Watch each step run in turn, with its counts and elapsed time, until the draft opens. */
async function watchEveryStep(page: Page) {
  await expect(page.getByRole("heading", { name: "Making your newsletter…" })).toBeVisible();
  const steps = page.locator('ol[aria-label="Steps"] > li');
  await expect(steps).toHaveCount(STEPS.length);
  for (const [n, [label, counts]] of STEPS.entries()) {
    const step = steps.nth(n);
    await expect(step.locator(".step-label")).toHaveText(label);
    await expect(step.locator(".step-state")).toHaveText("Running");
    await expect(step.locator(".step-time")).toHaveText(/^\d+ s$/);
    if (counts) await expect(step.getByRole("progressbar", { name: `${label}: ${counts}` })).toBeVisible();
    for (const before of STEPS.keys()) {
      if (before < n) await expect(steps.nth(before).locator(".step-state")).toHaveText("Done");
    }
    if (n === 1) {
      // The feed that checked well but failed in the run, with its reason.
      const feeds = steps.first();
      await expect(feeds).toContainText("2 of 3 loaded, 1 failed:");
      await expect(feeds.getByRole("list", { name: "Feeds that failed" })).toHaveText(`${FLAKY} HTTP 503`);
      await checkPage(page);
    }
  }
}

/** The run's draft: its flagged claim explained beside the claim. */
async function openedDraft(page: Page) {
  await expect(page.getByRole("heading", { level: 2, name: "Newsletter", exact: true })).toBeVisible({
    timeout: 30_000,
  });
  const note = page.getByTestId("draft-text").getByRole("listitem").getByRole("note");
  await expect(note).toContainText("The article says 71; the draft said 95.");
  await expect(note).toContainText("It scored 71 on the reasoning test.");
}

test("a visitor makes a newsletter with their key: feeds, focus, progress, draft, edit, approve, preview", async ({
  page,
}) => {
  const sent: Request[] = [];
  page.on("request", (r) => sent.push(r));

  await page.goto("/");
  const panel = page.getByRole("region", { name: "Model key" });
  await panel.getByLabel("Provider").selectOption("openrouter");
  await panel.getByLabel("API key").fill(KEY);
  await checkPage(page);
  await panel.getByRole("button", { name: "Save" }).click();

  const make = page.getByRole("button", { name: "Make my newsletter" });
  await expect(make).toBeDisabled();
  await addFeed(page, 1, GOOD);
  await expect(row(page, 1)).toContainText("OK, 20 items");
  await addFeed(page, 2, ALSO);
  await expect(row(page, 2)).toContainText("OK, 4 items");
  await addFeed(page, 3, DOWN);
  await expect(row(page, 3)).toContainText("This address did not answer");
  await expect(row(page, 3).getByLabel("Feed 3 address")).toHaveAttribute("aria-invalid", "true");
  await addFeed(page, 4, FLAKY);
  await expect(row(page, 4)).toContainText("OK, 6 items");
  await page.getByLabel(/Focus/).fill(FOCUS);
  await checkPage(page);

  const started = page.waitForRequest((r) => r.url().endsWith("/api/runs") && r.method() === "POST");
  await make.click();
  // The feed that failed its check is left out; the one that checked well goes in.
  expect((await started).postDataJSON()).toEqual({ feeds: [GOOD, ALSO, FLAKY], focus: FOCUS });

  await watchEveryStep(page);
  await openedDraft(page);
  await expect(page.getByTestId("focus")).toHaveText(`Focus: ${FOCUS}`);
  const report = page.getByRole("region", { name: "Run report" });
  await expect(report).toContainText(FLAKY);
  await expect(report).toContainText("HTTP 503");
  await checkPage(page);

  await page.getByRole("button", { name: "Edit" }).click();
  await page.getByLabel("Draft text").fill("# Newsletter\n\nEdited by the visitor.");
  await checkPage(page);
  await page.getByRole("button", { name: "Save" }).click();
  await page.getByRole("button", { name: "Done editing" }).click();
  await expect(page.getByTestId("draft-text")).toContainText("Edited by the visitor.");
  await expect(page.getByTestId("approval")).toHaveText("Not approved");

  await page.getByRole("button", { name: "Approve", exact: true }).click();
  await expect(page.getByTestId("approval")).toHaveText(/^Approved/);
  const preview = page.getByRole("region", { name: "Email preview" });
  await expect(preview).toContainText("Nothing was sent. This is what would go out.");
  await expect(preview.locator("iframe")).toHaveAttribute("sandbox", "");
  await expect(preview.frameLocator("iframe").locator("body")).toContainText("Edited by the visitor.");
  await checkPage(page);

  // The key went only as a header: never in a URL, a body or the page.
  expect(await page.content()).not.toContain(KEY);
  const api = sent.filter((r) => {
    const path = new URL(r.url()).pathname;
    return path.startsWith("/api/") && path !== "/api/providers";
  });
  expect(api.map((r) => new URL(r.url()).pathname)).toEqual(
    expect.arrayContaining(["/api/feeds/check", "/api/runs", "/api/drafts"]),
  );
  for (const r of sent) {
    expect(r.url()).not.toContain(KEY);
    expect(r.postData() ?? "").not.toContain(KEY);
  }
  for (const r of api) {
    const headers = await r.allHeaders();
    expect(headers["x-model-preset"]).toBe("openrouter");
    expect(headers["x-model-key"]).toBe(KEY);
  }
});

test("a visitor with a demo code starts from a preset and gets a draft", async ({ page }) => {
  const sent: Request[] = [];
  page.on("request", (r) => sent.push(r));

  await page.goto("/");
  const panel = page.getByRole("region", { name: "Model key" });
  await panel.getByLabel("Demo code", { exact: true }).fill("e2e-demo-code");
  await panel.getByRole("button", { name: "Save" }).click();

  await page.getByRole("button", { name: "AI news" }).click();
  await expect(row(page, 1).getByLabel("Feed 1 address")).toHaveValue(GOOD);
  await expect(row(page, 2).getByLabel("Feed 2 address")).toHaveValue(ALSO);
  await expect(row(page, 1)).toContainText("OK, 20 items");
  await expect(row(page, 2)).toContainText("OK, 4 items");
  await checkPage(page);

  const started = page.waitForRequest((r) => r.url().endsWith("/api/runs") && r.method() === "POST");
  await page.getByRole("button", { name: "Make my newsletter" }).click();
  expect((await started).postDataJSON()).toEqual({ feeds: [GOOD, ALSO] });

  await expect(page.getByRole("heading", { name: "Making your newsletter…" })).toBeVisible();
  await checkPage(page);
  await openedDraft(page);
  await expect(page.getByTestId("focus")).toHaveCount(0);
  await checkPage(page);

  expect(await page.content()).not.toContain("e2e-demo-code");
  const api = sent.filter((r) => {
    const path = new URL(r.url()).pathname;
    return path.startsWith("/api/") && path !== "/api/providers";
  });
  expect(api.length).toBeGreaterThan(0);
  for (const r of sent) {
    expect(r.url()).not.toContain("e2e-demo-code");
    expect(r.postData() ?? "").not.toContain("e2e-demo-code");
  }
  for (const r of api) {
    const headers = await r.allHeaders();
    expect(headers["x-model-preset"]).toBe("demo");
    expect(headers["x-demo-code"]).toBe("e2e-demo-code");
  }
});
