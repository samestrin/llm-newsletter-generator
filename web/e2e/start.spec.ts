import AxeBuilder from "@axe-core/playwright";
import { expect, test, type Page } from "@playwright/test";

const API = "http://127.0.0.1:8787";

/** Save the demo code; the page then opens on the start form. */
async function openForm(page: Page) {
  await page.goto("/");
  const panel = page.getByRole("region", { name: "Model key" });
  await panel.getByLabel("Demo code", { exact: true }).fill("e2e-demo-code");
  await panel.getByRole("button", { name: "Save" }).click();
  await expect(page.getByRole("button", { name: "Make my newsletter" })).toBeVisible();
}

const row = (page: Page, n: number) => page.getByRole("group", { name: `Feed ${n}` });

// Runs stay on their first step here; flow.spec.ts watches one go all the way.
test.beforeEach(async ({ request }) => {
  await request.post(`${API}/__reset`);
  await request.post(`${API}/__keys`);
  await request.post(`${API}/__hold`);
});

test("the form checks each feed, then starts a run with the focus", async ({ page }) => {
  await openForm(page);
  const make = page.getByRole("button", { name: "Make my newsletter" });
  await expect(make).toBeDisabled();
  await row(page, 1).getByLabel("Feed 1 address").fill("https://page.example.com/");
  await row(page, 1).getByLabel("Feed 1 address").blur();
  await expect(row(page, 1)).toContainText("This is a web page, not a feed");
  await page.getByRole("button", { name: "AI news" }).click();
  await expect(row(page, 1)).toContainText("OK, 20 items");
  await expect(row(page, 2)).toContainText("OK, 4 items");
  await page.getByRole("button", { name: "Add a feed" }).click();
  await row(page, 3).getByLabel("Feed 3 address").fill("https://flaky.example.com/feed.xml");
  await page.getByLabel(/Focus/).fill("practical AI tools");
  const { violations } = await new AxeBuilder({ page }).analyze();
  expect(violations.filter((v) => v.impact === "serious" || v.impact === "critical")).toEqual([]);
  const started = page.waitForRequest((r) => r.url().endsWith("/api/runs") && r.method() === "POST");
  await make.click();
  expect((await started).postDataJSON()).toEqual({
    feeds: ["https://good.example.com/feed.xml", "https://also.example.com/rss", "https://flaky.example.com/feed.xml"],
    focus: "practical AI tools",
  });
  await expect(page.getByRole("heading", { name: "Making your newsletter…" })).toBeVisible();
  const feeds = page.getByRole("list", { name: "Steps" }).getByRole("listitem").first();
  await expect(feeds).toContainText("Fetching feeds");
  await expect(feeds).toContainText("Running");
  await expect(feeds.getByRole("list", { name: "Feeds that failed" })).toContainText("https://flaky.example.com/feed.xml HTTP 503");
  await expect(page.getByRole("progressbar", { name: "Fetching feeds: 3 of 3" })).toBeVisible();
  await expect(page.locator("body")).not.toContainText("e2e-owner-token");
  const after = await new AxeBuilder({ page }).analyze();
  expect(after.violations.filter((v) => v.impact === "serious" || v.impact === "critical")).toEqual([]);
  // A reload resumes the same run from its stored id.
  await page.reload();
  await expect(page.getByRole("heading", { name: "Making your newsletter…" })).toBeVisible();
  expect(await page.evaluate(() => sessionStorage.getItem("newsletter.run"))).toBe("run00000000000000000000000000001");
});

test("a busy server shows the friendly text and a retry", async ({ page, request }) => {
  await request.post(`${API}/__busy`);
  await openForm(page);
  await page.getByRole("button", { name: "AI news" }).click();
  await expect(row(page, 1)).toContainText("OK, 20 items");
  await page.getByRole("button", { name: "Make my newsletter" }).click();
  const alert = page.getByRole("alert");
  await expect(alert).toContainText("Someone else is making one right now. Try again in a minute.");
  await expect(alert.getByRole("button", { name: "Try again" })).toBeEnabled();
});

test("See an example opens the seeded draft, and Earlier drafts lists it", async ({ page }) => {
  await openForm(page);
  await expect(page.getByRole("navigation", { name: "Earlier drafts" }).getByRole("button", { name: /AI Weekly/ })).toBeVisible();
  await page.getByRole("button", { name: "See an example" }).click();
  await expect(page.getByRole("heading", { level: 2, name: "AI Weekly" })).toBeVisible();
});
