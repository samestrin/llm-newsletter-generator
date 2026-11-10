// The README's screenshots, saved to `docs/img/` on every run of the browser test. The review page
// is shot in light, dark and at phone width; the issue email once. All show the open-models-week eval issue: `fixtures/draft.json` and `fixtures/issue.html`
// come from `uv run python tests/readme_issue.py`, and pytest checks they are current.
import { readFileSync } from "node:fs";
import { expect, test } from "@playwright/test";

const fixture = (name: string) => new URL(`fixtures/${name}`, import.meta.url);
const image = (name: string) => new URL(`../../docs/img/${name}`, import.meta.url).pathname;
const draft = JSON.parse(readFileSync(fixture("draft.json"), "utf8"));

const shots = [
  { name: "review-page-light.png", colorScheme: "light", viewport: { width: 1280, height: 900 } },
  { name: "review-page-dark.png", colorScheme: "dark", viewport: { width: 1280, height: 900 } },
  { name: "review-page-phone.png", colorScheme: "light", viewport: { width: 360, height: 740 } },
] as const;

for (const shot of shots) {
  test.describe(shot.name, () => {
    test.use({ colorScheme: shot.colorScheme, viewport: shot.viewport });

    test("the review page, with a draft's flags and run report", async ({ request, page }) => {
      await request.post("http://127.0.0.1:8787/__reset", { data: draft });
      await page.goto("/");
      await page.getByLabel("API token").fill("e2e-token");
      await page.getByRole("button", { name: "Connect" }).click();
      await page.getByRole("button", { name: new RegExp(draft.title) }).click();
      await expect(page.getByRole("heading", { level: 2, name: draft.title })).toBeVisible();
      await expect(page.getByRole("region", { name: `Flags (${draft.flags.length})` })).toBeVisible();
      await expect(page.getByRole("region", { name: "Run report" })).toBeVisible();
      await page.screenshot({ path: image(shot.name), fullPage: true });
    });
  });
}

test.use({ viewport: { width: 1280, height: 900 } });

test("an issue, rendered as the HTML email", async ({ page }) => {
  await page.goto(fixture("issue.html").href);
  await expect(page.getByText("Flagged: unsupported").first()).toBeVisible();
  await page.screenshot({ path: image("issue.png"), fullPage: true });
});
