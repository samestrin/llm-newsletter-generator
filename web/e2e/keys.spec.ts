import AxeBuilder from "@axe-core/playwright";
import { expect, test } from "@playwright/test";

const KEY = "sk-or-e2e-key";

test.beforeEach(async ({ request }) => {
  await request.post("http://127.0.0.1:8787/__reset");
  await request.post("http://127.0.0.1:8787/__keys");
});

test("the key panel opens on the first visit and passes axe", async ({ page }) => {
  await page.goto("/");
  const panel = page.getByRole("region", { name: "Model key" });
  await expect(panel).toBeVisible();
  await expect(panel).toContainText("The key stays in this browser and is sent only with your requests.");
  await expect(page.getByLabel("API token")).toHaveCount(0);
  await panel.getByLabel("Provider").selectOption("synthetic");
  await expect(panel).toContainText("Synthetic has no embedding model: stories will not be grouped by meaning.");
  const { violations } = await new AxeBuilder({ page }).analyze();
  expect(violations.filter((v) => v.impact === "serious" || v.impact === "critical")).toEqual([]);
});

test("a saved key lists the drafts, goes only in headers and is never on the page", async ({ page }) => {
  const sent: { url: string; body: string | null; headers: Record<string, string> }[] = [];
  page.on("request", (r) => sent.push({ url: r.url(), body: r.postData(), headers: r.headers() }));
  await page.goto("/");
  const panel = page.getByRole("region", { name: "Model key" });
  await panel.getByLabel("Provider").selectOption("openrouter");
  await panel.getByLabel("API key").fill(KEY);
  await panel.getByRole("button", { name: "Save" }).click();
  await page.getByRole("button", { name: /AI Weekly/ }).click();
  await expect(page.getByRole("heading", { level: 2, name: "AI Weekly" })).toBeVisible();
  expect(await page.content()).not.toContain(KEY);
  const api = sent.filter((r) => new URL(r.url).pathname.startsWith("/api/drafts"));
  expect(api.length).toBeGreaterThan(0);
  for (const r of sent) {
    expect(r.url).not.toContain(KEY);
    expect(r.body ?? "").not.toContain(KEY);
  }
  for (const r of api) expect(r.headers["x-model-key"]).toBe(KEY);
});

test("a refused demo code shows the API's message in the panel", async ({ page }) => {
  await page.goto("/");
  const panel = page.getByRole("region", { name: "Model key" });
  await panel.getByLabel("Demo code", { exact: true }).fill("wrong-code");
  await panel.getByRole("button", { name: "Save" }).click();
  await expect(page.getByRole("region", { name: "Model key" }).getByRole("alert")).toHaveText(
    "X-Demo-Code was not accepted",
  );
});
