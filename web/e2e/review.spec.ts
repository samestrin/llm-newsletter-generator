import { expect, test, type Page } from "@playwright/test";

const MARKUP = '<img src=x onerror="alert(1)"><script>alert(2)</script>';

async function openDraft(page: Page) {
  await page.goto("/");
  await page.getByLabel("API token").fill("e2e-token");
  await page.getByRole("button", { name: "Connect" }).click();
  await page.getByRole("button", { name: /AI Weekly/ }).click();
  await expect(page.getByRole("heading", { level: 2, name: "AI Weekly" })).toBeVisible();
}

test.beforeEach(async ({ request, page }) => {
  await request.post("http://127.0.0.1:8787/__reset");
  page.on("dialog", (dialog) => {
    throw new Error(`the draft ran script: ${dialog.message()}`);
  });
});

test("a draft containing markup is shown as text", async ({ page }) => {
  await openDraft(page);
  await expect(page.getByTestId("draft-text")).toContainText(MARKUP);
  await expect(page.locator("main img, main script")).toHaveCount(0);
});

test("flags and the run report are visible", async ({ page }) => {
  await openDraft(page);
  const flags = page.getByRole("region", { name: "Flags (1)" });
  await expect(flags).toBeVisible();
  await expect(flags).toContainText("The model scored 99% on every benchmark.");
  await expect(flags).toContainText("The source gives 91%.");
  const report = page.getByRole("region", { name: "Run report" });
  await expect(report).toBeVisible();
  await expect(report).toContainText("https://down.example.com/feed.xml");
  await expect(report).toContainText("fast-model");
});

test("edit, then approve, and both are kept by the API", async ({ page }) => {
  await openDraft(page);
  await page.getByRole("button", { name: "Approve" }).click();
  await expect(page.getByTestId("approval")).toHaveText(/^Approved/);

  await page.getByRole("button", { name: "Edit" }).click();
  await page.getByLabel("Draft text").fill("# AI Weekly\n\nEdited by the reviewer.");
  await expect(page.getByRole("button", { name: "Approve" })).toBeDisabled();
  await page.getByRole("button", { name: "Save" }).click();
  await expect(page.getByTestId("draft-text")).toContainText("Edited by the reviewer.");
  await expect(page.getByTestId("approval")).toHaveText("Not approved");

  await page.getByRole("button", { name: "Approve" }).click();
  await expect(page.getByTestId("approval")).toHaveText(/^Approved/);

  // A fresh load reads it back from the API: the edit and the approval were stored.
  await openDraft(page);
  await expect(page.getByTestId("draft-text")).toContainText("Edited by the reviewer.");
  await expect(page.getByTestId("approval")).toHaveText(/^Approved/);
});
