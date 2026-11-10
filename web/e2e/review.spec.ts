import AxeBuilder from "@axe-core/playwright";
import { expect, test, type Locator, type Page } from "@playwright/test";

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
  const approve = page.getByRole("button", { name: "Approve", exact: true });
  await openDraft(page);

  await page.getByRole("button", { name: "Edit" }).click();
  await page.getByLabel("Draft text").fill("# AI Weekly\n\nEdited by the reviewer.");
  await expect(approve).toBeDisabled();
  await page.getByRole("button", { name: "Save" }).click();
  await expect(page.getByTestId("draft-text")).toContainText("Edited by the reviewer.");
  await expect(page.getByTestId("approval")).toHaveText("Not approved");

  await approve.click();
  await expect(page.getByTestId("approval")).toHaveText(/^Approved/);

  // A fresh load reads it back from the API: the edit and the approval were stored.
  await openDraft(page);
  await expect(page.getByTestId("draft-text")).toContainText("Edited by the reviewer.");
  await expect(page.getByTestId("approval")).toHaveText(/^Approved/);
});

test("an edit after approval clears it", async ({ page }) => {
  await openDraft(page);
  await page.getByRole("button", { name: "Approve", exact: true }).click();
  await expect(page.getByTestId("approval")).toHaveText(/^Approved/);

  await page.getByRole("button", { name: "Edit" }).click();
  await page.getByLabel("Draft text").fill("Changed after approval.");
  await page.getByRole("button", { name: "Save" }).click();
  await expect(page.getByTestId("approval")).toHaveText("Not approved");

  await openDraft(page);
  await expect(page.getByTestId("approval")).toHaveText("Not approved");
});

/** Every serious or critical axe finding on the page as it is now, one line each. */
async function axeIssues(page: Page) {
  const { violations } = await new AxeBuilder({ page }).analyze();
  return violations
    .filter((v) => v.impact === "serious" || v.impact === "critical")
    .map((v) => `${v.id} (${v.impact}): ${v.nodes.map((n) => n.target.join(" ")).join(", ")}`);
}

/** Tab forward until `target` has focus; fails if it takes more than `limit` presses. */
async function tabTo(page: Page, target: Locator, limit = 15) {
  for (let n = 0; n < limit; n++) {
    await page.keyboard.press("Tab");
    if (await target.evaluate((el) => el === document.activeElement)) return;
  }
  throw new Error(`Tab never reached ${target}`);
}

async function expectFocusRing(target: Locator) {
  await expect(target).toBeFocused();
  const ring = await target.evaluate((el) => {
    const style = getComputedStyle(el);
    return { style: style.outlineStyle, width: parseFloat(style.outlineWidth) };
  });
  expect(ring.style).not.toBe("none");
  expect(ring.width).toBeGreaterThanOrEqual(2);
}

test("axe finds no serious or critical issues on the token form, a draft and the editor", async ({ page }) => {
  await page.goto("/");
  await expect(page.getByLabel("API token")).toBeVisible();
  expect(await axeIssues(page)).toEqual([]);

  await openDraft(page);
  expect(await axeIssues(page)).toEqual([]);

  await page.getByRole("button", { name: "Edit" }).click();
  await expect(page.getByLabel("Draft text")).toBeVisible();
  expect(await axeIssues(page)).toEqual([]);
});

test("the colors follow the system light or dark setting", async ({ page }, testInfo) => {
  const scheme = testInfo.project.use.colorScheme ?? "light";
  await openDraft(page);
  // Relative brightness (0 black, 1 white) of the page background and of its text.
  const [background, text] = await page.evaluate(() => {
    const brightness = (color: string) => {
      const [r, g, b] = color.match(/\d+/g)!.map(Number);
      return (0.2126 * r + 0.7152 * g + 0.0722 * b) / 255;
    };
    const style = getComputedStyle(document.documentElement);
    return [brightness(style.backgroundColor), brightness(style.color)];
  });
  if (scheme === "dark") {
    expect(background).toBeLessThan(0.2);
    expect(text).toBeGreaterThan(0.8);
  } else {
    expect(background).toBeGreaterThan(0.8);
    expect(text).toBeLessThan(0.2);
  }
});

test("the page fits the window width, with nothing scrolling sideways", async ({ page }) => {
  const fits = () =>
    page.evaluate(() => document.documentElement.scrollWidth <= document.documentElement.clientWidth);
  await page.goto("/");
  expect(await fits()).toBe(true);
  await openDraft(page);
  expect(await fits()).toBe(true);
  await page.getByRole("button", { name: "Edit" }).click();
  expect(await fits()).toBe(true);
});

test("the page has a banner, the drafts navigation and a main landmark", async ({ page }) => {
  await openDraft(page);
  await expect(page.getByRole("banner")).toContainText("Newsletter review");
  await expect(page.getByRole("navigation", { name: "Drafts" })).toBeVisible();
  await expect(page.getByRole("main")).toContainText("AI Weekly");
});

test("a keyboard user can skip to the content, connect and open a draft, with focus always shown", async ({
  page,
}) => {
  await page.goto("/");
  const skip = page.getByRole("link", { name: "Skip to main content" });
  await page.keyboard.press("Tab");
  await expectFocusRing(skip);
  await expect(skip).toBeInViewport();
  await page.keyboard.press("Enter");
  await expect(page.getByRole("main")).toBeFocused();

  const token = page.getByLabel("API token");
  await page.keyboard.press("Tab");
  await expectFocusRing(token);
  await page.keyboard.type("e2e-token");
  await page.keyboard.press("Enter");

  const item = page.getByRole("button", { name: /AI Weekly/ });
  await tabTo(page, item);
  await expectFocusRing(item);
  await page.keyboard.press("Enter");
  await expect(page.getByRole("heading", { level: 2, name: "AI Weekly" })).toBeVisible();

  await tabTo(page, page.getByRole("button", { name: "Edit" }));
  await expectFocusRing(page.getByRole("button", { name: "Edit" }));
});

test("a flagged claim is marked by text and an icon, not by color alone", async ({ page }) => {
  await openDraft(page);
  const flag = page.getByRole("region", { name: "Flags (1)" }).getByRole("listitem");
  await expect(flag).toContainText("Flagged: unsupported");
  await expect(flag.getByTestId("flag-icon")).toBeVisible();
});
