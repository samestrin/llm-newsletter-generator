// Records the README demo GIF from a real, running instance (`npm run capture:live`,
// LIVE_URL set; nothing is mocked). Follows llm-pdf-chatbot's frontend/e2e/live/demo.spec.ts:
// the drafts list, the seeded issue (T3's `serve --seed-sample`), a flagged claim with its
// citation, the approval, and the API reference at /api/docs. The recording clicks Approve
// on the seeded draft; the operator restarts the stack or deletes the draft afterwards so
// the next visitor sees it unapproved (epic 07 plan, AC3).
//
// LIVE_TOKEN is the instance's NEWSLETTER_API_TOKEN (the page asks for it; it defaults to
// the local demo token). Behind Authelia, run this from a session that already reaches the
// page: a login page in a frame fails the recording.
import { writeFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { expect, test } from "@playwright/test";
import gifenc from "gifenc";
import { PNG } from "pngjs";
import { checkFrames, type Frame } from "./frames";

const { GIFEncoder, applyPalette, quantize } = gifenc;

const OUT = fileURLToPath(new URL("../../../docs/img/demo.gif", import.meta.url));
const BASE = process.env.LIVE_URL!; // playwright.live.config.ts refuses to load without it
const TOKEN = process.env.LIVE_TOKEN ?? "demo-token-change-me";

test("records the README demo GIF from a live instance", async ({ page }) => {
  const frames: Frame[] = [];
  const snap = async (delay: number) => frames.push({ png: await page.screenshot(), delay });

  // 1. The drafts list. A redirect off the instance's origin means a login page.
  await page.goto("/");
  if (new URL(page.url()).origin !== new URL(BASE).origin) {
    throw new Error(`LIVE_URL redirected to ${page.url()}; the capture needs a session that reaches the page`);
  }
  await expect(page.getByRole("heading", { name: "Newsletter review" })).toBeVisible();
  await page.getByLabel("API token").fill(TOKEN);
  await page.getByRole("button", { name: "Connect" }).click();
  const drafts = page.getByRole("navigation", { name: "Drafts" });
  await expect(drafts.getByRole("listitem").first()).toBeVisible();
  await snap(4000);

  // 2. The seeded issue: the review instance holds one draft (serve --seed-sample).
  await drafts.getByRole("listitem").first().getByRole("button").click();
  await expect(page.getByRole("main").getByRole("heading", { level: 2 })).toBeVisible();
  await expect(page.getByTestId("draft-text")).toBeVisible();
  await snap(4000);

  // 3. A flagged claim and its citation, scrolled into the frame.
  const flag = page.getByRole("region", { name: /^Flags \([1-9]/ }).getByRole("listitem").first();
  await expect(flag).toContainText("Flagged:");
  await expect(flag).toContainText("Source:");
  await expect(flag.getByRole("link").first()).toBeVisible();
  await flag.scrollIntoViewIfNeeded();
  await snap(6000);

  // 4. Approve, with the approval line back in the frame. An already-approved draft
  // means a previous recording was not cleaned up (restart the stack or delete the draft).
  if ((await page.getByTestId("approval").innerText()).startsWith("Approved")) {
    throw new Error("the draft is already approved; restart the stack or delete the draft so it can be recorded");
  }
  await page.getByRole("button", { name: "Approve", exact: true }).click();
  await expect(page.getByTestId("approval")).toHaveText(/^Approved/);
  await page.evaluate(() => window.scrollTo(0, 0));
  await snap(4000);

  // 5. The API reference (epic 07 T1): Scalar, not the SPA page, with the spec's
  // routes rendered (the sidebar names each operation, e.g. "List Drafts").
  await page.goto("/api/docs");
  await expect(page).toHaveTitle(/API reference/);
  await expect(page.locator("#root")).toHaveCount(0);
  await expect(page.getByText("List Drafts").first()).toBeVisible({ timeout: 60_000 });
  await snap(6000);

  // One size (1280x800), at most 30 s of delay, or no GIF.
  checkFrames(frames);

  const gif = GIFEncoder();
  for (const { png, delay } of frames) {
    const { data, width, height } = PNG.sync.read(png);
    const palette = quantize(data, 256);
    gif.writeFrame(applyPalette(data, palette), width, height, { palette, delay });
  }
  gif.finish();
  writeFileSync(OUT, gif.bytes());
});
