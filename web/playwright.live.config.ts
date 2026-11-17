import { defineConfig, devices } from "@playwright/test";

// Records the README demo GIF from a real, running instance (`npm run capture:live`).
// Nothing is mocked: the page and the API are live, and the instance must already hold
// the seeded draft (`newsletter serve --seed-sample`). LIVE_URL is the instance's address.
const url = process.env.LIVE_URL;
if (!url) throw new Error("LIVE_URL is required, for example LIVE_URL=http://127.0.0.1:8080");

export default defineConfig({
  testDir: "e2e/live",
  // Only the recording; frames.test.ts is a vitest unit test (`npm test` runs it).
  testMatch: "demo.spec.ts",
  workers: 1,
  // No retries: the recording approves the seeded draft, so a second attempt would
  // find it already approved and no Approve button to click.
  retries: 0,
  timeout: 5 * 60 * 1000,
  reporter: "list",
  use: { baseURL: url, trace: "retain-on-failure" },
  // Every frame is 1280x800; checkFrames (e2e/live/frames.ts) refuses anything else.
  projects: [
    { name: "chromium", use: { ...devices["Desktop Chrome"], viewport: { width: 1280, height: 800 } } },
  ],
});
