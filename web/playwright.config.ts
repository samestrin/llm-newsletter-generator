import { defineConfig, devices } from "@playwright/test";

const API_PORT = 8787;
const PAGE_PORT = 4173;
const PAGE = `http://127.0.0.1:${PAGE_PORT}`;

// The built page, served by `vite preview`, whose `/api` proxy goes to the fake API.
export default defineConfig({
  testDir: "e2e",
  workers: 1,
  forbidOnly: !!process.env.CI,
  retries: 0,
  reporter: process.env.CI ? [["list"], ["html", { open: "never" }]] : "list",
  use: { baseURL: PAGE, trace: "retain-on-failure" },
  // The review page is checked at desktop and phone (360 px) widths, in light and dark.
  // The README screenshots are taken once, at the desktop width in light mode.
  projects: [
    { name: "screenshots", testMatch: "screenshots.spec.ts", use: { ...devices["Desktop Chrome"] } },
    ...(["light", "dark"] as const).flatMap((colorScheme) => [
      {
        name: `desktop-${colorScheme}`,
        testIgnore: "screenshots.spec.ts",
        use: { ...devices["Desktop Chrome"], colorScheme },
      },
      {
        name: `phone-${colorScheme}`,
        testIgnore: "screenshots.spec.ts",
        use: { ...devices["Desktop Chrome"], viewport: { width: 360, height: 740 }, colorScheme },
      },
    ]),
  ],
  webServer: [
    {
      command: "node e2e/fake-api.mjs",
      url: `http://127.0.0.1:${API_PORT}/api/drafts`,
      env: { FAKE_API_PORT: String(API_PORT), FAKE_API_ORIGINS: PAGE },
      reuseExistingServer: false,
    },
    {
      command: `npx vite preview --host 127.0.0.1 --port ${PAGE_PORT} --strictPort`,
      url: PAGE,
      env: { NEWSLETTER_API_URL: `http://127.0.0.1:${API_PORT}` },
      reuseExistingServer: false,
    },
  ],
});
