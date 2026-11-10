// The design tokens in styles.css: every text color on every background it is used on
// reaches 4.5:1 (WCAG AA for body text) in light and in dark mode, and the focus ring
// reaches 3:1 (non-text contrast). axe in the browser test checks the rendered page too.
import { readFileSync } from "node:fs";
import { describe, expect, it } from "vitest";

const css = readFileSync(new URL("./styles.css", import.meta.url), "utf8");

/** The `--color-*` tokens declared in the first `:root { ... }` block of `source`. */
function tokens(source: string): Record<string, string> {
  const block = /:root\s*{([^}]*)}/.exec(source)?.[1] ?? "";
  return Object.fromEntries([...block.matchAll(/--color-([\w-]+):\s*(#[0-9a-f]{6})\s*;/gi)].map((m) => [m[1], m[2]]));
}

const light = tokens(css);
const darkBlock = /@media\s*\(prefers-color-scheme:\s*dark\)\s*{([\s\S]*?})\s*}/.exec(css);
const dark = { ...light, ...tokens(darkBlock?.[1] ?? "") };

function luminance(hex: string) {
  const [r, g, b] = [1, 3, 5].map((i) => {
    const c = parseInt(hex.slice(i, i + 2), 16) / 255;
    return c <= 0.03928 ? c / 12.92 : ((c + 0.055) / 1.055) ** 2.4;
  });
  return 0.2126 * r + 0.7152 * g + 0.0722 * b;
}

function contrast(a: string, b: string) {
  const [hi, lo] = [luminance(a), luminance(b)].sort((x, y) => y - x);
  return (hi + 0.05) / (lo + 0.05);
}

const TEXT = ["text", "text-muted", "accent", "good", "bad"];
const BACKGROUNDS = ["bg", "surface", "flag-bg"];

describe.each([
  ["light", light],
  ["dark", dark],
])("%s mode tokens", (_, colors) => {
  it("declares every token the checks need", () => {
    for (const name of [...TEXT, ...BACKGROUNDS, "on-accent", "focus", "border"]) {
      expect(colors[name], `--color-${name}`).toMatch(/^#[0-9a-f]{6}$/i);
    }
  });

  it.each(TEXT.flatMap((text) => BACKGROUNDS.map((bg) => [text, bg])))("%s on %s reaches 4.5:1", (text, bg) => {
    expect(contrast(colors[text], colors[bg])).toBeGreaterThanOrEqual(4.5);
  });

  it("button text on the accent color reaches 4.5:1", () => {
    expect(contrast(colors["on-accent"], colors.accent)).toBeGreaterThanOrEqual(4.5);
  });

  it.each(BACKGROUNDS)("the focus ring on %s reaches 3:1", (bg) => {
    expect(contrast(colors.focus, colors[bg])).toBeGreaterThanOrEqual(3);
  });
});

it("dark mode changes the background, not only light mode's", () => {
  expect(darkBlock, "a prefers-color-scheme: dark block").not.toBeNull();
  expect(dark.bg).not.toBe(light.bg);
});
