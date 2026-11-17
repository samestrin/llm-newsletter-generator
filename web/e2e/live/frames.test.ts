// The frame check the live capture runs before it writes the GIF (demo.spec.ts).
// Real PNG buffers, so the check is tested on what a screenshot actually produces.
import { describe, expect, test } from "vitest";
import { PNG } from "pngjs";
import { checkFrames, FRAME_HEIGHT, FRAME_WIDTH, MAX_TOTAL_DELAY_MS, type Frame } from "./frames";

function png(width: number, height: number): Buffer {
  const image = new PNG({ width, height });
  image.data.fill(128);
  return PNG.sync.write(image);
}

const frame = (delay: number, width = FRAME_WIDTH, height = FRAME_HEIGHT): Frame => ({
  png: png(width, height),
  delay,
});

describe("checkFrames", () => {
  test("same-size 1280x800 frames within the delay budget pass", () => {
    expect(() => checkFrames([frame(4000), frame(6000), frame(4000)])).not.toThrow();
  });

  test("a total delay of exactly 30 s passes", () => {
    expect(() => checkFrames([frame(MAX_TOTAL_DELAY_MS / 2), frame(MAX_TOTAL_DELAY_MS / 2)])).not.toThrow();
  });

  test("mixed sizes are refused, and the error names them", () => {
    expect(() => checkFrames([frame(4000), frame(4000, FRAME_WIDTH, FRAME_HEIGHT + 1)])).toThrow(
      /differ in size \(1280x800, 1280x801\)/,
    );
  });

  test("one size that is not 1280x800 is refused", () => {
    expect(() => checkFrames([frame(4000, 640, 400), frame(4000, 640, 400)])).toThrow(/640x400, not 1280x800/);
  });

  test("a total delay over 30 s is refused", () => {
    expect(() => checkFrames([frame(MAX_TOTAL_DELAY_MS - 1), frame(2)])).toThrow(/over the 30000 ms limit/);
  });

  test("no frames is refused", () => {
    expect(() => checkFrames([])).toThrow(/no frames/);
  });

  test("a negative or non-finite delay is refused", () => {
    expect(() => checkFrames([frame(-1)])).toThrow(/delay/);
    expect(() => checkFrames([frame(Number.NaN)])).toThrow(/delay/);
    expect(() => checkFrames([frame(Number.POSITIVE_INFINITY)])).toThrow(/delay/);
  });
});
