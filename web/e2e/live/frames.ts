// The check the live capture runs before it writes the README GIF (demo.spec.ts):
// every frame is 1280x800, one size only, and the delays total at most 30 s.
// Unit-tested in frames.test.ts, which `npm test` runs; the Playwright configs skip it.
import { PNG } from "pngjs";

export const FRAME_WIDTH = 1280;
export const FRAME_HEIGHT = 800;
export const MAX_TOTAL_DELAY_MS = 30_000;

export type Frame = { png: Buffer; delay: number };

/** Throws when the frames cannot become the README GIF; says why. */
export function checkFrames(frames: Frame[]): void {
  if (frames.length === 0) throw new Error("no frames recorded; refusing to write an empty GIF");
  for (const { delay } of frames) {
    if (!Number.isFinite(delay) || delay < 0) {
      throw new Error(`a frame delay is ${delay}, not a finite number of milliseconds >= 0`);
    }
  }
  const sizes = new Set(
    frames.map(({ png }) => {
      const { width, height } = PNG.sync.read(png);
      return `${width}x${height}`;
    }),
  );
  if (sizes.size !== 1) {
    throw new Error(`GIF frames differ in size (${[...sizes].join(", ")}); refusing to write the GIF`);
  }
  const wanted = `${FRAME_WIDTH}x${FRAME_HEIGHT}`;
  if (!sizes.has(wanted)) {
    throw new Error(`GIF frames are ${[...sizes][0]}, not ${wanted}; refusing to write the GIF`);
  }
  const total = frames.reduce((sum, { delay }) => sum + delay, 0);
  if (total > MAX_TOTAL_DELAY_MS) {
    throw new Error(`GIF delays total ${total} ms, over the ${MAX_TOTAL_DELAY_MS} ms limit; refusing to write the GIF`);
  }
}
