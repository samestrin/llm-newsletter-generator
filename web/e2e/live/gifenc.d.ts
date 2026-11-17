// gifenc ships no TypeScript types; this declares the three functions the live
// capture uses (e2e/live/demo.spec.ts), as llm-pdf-chatbot does.
declare module "gifenc" {
  const gifenc: {
    GIFEncoder: typeof GIFEncoder;
    quantize: typeof quantize;
    applyPalette: typeof applyPalette;
  };
  export default gifenc;
  export function GIFEncoder(): {
    writeFrame(
      index: Uint8Array,
      width: number,
      height: number,
      opts: { palette: number[][]; delay: number },
    ): void;
    finish(): void;
    bytes(): Uint8Array;
  };
  export function quantize(data: Uint8Array | Uint8ClampedArray, maxColors: number): number[][];
  export function applyPalette(data: Uint8Array | Uint8ClampedArray, palette: number[][]): Uint8Array;
}
