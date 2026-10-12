import type { Browser } from "@playwright/test";

/** What `pixelDiff` found. */
export interface PixelDiff {
  /** Share of pixels (0–1) whose largest channel differs by more than `threshold`. */
  mismatch: number;
  /** Number of pixels that differ. */
  pixels: number;
  width: number;
  height: number;
}

/**
 * Compares two PNG screenshots of the same size in the browser's own canvas, so the e2e suite needs no
 * image library. `region` limits the comparison to a rectangle in screenshot pixels.
 */
export async function pixelDiff(
  browser: Browser,
  a: Buffer,
  b: Buffer,
  options: { threshold?: number; region?: { x: number; y: number; width: number; height: number } } = {},
): Promise<PixelDiff> {
  const context = await browser.newContext();
  try {
    const page = await context.newPage();
    return await page.evaluate(
      async ({ a64, b64, threshold, region }) => {
        const load = async (b64png: string): Promise<ImageData> => {
          const blob = await (await fetch(`data:image/png;base64,${b64png}`)).blob();
          const bitmap = await createImageBitmap(blob);
          const canvas = new OffscreenCanvas(bitmap.width, bitmap.height);
          const ctx = canvas.getContext("2d");
          if (!ctx) throw new Error("no 2d context");
          ctx.drawImage(bitmap, 0, 0);
          return ctx.getImageData(0, 0, bitmap.width, bitmap.height);
        };
        const [first, second] = await Promise.all([load(a64), load(b64)]);
        if (first.width !== second.width || first.height !== second.height) {
          throw new Error(`size differs: ${first.width}x${first.height} against ${second.width}x${second.height}`);
        }
        const x0 = region?.x ?? 0;
        const y0 = region?.y ?? 0;
        const x1 = x0 + (region?.width ?? first.width - x0);
        const y1 = y0 + (region?.height ?? first.height - y0);
        let pixels = 0;
        for (let y = y0; y < y1; y++) {
          for (let x = x0; x < x1; x++) {
            const i = (y * first.width + x) * 4;
            const d = Math.max(
              Math.abs((first.data[i] as number) - (second.data[i] as number)),
              Math.abs((first.data[i + 1] as number) - (second.data[i + 1] as number)),
              Math.abs((first.data[i + 2] as number) - (second.data[i + 2] as number)),
            );
            if (d > threshold) pixels++;
          }
        }
        return { mismatch: pixels / ((x1 - x0) * (y1 - y0)), pixels, width: first.width, height: first.height };
      },
      { a64: a.toString("base64"), b64: b.toString("base64"), threshold: options.threshold ?? 24, region: options.region },
    );
  } finally {
    await context.close();
  }
}
