// @vitest-environment node
// The photo fixtures read files (the SVGs and the mockup they came from), so they run in Node.
import { readFileSync } from "node:fs";
import { inflateSync } from "node:zlib";
import { describe, expect, it } from "vitest";
import { encodeIndexedPng, hexToRgb, scenePng } from "./png";
import { layerAt, SCENES, sceneSvg, SUN_FRACTION_THUMB, type Scene } from "./scenes";

const root = new URL("../../", import.meta.url);
const read = (path: string): string => readFileSync(new URL(path, root), "utf8");

describe("scenes", () => {
  it("are the twelve photos of design/mockups/photos.html", () => {
    const html = read("design/mockups/photos.html");
    const pattern =
      /\{ name: '(DSC\d+)', sky: '(#\w+)', far: '(#\w+)', near: '(#\w+)', sun: '(#\w+)', sx: (\d+), sy: (\d+) \}/g;
    const fromMockup: Scene[] = [...html.matchAll(pattern)].map((m) => ({
      name: m[1] as string,
      sky: m[2] as string,
      far: m[3] as string,
      near: m[4] as string,
      sun: m[5] as string,
      sx: Number(m[6]),
      sy: Number(m[7]),
    }));
    expect(fromMockup).toHaveLength(12);
    expect(SCENES).toEqual(fromMockup);
  });

  it("have unique names DSC04201 to DSC04212", () => {
    expect(SCENES.map((s) => s.name)).toEqual(Array.from({ length: 12 }, (_, i) => `DSC042${String(i + 1).padStart(2, "0")}`));
  });

  it("layer the shapes in the mockup's order: near over far over sun over sky", () => {
    const scene = SCENES[3] as Scene; // DSC04204: sun at 66 %, 12 %
    expect(layerAt(scene, 0.5, 0.95)).toBe("near");
    expect(layerAt(scene, 0.5, 0.7)).toBe("far");
    expect(layerAt(scene, 0.02, 0.5)).toBe("sky");
    // Centre of the sun: left 66 % plus half of 9 % of the width; top 12 % plus its radius.
    const u = 0.66 + 0.045;
    const v = 0.12 + (0.09 * 1.5) / 2;
    expect(layerAt(scene, u, v)).toBe("sun");
  });
});

describe("SVG fixtures", () => {
  it.each(SCENES)("$name.svg is generated from the scene data", (scene) => {
    expect(read(`src/fixtures/${scene.name}.svg`)).toBe(sceneSvg(scene));
  });

  it("are 3:2 documents with the four shapes", () => {
    const svg = sceneSvg(SCENES[0] as Scene);
    expect(svg).toContain('viewBox="0 0 1500 1000"');
    expect(svg.match(/<(rect|circle|path)\b/g)).toHaveLength(4);
  });
});

/** Walks the chunks of a PNG, checking each CRC with Node's zlib as the independent reference. */
function readPng(bytes: Uint8Array) {
  const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
  const chunks: Record<string, Uint8Array> = {};
  let pos = 8;
  while (pos < bytes.length) {
    const length = view.getUint32(pos);
    const type = String.fromCharCode(...bytes.subarray(pos + 4, pos + 8));
    const data = bytes.subarray(pos + 8, pos + 8 + length);
    const crc = view.getUint32(pos + 8 + length);
    const expected = crcOf(bytes.subarray(pos + 4, pos + 8 + length));
    expect(crc, `CRC of ${type}`).toBe(expected);
    chunks[type] = data;
    pos += 12 + length;
  }
  const ihdr = new DataView(chunks.IHDR!.buffer, chunks.IHDR!.byteOffset, 13);
  return {
    chunks,
    width: ihdr.getUint32(0),
    height: ihdr.getUint32(4),
    bitDepth: ihdr.getUint8(8),
    colorType: ihdr.getUint8(9),
    pixels: inflateSync(chunks.IDAT!),
  };
}

function crcOf(bytes: Uint8Array): number {
  // zlib.crc32 exists from Node 22.2; fall back to a bitwise version otherwise.
  let crc = 0xffffffff;
  for (const byte of bytes) {
    crc ^= byte;
    for (let k = 0; k < 8; k++) crc = crc & 1 ? 0xedb88320 ^ (crc >>> 1) : crc >>> 1;
  }
  return (crc ^ 0xffffffff) >>> 0;
}

describe("PNG thumbnails", () => {
  const scene = SCENES[3] as Scene;
  const png = scenePng(scene);
  const decoded = readPng(png);

  it("start with the PNG signature and end with IEND", () => {
    expect([...png.subarray(0, 8)]).toEqual([0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a]);
    expect(Object.keys(decoded.chunks)).toEqual(["IHDR", "PLTE", "IDAT", "IEND"]);
  });

  it("are 240 × 160 indexed colour, within the 240 px bridge limit", () => {
    expect(decoded.width).toBe(240);
    expect(decoded.height).toBe(160);
    expect(decoded.colorType).toBe(3);
    expect(Math.max(decoded.width, decoded.height)).toBeLessThanOrEqual(240);
  });

  it("inflate to the pixels of the scene", () => {
    const rowBytes = Math.ceil((decoded.width * decoded.bitDepth) / 8);
    expect(decoded.pixels.length).toBe((rowBytes + 1) * decoded.height);
    const palette = decoded.chunks.PLTE!;
    const colorAt = (x: number, y: number): string => {
      const row = y * (rowBytes + 1);
      expect(decoded.pixels[row]).toBe(0); // filter: none
      const perByte = 8 / decoded.bitDepth;
      const byte = decoded.pixels[row + 1 + Math.floor(x / perByte)] as number;
      const shift = 8 - decoded.bitDepth * ((x % perByte) + 1);
      const index = (byte >> shift) & (2 ** decoded.bitDepth - 1);
      return [0, 1, 2].map((c) => (palette[index * 3 + c] as number).toString(16).padStart(2, "0")).join("");
    };
    for (const [x, y] of [[3, 3], [120, 130], [120, 100], [175, 28], [239, 159]] as const) {
      const layer = layerAt(scene, (x + 0.5) / 240, (y + 0.5) / 160, SUN_FRACTION_THUMB, 240 / 160);
      expect(colorAt(x, y)).toBe(hexToRgb(scene[layer]).map((c) => c.toString(16).padStart(2, "0")).join(""));
    }
  });

  it("are small: a few kilobytes, not the 115 KB of raw RGB", () => {
    expect(png.length).toBeLessThan(12_000);
  });

  it("encodeIndexedPng rejects a palette the bit depth can't address", () => {
    const palette = Array.from({ length: 5 }, () => [0, 0, 0] as const);
    expect(() => encodeIndexedPng(1, 1, palette, new Uint8Array(1), 2)).toThrow(/palette/);
    expect(() => encodeIndexedPng(2, 2, [[0, 0, 0]], new Uint8Array(1), 2)).toThrow(/size/);
  });

  it("handles images wider than one deflate block", () => {
    const big = encodeIndexedPng(300, 300, [[1, 2, 3], [4, 5, 6]], new Uint8Array(300 * 300).fill(1), 8);
    const out = readPng(big);
    expect(out.pixels.length).toBe((300 + 1) * 300); // spans two stored blocks
    expect(out.pixels[1]).toBe(1);
  });
});
