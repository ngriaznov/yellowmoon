/**
 * A tiny PNG encoder for the mock API's thumbnails. The core sends thumbnails across the bridge
 * as PNG bytes (`thumbnail_ready { id, png_bytes }`), so the mock does the same, and the code
 * that turns bytes into an image is exercised in the browser exactly as it is in the app.
 *
 * Indexed colour, no compression (stored deflate blocks): the fixtures have four flat colours,
 * so a 240 × 160 thumbnail packs into about 10 KB at two bits per pixel.
 */
import { layerAt, SUN_FRACTION_THUMB, type Scene, type SceneLayer } from "./scenes";

const SIGNATURE = [0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a];

let crcTable: Uint32Array | null = null;

function crc32(bytes: Uint8Array): number {
  if (!crcTable) {
    crcTable = new Uint32Array(256);
    for (let n = 0; n < 256; n++) {
      let c = n;
      for (let k = 0; k < 8; k++) c = c & 1 ? 0xedb88320 ^ (c >>> 1) : c >>> 1;
      crcTable[n] = c >>> 0;
    }
  }
  let crc = 0xffffffff;
  for (const byte of bytes) crc = (crcTable[(crc ^ byte) & 0xff] as number) ^ (crc >>> 8);
  return (crc ^ 0xffffffff) >>> 0;
}

function adler32(bytes: Uint8Array): number {
  let a = 1;
  let b = 0;
  for (const byte of bytes) {
    a = (a + byte) % 65521;
    b = (b + a) % 65521;
  }
  return ((b << 16) | a) >>> 0;
}

function u32(value: number): number[] {
  return [(value >>> 24) & 0xff, (value >>> 16) & 0xff, (value >>> 8) & 0xff, value & 0xff];
}

function chunk(type: string, data: Uint8Array): Uint8Array {
  const typeBytes = Array.from(type, (c) => c.charCodeAt(0));
  const body = new Uint8Array(typeBytes.length + data.length);
  body.set(typeBytes, 0);
  body.set(data, typeBytes.length);
  const out = new Uint8Array(4 + body.length + 4);
  out.set(u32(data.length), 0);
  out.set(body, 4);
  out.set(u32(crc32(body)), 4 + body.length);
  return out;
}

/** Wraps raw bytes in a zlib stream made of stored (uncompressed) deflate blocks. */
function zlibStored(raw: Uint8Array): Uint8Array {
  const blocks = Math.max(1, Math.ceil(raw.length / 65535));
  const out = new Uint8Array(2 + raw.length + blocks * 5 + 4);
  let pos = 0;
  out[pos++] = 0x78;
  out[pos++] = 0x01;
  for (let i = 0; i < blocks; i++) {
    const start = i * 65535;
    const len = Math.min(65535, raw.length - start);
    out[pos++] = i === blocks - 1 ? 1 : 0;
    out[pos++] = len & 0xff;
    out[pos++] = len >>> 8;
    out[pos++] = ~len & 0xff;
    out[pos++] = (~len >>> 8) & 0xff;
    out.set(raw.subarray(start, start + len), pos);
    pos += len;
  }
  out.set(u32(adler32(raw)), pos);
  return out;
}

/** Parses `#RRGGBB` into `[r, g, b]`. */
export function hexToRgb(hex: string): [number, number, number] {
  const n = Number.parseInt(hex.slice(1), 16);
  return [(n >> 16) & 255, (n >> 8) & 255, n & 255];
}

/**
 * Encodes an indexed-colour PNG. `indices` holds one palette index per pixel, row by row;
 * `bitDepth` is 1, 2, 4 or 8 and must be able to address the whole palette.
 */
export function encodeIndexedPng(
  width: number,
  height: number,
  palette: ReadonlyArray<readonly [number, number, number]>,
  indices: Uint8Array,
  bitDepth: 1 | 2 | 4 | 8,
): Uint8Array {
  if (indices.length !== width * height) throw new Error("encodeIndexedPng: indices do not match the size");
  if (palette.length > 2 ** bitDepth) throw new Error("encodeIndexedPng: palette too large for the bit depth");

  const rowBytes = Math.ceil((width * bitDepth) / 8);
  const raw = new Uint8Array((rowBytes + 1) * height);
  const perByte = 8 / bitDepth;
  for (let y = 0; y < height; y++) {
    const rowStart = y * (rowBytes + 1);
    raw[rowStart] = 0; // filter type: none
    for (let x = 0; x < width; x++) {
      const shift = 8 - bitDepth * ((x % perByte) + 1);
      const at = rowStart + 1 + Math.floor(x / perByte);
      raw[at] = (raw[at] as number) | (((indices[y * width + x] as number) & (2 ** bitDepth - 1)) << shift);
    }
  }

  const ihdr = new Uint8Array([...u32(width), ...u32(height), bitDepth, 3, 0, 0, 0]);
  const plte = new Uint8Array(palette.flatMap((rgb) => [...rgb]));
  const parts = [
    new Uint8Array(SIGNATURE),
    chunk("IHDR", ihdr),
    chunk("PLTE", plte),
    chunk("IDAT", zlibStored(raw)),
    chunk("IEND", new Uint8Array(0)),
  ];
  const out = new Uint8Array(parts.reduce((sum, p) => sum + p.length, 0));
  let pos = 0;
  for (const part of parts) {
    out.set(part, pos);
    pos += part.length;
  }
  return out;
}

const LAYER_INDEX: Record<SceneLayer, number> = { sky: 0, sun: 1, far: 2, near: 3 };

/** A scene as a PNG thumbnail (240 × 160 by default; the bridge allows up to 240 px on the long edge). */
export function scenePng(scene: Scene, width = 240, height = 160): Uint8Array {
  const palette = [scene.sky, scene.sun, scene.far, scene.near].map(hexToRgb);
  const indices = new Uint8Array(width * height);
  const aspect = width / height;
  for (let y = 0; y < height; y++) {
    for (let x = 0; x < width; x++) {
      const layer = layerAt(scene, (x + 0.5) / width, (y + 0.5) / height, SUN_FRACTION_THUMB, aspect);
      indices[y * width + x] = LAYER_INDEX[layer];
    }
  }
  return encodeIndexedPng(width, height, palette, indices, 2);
}
