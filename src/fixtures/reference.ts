/**
 * What the mock API stands in for when the page asks the core about a Reference Grade node's reference (`docs/34` §9.5): a stand-in picture, the errors
 * and notes a file name asks for, a **fixed fit** and the shares of the elements it "found". Nothing here reads a file or runs a fit; the real work is the core's
 * (`Core::reference_choose_file`, `Core::reference_fit`). The fit is the node of `docs/34` §3.1 with six knots on each hue curve, as a fit writes them, and the
 * element grades of `design/reference-grade/` (the sky 0.45 EV lower and 14° toward teal, the foliage at 75 %, the water not found).
 */
import type { ElementShare, NodePatch } from "../api/types";

/** Sentences of `docs/34` §9.3 the mock raises, the core's words. */
export const MOCK_REFERENCE_TEXT = {
  unreadable: "This file can't be read. It may be damaged. Choose another picture.",
  unsupported: "Yellowmoon reads JPEG, PNG and TIFF pictures. Save this one as one of those, then choose it again.",
  tooSmall: "This picture is too small to match. Use one that is at least 256 pixels on its short side.",
  empty: "There's no picture inside the black bars. Choose another picture.",
  samePhoto: "That's the photo you're grading. Choose another.",
  notReady: "This photo isn't ready yet.",
  noReference: "Choose a reference first.",
  blackBars: "Black bars were removed from the reference.",
  profileUnreadable: "The picture's colour profile couldn't be read, so it is taken as sRGB.",
  noColour: "This picture has no colour. The photo will be matched in black and white.",
  noSkin: "No skin was found, so Skin protection did nothing.",
} as const;

/** What a file name asks of the mock, as the file's content would of the core: an error, or notes. (`~/Pictures/harbour-dusk.jpg` asks for nothing.) */
export function referenceFileOutcome(path: string): { error: string } | { notes: string[]; grey: boolean } {
  const name = (path.split("/").pop() ?? path).toLowerCase();
  if (/\.(heic|heif|webp|gif|bmp|pdf|raw|arw|dng)$/.test(name)) return { error: MOCK_REFERENCE_TEXT.unsupported };
  if (name.includes("damaged")) return { error: MOCK_REFERENCE_TEXT.unreadable };
  if (name.includes("tiny")) return { error: MOCK_REFERENCE_TEXT.tooSmall };
  if (name.includes("empty")) return { error: MOCK_REFERENCE_TEXT.empty };
  const notes: string[] = [];
  if (name.includes("bars")) notes.push(MOCK_REFERENCE_TEXT.blackBars);
  if (name.includes("profile")) notes.push(MOCK_REFERENCE_TEXT.profileUnreadable);
  const grey = name.includes("grey") || name.includes("gray");
  if (grey) notes.push(MOCK_REFERENCE_TEXT.noColour);
  return { notes, grey };
}

/** The name the card shows under a reference's picture: the file's name. */
export const referenceLabelOf = (path: string): string => path.split("/").pop() ?? path;

/** A stand-in for a reference's picture: a harbour at dusk, drawn as an SVG (teal water, an amber sky), 16:9, as a data address. */
export function referencePicture(seed: string): string {
  const hue = [...seed].reduce((n, c) => (n * 31 + c.charCodeAt(0)) % 360, 0);
  const svg =
    `<svg xmlns="http://www.w3.org/2000/svg" width="256" height="144" viewBox="0 0 256 144">` +
    `<defs><linearGradient id="s" x1="0" y1="0" x2="0" y2="1"><stop offset="0" stop-color="#1f4a63"/><stop offset=".62" stop-color="#e59a4a"/></linearGradient></defs>` +
    `<rect width="256" height="144" fill="url(#s)"/><rect y="88" width="256" height="56" fill="#0f3a4a"/>` +
    `<rect x="${20 + (hue % 40)}" y="62" width="150" height="26" fill="#10222c"/>` +
    `<g fill="#f2c26b"><rect x="40" y="70" width="6" height="6"/><rect x="70" y="70" width="6" height="6"/><rect x="100" y="70" width="6" height="6"/><rect x="130" y="70" width="6" height="6"/></g></svg>`;
  return `data:image/svg+xml;charset=utf-8,${encodeURIComponent(svg)}`;
}

/** The six elements' shares of the photo the fixed fit "found"; Water is under 2 %, so its card says it was not found. */
export const MOCK_SHARES: readonly ElementShare[] = [
  { element: "sky", share: 0.26 },
  { element: "vegetation", share: 0.29 },
  { element: "water", share: 0.006 },
  { element: "ground", share: 0.19 },
  { element: "built", share: 0.12 },
  { element: "skin", share: 0.033 },
];

/** The knots of the six hues, 0° to 300°. */
const HUES = [0, 60, 120, 180, 240, 300] as const;

/**
 * The fixed fit (`docs/34` §3.1's example with a fit's six knots on each hue curve): the fitted fields of the node, in one patch. Water is left at the
 * identity (it was not found). The same numbers every time, whatever the photo or the reference.
 */
export function fixedFitPatch(): NodePatch {
  const turns = [-4, 6, 12, 4, -3, 2];
  const chroma = [1.05, 1.08, 0.85, 1.12, 1.06, 0.92];
  return {
    grade: {
      saturation: { kind: "manual", value: -12 },
      curve: {
        luma: [[0, 0], [0.2136, 0.205], [0.2784, 0.2784], [0.408, 0.431], [1, 1]],
        red: [[0, 0], [0.2784, 0.284], [1, 1]],
        green: [[0, 0], [0.2784, 0.2784], [1, 1]],
        blue: [[0, 0], [0.2784, 0.27], [1, 1]],
      },
    },
    hue_curves: {
      hue_vs_hue: HUES.map((hue_deg, i) => ({ hue_deg, value: turns[i] as number })),
      hue_vs_sat: HUES.map((hue_deg, i) => ({ hue_deg, value: chroma[i] as number })),
    },
    elements: {
      sky: { exposure_ev: -0.45, hue_deg: -14, chroma: 1.25 },
      vegetation: { exposure_ev: -0.25, hue_deg: 20, chroma: 0.75 },
      water: { exposure_ev: 0, hue_deg: 0, chroma: 1 },
      ground: { exposure_ev: 0.1, hue_deg: -6, chroma: 1.1 },
      built: { exposure_ev: 0, hue_deg: 4, chroma: 1.05 },
      skin: { exposure_ev: 0, hue_deg: 3, chroma: 0.95 },
      subject: { exposure_ev: 0.2 },
    },
  };
}

/** How long each step of the mock's fit takes at speed 1, in milliseconds: the scene, then three searches. */
export const MOCK_FIT_STEP_MS = 400;
