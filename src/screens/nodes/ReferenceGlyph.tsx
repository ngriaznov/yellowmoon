/**
 * The Reference Grade node's glyph (`design/reference-grade/mockup-notes.md` item 20): a picture beside a picture, with a grade curve in the front one. A
 * rounded square with an S-shaped curve in it and the corner of a second frame behind it, in the 2 px stroke and round caps of the Blur and Colour Uniform
 * glyphs. It marks the node on its card's footer (13 px) and in the Add node menu. It is drawn in the current colour and is decorative: the text beside it,
 * or the tooltip around it, names it.
 */
export function ReferenceGlyph({ size = 13 }: { size?: number }) {
  return (
    <svg aria-hidden="true" width={size} height={size} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
      <path d="M8 5V4a1 1 0 0 1 1-1h11a1 1 0 0 1 1 1v11a1 1 0 0 1-1 1h-1" />
      <rect x="3" y="8" width="13" height="13" rx="2" />
      <path d="M6 18c3.5 0 3-6 7-6" />
    </svg>
  );
}
