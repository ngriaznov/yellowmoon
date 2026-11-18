/**
 * The Blur node's glyph (`docs/33` section 4): three concentric dotted rings. It marks the node on its card's footer and in the Add node menu.
 * It is drawn in the current colour and is decorative: the text beside it, or the tooltip around it, names it.
 */
export function BlurGlyph({ size = 13 }: { size?: number }) {
  return (
    <svg aria-hidden="true" width={size} height={size} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round">
      <circle cx="12" cy="12" r="3" strokeDasharray="0.1 3.1" />
      <circle cx="12" cy="12" r="6.5" strokeDasharray="0.1 4.1" />
      <circle cx="12" cy="12" r="10" strokeDasharray="0.1 5.2" />
    </svg>
  );
}
