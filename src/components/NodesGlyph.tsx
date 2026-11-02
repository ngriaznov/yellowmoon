import type { SVGProps } from "react";

/**
 * The nodes glyph (`docs/23` §6): three nodes and the links between them, as in the mockups (`design/nodes/04-simple-photo-has-nodes.jpg`).
 * It marks the notice of a photo that has nodes and the filmstrip thumbnail of one. Drawn in `currentColor`, 24 × 24, hidden from
 * assistive technology: the words beside it say what it says.
 */
export function NodesGlyph(props: SVGProps<SVGSVGElement>) {
  return (
    <svg
      xmlns="http://www.w3.org/2000/svg"
      width="16"
      height="16"
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth="2"
      strokeLinecap="round"
      strokeLinejoin="round"
      aria-hidden="true"
      focusable="false"
      {...props}
    >
      <circle cx="5.5" cy="12" r="2.2" />
      <circle cx="18.5" cy="6" r="2.2" />
      <circle cx="18.5" cy="18" r="2.2" />
      <path d="M7.7 12h2.3c2 0 2.2-6 5-6h1.3" />
      <path d="M10 12c2.8 0 3 6 5.8 6h.5" />
    </svg>
  );
}
