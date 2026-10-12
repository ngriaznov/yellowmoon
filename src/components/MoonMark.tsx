import type { CSSProperties } from "react";

interface Props {
  /** Diameter in CSS pixels, border included. Defaults to the `--moon-size` token (16 px). */
  size?: number;
  className?: string;
}

/**
 * The Yellowmoon mark: a yellow circle with a 2 px ink border, as drawn in the header of
 * `design/mockups/start.html`. The mockup builds it from CSS (a 16 px `span` with
 * `border-radius: 50%`), not from an SVG, so this does the same and takes its colours from
 * the tokens. Decorative: the product name always sits next to it.
 */
export function MoonMark({ size, className }: Props) {
  const style: CSSProperties | undefined = size === undefined ? undefined : { width: size, height: size };
  return <span aria-hidden="true" className={className ? `ym-moon ${className}` : "ym-moon"} style={style} />;
}
