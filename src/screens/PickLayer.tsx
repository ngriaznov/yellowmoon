import { useEffect, useRef, useState } from "react";
import type { KeyboardEvent, PointerEvent as ReactPointerEvent } from "react";
import { fitInCell } from "./crop/cropMath";

/** What a click on the photo picks: a white for the white balance, a colour for the colour warp, the colour of a Colour range mask, or the depth for a Depth mask. */
export type PickMode = "white" | "color" | "range" | "depth";

/** What the layer does: a pick, or "Place points" of the Object mask (`docs/28` A-11), where Return adds a point instead of picking a spot. */
export type LayerMode = PickMode | "points";

/** The layer's accessible name in each mode: what to click, and how to do it from the keyboard. */
const LAYER_LABELS: Readonly<Record<LayerMode, string>> = {
  white: "Pick white. Click something neutral, or move the crosshair with the arrow keys and press Return.",
  color: "Pick colour. Click a colour in the photo, or move the crosshair with the arrow keys and press Return.",
  range: "Pick colour for the Colour range mask. Click a colour in the photo, or move the crosshair with the arrow keys and press Return.",
  depth: "Pick depth. Click the part of the photo to select, or move the crosshair with the arrow keys and press Return.",
  points: "Place points. Move the crosshair with the arrow keys. Return adds a point, Shift Return takes a part away, Backspace removes the last point, Esc ends.",
};

export interface PickLayerProps {
  /** Pick white (the default), Pick colour or the Colour range's pick. Only the accessible name differs: what a click does is the screen's (`onPick`). */
  mode?: LayerMode;
  /** Width ÷ height of the picture as it is shown (the crop, when there is one). */
  displayAspect: number;
  /**
   * The picture is letterboxed on paper (zoom Fit). At a fixed zoom it fills the cell and every click counts. The layer needs this only
   * to ignore clicks on the paper; where on the photo a click lands is the core's to say.
   */
  letterboxed?: boolean;
  /** The point clicked, in CSS pixels from the corner of the photo cell: what `viewport_input` and `sample_chroma` take. */
  onPick: (point: { x: number; y: number }, negative?: boolean) => void;
  /** "Place points" only: Backspace, which takes the last point out. */
  onBack?: () => void;
}

const KEY_STEP = 0.01;

/**
 * Pick white and Pick colour (`docs/11-ui.md` §Pick white, §Colour warp): a crosshair over the photo, drawn white with a dark outline. A click reports the point under
 * it; the screen samples it (Pick white then ends, Pick colour stays on for the next click). From the keyboard the layer takes focus,
 * the arrow keys move the crosshair (Shift: ten times as far) and Return or Space picks. Esc, handled by the screen, ends the mode.
 * Only the picture counts; the paper around a letterboxed photo does not.
 *
 * The layer reports the point in the cell, not in the photo: the straightening, the quarter turn and the lens crop that lie between the
 * two are the core's (`viewport_input` of kind `pick_white`, `sample_chroma`), and a conversion here would be wrong for any photo that
 * has one.
 */
export function PickLayer({ mode = "white", displayAspect, letterboxed = true, onPick, onBack }: PickLayerProps) {
  const points = mode === "points";
  const layer = useRef<HTMLDivElement>(null);
  // The crosshair position as a share of the cell; null until the pointer or a key puts it somewhere.
  const [at, setAt] = useState<{ x: number; y: number } | null>(points ? { x: 0.5, y: 0.5 } : null);

  useEffect(() => {
    layer.current?.focus({ preventScroll: true });
  }, []);

  /** The layer's size in CSS pixels, or null before it has one. */
  const measure = (): DOMRect | null => {
    const box = layer.current?.getBoundingClientRect();
    return box && box.width > 0 && box.height > 0 ? box : null;
  };

  /** A point of the pointer as a share of the cell. */
  const cellPoint = (event: ReactPointerEvent<HTMLElement>): { x: number; y: number } | null => {
    const box = measure();
    return box ? { x: (event.clientX - box.left) / box.width, y: (event.clientY - box.top) / box.height } : null;
  };

  /** Reports a point of the cell, unless it is on the paper around the picture. `share` is the point as a share of the cell. */
  const pick = (share: { x: number; y: number }, px: { x: number; y: number }, negative = false): void => {
    if (letterboxed) {
      // The cell is 3:2, or the whole window in full screen (F): where the picture sits depends on the cell's own shape.
      const box = measure();
      const picture = fitInCell(box ? box.width / box.height : 1.5, displayAspect);
      const u = (share.x - picture.x) / picture.width;
      const v = (share.y - picture.y) / picture.height;
      if (u < 0 || u > 1 || v < 0 || v > 1) return;
    }
    onPick(px, negative);
  };

  const onKeyDown = (event: KeyboardEvent<HTMLDivElement>): void => {
    const unit = KEY_STEP * (event.shiftKey ? 10 : 1);
    const move = (dx: number, dy: number): void => {
      event.preventDefault();
      const from = at ?? { x: 0.5, y: 0.5 };
      setAt({ x: Math.min(1, Math.max(0, from.x + dx)), y: Math.min(1, Math.max(0, from.y + dy)) });
    };
    if (event.key === "ArrowLeft") move(-unit, 0);
    else if (event.key === "ArrowRight") move(unit, 0);
    else if (event.key === "ArrowUp") move(0, -unit);
    else if (event.key === "ArrowDown") move(0, unit);
    else if (event.key === "Enter" || (event.key === " " && !points)) {
      event.preventDefault();
      // A key held down adds one point, not a row of them.
      if (points && event.repeat) return;
      const box = measure();
      const share = at ?? { x: 0.5, y: 0.5 };
      pick(share, { x: share.x * (box?.width ?? 0), y: share.y * (box?.height ?? 0) }, points && event.shiftKey);
    } else if (points && event.key === "Backspace") {
      event.preventDefault();
      if (!event.repeat) onBack?.();
    }
  };

  return (
    <div
      ref={layer}
      className="ym-pick-layer"
      data-mode={mode}
      role="application"
      tabIndex={0}
      aria-label={LAYER_LABELS[mode]}
      onPointerMove={(event) => {
        const point = cellPoint(event);
        if (point) setAt(point);
      }}
      onPointerLeave={points ? undefined : () => setAt(null)}
      onClick={(event) => {
        const box = measure();
        const share = cellPoint(event as unknown as ReactPointerEvent<HTMLElement>);
        if (box && share) pick(share, { x: event.clientX - box.left, y: event.clientY - box.top }, points && event.altKey);
      }}
      onKeyDown={onKeyDown}
    >
      {at ? (
        <svg className="ym-pick-layer__mark" style={{ left: `${at.x * 100}%`, top: `${at.y * 100}%` }} viewBox="0 0 48 48" aria-hidden="true">
          {/* White, with a dark outline under it, so that it reads on light and dark parts of a photo. The circle is open: the spot under
              it stays in view, and the middle of the circle is the point that is picked. */}
          <g fill="none" strokeLinecap="round">
            <g stroke="rgb(0 0 0 / 0.6)" strokeWidth="4" data-part="outline">
              <circle cx="24" cy="24" r="10" />
              <path d="M24 2v12M24 34v12M2 24h12M34 24h12" />
            </g>
            <g stroke="#fff" strokeWidth="1.6" data-part="line">
              <circle cx="24" cy="24" r="10" />
              <path d="M24 2v12M24 34v12M2 24h12M34 24h12" />
            </g>
          </g>
        </svg>
      ) : null}
    </div>
  );
}
