import { useEffect, useRef, useState } from "react";
import type { KeyboardEvent, PointerEvent as ReactPointerEvent } from "react";
import { fitInCell } from "./crop/cropMath";

export interface PickWhiteLayerProps {
  /** Width ÷ height of the picture as it is shown (the crop, when there is one). */
  displayAspect: number;
  /**
   * The picture is letterboxed on paper (zoom Fit). At a fixed zoom it fills the cell and every click counts. The layer needs this only
   * to ignore clicks on the paper; where on the photo a click lands is the core's to say.
   */
  letterboxed?: boolean;
  /** The point clicked, in CSS pixels from the corner of the photo cell: what `viewport_input` takes. */
  onPick: (point: { x: number; y: number }) => void;
}

const KEY_STEP = 0.01;

/**
 * Pick white (`docs/11-ui.md` §Pick white): a crosshair over the photo. A click samples what is
 * under it and ends the mode. From the keyboard the layer takes focus, the arrow keys move the
 * crosshair (Shift: ten times as far) and Return or Space picks. Esc, handled by the screen,
 * cancels. Only the picture counts; the paper around a letterboxed photo does not.
 *
 * The layer reports the point in the cell, not in the photo: the straightening, the quarter turn and the lens crop that lie between the
 * two are the core's (`viewport_input` of kind `pick_white`), and a conversion here would be wrong for any photo that has one.
 */
export function PickWhiteLayer({ displayAspect, letterboxed = true, onPick }: PickWhiteLayerProps) {
  const layer = useRef<HTMLDivElement>(null);
  // The crosshair position as a share of the cell; null until the pointer or a key puts it somewhere.
  const [at, setAt] = useState<{ x: number; y: number } | null>(null);
  const picture = fitInCell(1.5, displayAspect);

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
  const pick = (share: { x: number; y: number }, px: { x: number; y: number }): void => {
    if (letterboxed) {
      const u = (share.x - picture.x) / picture.width;
      const v = (share.y - picture.y) / picture.height;
      if (u < 0 || u > 1 || v < 0 || v > 1) return;
    }
    onPick(px);
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
    else if (event.key === "Enter" || event.key === " ") {
      event.preventDefault();
      const box = measure();
      const share = at ?? { x: 0.5, y: 0.5 };
      pick(share, { x: share.x * (box?.width ?? 0), y: share.y * (box?.height ?? 0) });
    }
  };

  return (
    <div
      ref={layer}
      className="ym-pick-white"
      role="application"
      tabIndex={0}
      aria-label="Pick white. Click something neutral, or move the crosshair with the arrow keys and press Return."
      onPointerMove={(event) => {
        const point = cellPoint(event);
        if (point) setAt(point);
      }}
      onPointerLeave={() => setAt(null)}
      onClick={(event) => {
        const box = measure();
        const share = cellPoint(event as unknown as ReactPointerEvent<HTMLElement>);
        if (box && share) pick(share, { x: event.clientX - box.left, y: event.clientY - box.top });
      }}
      onKeyDown={onKeyDown}
    >
      {at ? (
        <svg className="ym-pick-white__mark" style={{ left: `${at.x * 100}%`, top: `${at.y * 100}%` }} viewBox="0 0 28 28" aria-hidden="true">
          <g fill="none" strokeLinecap="square">
            <path d="M14 2v8M14 18v8M2 14h8M18 14h8" stroke="#fff" strokeWidth="5" />
            <path d="M14 2v8M14 18v8M2 14h8M18 14h8" stroke="#000" strokeWidth="2" />
          </g>
        </svg>
      ) : null}
    </div>
  );
}
