import { useEffect, useRef, useState } from "react";
import type { KeyboardEvent, PointerEvent as ReactPointerEvent } from "react";
import { fitInCell } from "./crop/cropMath";
import type { Rect } from "./crop/cropMath";

export interface PickWhiteLayerProps {
  /** Width ÷ height of the picture as it is shown (the crop, when there is one). */
  displayAspect: number;
  /** The crop being shown, so a click maps back to the whole photo. */
  crop: Rect | null;
  /** The point clicked, in the whole photo's coordinates, each 0–1. */
  onPick: (point: [number, number]) => void;
}

const KEY_STEP = 0.01;

/**
 * Pick white (`docs/11-ui.md` §Pick white): a crosshair over the photo. A click samples what is
 * under it and ends the mode. From the keyboard the layer takes focus, the arrow keys move the
 * crosshair (Shift: ten times as far) and Return or Space picks. Esc, handled by the screen,
 * cancels. Only the picture counts; the paper around a letterboxed photo does not.
 */
export function PickWhiteLayer({ displayAspect, crop, onPick }: PickWhiteLayerProps) {
  const layer = useRef<HTMLDivElement>(null);
  // The crosshair position as a share of the cell; null until the pointer or a key puts it somewhere.
  const [at, setAt] = useState<{ x: number; y: number } | null>(null);
  const picture = fitInCell(1.5, displayAspect);

  useEffect(() => {
    layer.current?.focus({ preventScroll: true });
  }, []);

  const cellPoint = (event: ReactPointerEvent<HTMLElement>): { x: number; y: number } | null => {
    const box = layer.current?.getBoundingClientRect();
    if (!box || box.width <= 0 || box.height <= 0) return null;
    return { x: (event.clientX - box.left) / box.width, y: (event.clientY - box.top) / box.height };
  };

  const toPhoto = (point: { x: number; y: number }): [number, number] | null => {
    const u = (point.x - picture.x) / picture.width;
    const v = (point.y - picture.y) / picture.height;
    if (u < 0 || u > 1 || v < 0 || v > 1) return null;
    return crop ? [crop.x + u * crop.width, crop.y + v * crop.height] : [u, v];
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
      const photo = toPhoto(at ?? { x: 0.5, y: 0.5 });
      if (photo) onPick(photo);
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
        const point = cellPoint(event as unknown as ReactPointerEvent<HTMLElement>);
        const photo = point ? toPhoto(point) : null;
        if (photo) onPick(photo);
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
