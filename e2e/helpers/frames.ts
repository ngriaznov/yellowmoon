import type { Page } from "@playwright/test";

// Selection frames: the ring that tells a selected thing from the others (the look tile in use, the photo on screen in the filmstrip,
// the element that has keyboard focus). A ring is drawn with an outline or a box-shadow, outside the box it belongs to, so a box
// that clips its content (`overflow: hidden`, `auto`, `scroll` or `clip`) cuts it off whenever the item stands closer to the edge
// than the ring is wide. `frameClipping` measures that in the page: where each ring is, and which ancestor clips which of its sides.

/** A side of a ring. */
export type Side = "left" | "right" | "top" | "bottom";

/** One ancestor that clips a ring, and the sides it cuts. */
export interface Clip {
  /** The ancestor, as `tag.class` plus its `data-slot`. */
  by: string;
  /** What the ancestor does with its content on the cut side: `hidden`, `auto`, `scroll` or `clip`, or `viewport` for the window. */
  overflow: string;
  /** For each cut side, how many CSS pixels of the ring lie beyond the clip. */
  cut: Partial<Record<Side, number>>;
}

/** One ring, as the page draws it now. */
export interface FrameReport {
  /** The item, as `tag "accessible name"`. */
  item: string;
  /** The ring's outer edge in viewport coordinates: the box with its outline and the spread of its shadows. */
  ring: { left: number; top: number; right: number; bottom: number };
  /** Every ancestor that clips part of the ring. Empty when the whole ring is drawn. */
  clips: Clip[];
}

/** What to measure: the items by CSS selector, and for an item whose ring is drawn on a part of it, that part's selector. */
export interface FrameTarget {
  /** The items that carry the selection, for example `.ym-look-tile[aria-pressed="true"]`. */
  selector: string;
  /** The element inside the item that draws the ring (the look tile's picture). The item itself when omitted. */
  ring?: string;
}

/** The two selection frames the app draws, plus the focus ring of whatever has keyboard focus. */
export const SELECTION_FRAMES: readonly FrameTarget[] = [
  { selector: '.ym-look-tile[aria-pressed="true"]', ring: ".ym-look-tile__picture" },
  { selector: '.ym-filmstrip__thumb[aria-current="true"]' },
  { selector: ".ym-look-tile:focus-visible", ring: ".ym-look-tile__picture" },
  { selector: ".ym-filmstrip__thumb:focus-visible" },
];

/**
 * Measures every selection frame on the page as it is now.
 *
 * A ring's outer edge is the item's border box grown by the outline (its offset and width) and by the spread of every sharp shadow that is not
 * inset. A clip is every ancestor whose overflow is not `visible` on that axis; it cuts the ring where the ring leaves the ancestor's
 * padding box. The window clips too. A scroll container that has simply scrolled an item half out of sight cuts it by its own rule,
 * so callers that mean "the item is on show" scroll it fully into view first.
 */
export function frameClipping(page: Page, targets: readonly FrameTarget[] = SELECTION_FRAMES): Promise<FrameReport[]> {
  return page.evaluate((list) => {
    const px = (value: string): number => Number.parseFloat(value) || 0;
    /** The spread, offset included, that a shadow list adds on each side. Colour functions are removed first so their commas do not split. */
    const shadowReach = (value: string): { left: number; top: number; right: number; bottom: number } => {
      const reach = { left: 0, top: 0, right: 0, bottom: 0 };
      if (value === "none") return reach;
      const parts = value.replace(/(?:rgba?|hsla?|oklch|oklab|lab|lch|color)\([^)]*\)/g, "").split(/,(?![^(]*\))/);
      for (const part of parts) {
        if (/\binset\b/.test(part)) continue;
        // The lengths are x, y, blur and spread. A shadow with a blur is a soft glow, not a frame; a spread wider than a ring is a veil, such
        // as the crop's dimming of the rest of the photo (4000 px). Neither is part of the ring.
        const [x = 0, y = 0, blur = 0, spread = 0] = (part.match(/-?\d*\.?\d+px/g) ?? []).map(px);
        if (blur > 0 || spread > 12) continue;
        reach.left = Math.max(reach.left, spread - x);
        reach.right = Math.max(reach.right, spread + x);
        reach.top = Math.max(reach.top, spread - y);
        reach.bottom = Math.max(reach.bottom, spread + y);
      }
      return reach;
    };
    const name = (el: Element): string => {
      const text = (el.getAttribute("aria-label") || el.textContent || "").trim().replace(/\s+/g, " ").slice(0, 40);
      return `${el.tagName.toLowerCase()} "${text}"`;
    };
    const describe = (el: Element): string => {
      const classes = [...el.classList].slice(0, 2).join(".");
      const slot = el.getAttribute("data-slot");
      return `${el.tagName.toLowerCase()}${classes ? "." + classes : ""}${slot ? `[data-slot=${slot}]` : ""}`;
    };

    const reports: FrameReport[] = [];
    for (const { selector, ring: ringSelector } of list) {
      for (const item of document.querySelectorAll<HTMLElement>(selector)) {
        const drawn = ringSelector ? (item.querySelector<HTMLElement>(ringSelector) ?? item) : item;
        const style = getComputedStyle(drawn);
        const box = drawn.getBoundingClientRect();
        const outline = style.outlineStyle !== "none" ? px(style.outlineWidth) + px(style.outlineOffset) : 0;
        const shadow = shadowReach(style.boxShadow);
        const ring = {
          left: box.left - Math.max(outline, shadow.left),
          top: box.top - Math.max(outline, shadow.top),
          right: box.right + Math.max(outline, shadow.right),
          bottom: box.bottom + Math.max(outline, shadow.bottom),
        };

        const clips: Clip[] = [];
        for (let node = drawn.parentElement; node; node = node.parentElement) {
          const s = getComputedStyle(node);
          const clipX = s.overflowX !== "visible";
          const clipY = s.overflowY !== "visible";
          if (!clipX && !clipY) continue;
          const r = node.getBoundingClientRect();
          // The padding box: inside the border, and inside a scrollbar if there is one (`clientWidth` leaves it out).
          const left = r.left + node.clientLeft;
          const top = r.top + node.clientTop;
          const right = left + node.clientWidth;
          const bottom = top + node.clientHeight;
          const cut: Clip["cut"] = {};
          if (clipX && ring.left < left - 0.5) cut.left = left - ring.left;
          if (clipX && ring.right > right + 0.5) cut.right = ring.right - right;
          if (clipY && ring.top < top - 0.5) cut.top = top - ring.top;
          if (clipY && ring.bottom > bottom + 0.5) cut.bottom = ring.bottom - bottom;
          if (Object.keys(cut).length) clips.push({ by: describe(node), overflow: clipY && !clipX ? s.overflowY : s.overflowX, cut });
        }
        const view: Clip["cut"] = {};
        if (ring.left < -0.5) view.left = -ring.left;
        if (ring.top < -0.5) view.top = -ring.top;
        if (ring.right > innerWidth + 0.5) view.right = ring.right - innerWidth;
        if (ring.bottom > innerHeight + 0.5) view.bottom = ring.bottom - innerHeight;
        if (Object.keys(view).length) clips.push({ by: "window", overflow: "viewport", cut: view });

        reports.push({ item: name(item), ring, clips });
      }
    }
    return reports;
  }, targets);
}

/** A one-line account of what is cut, for an assertion message: `Neutral look: .ym-look-group > collapsible-content cuts left 4, top 4`. */
export function describeClips(reports: readonly FrameReport[]): string[] {
  return reports.flatMap((report) =>
    report.clips.map((clip) => `${report.item} is cut by ${clip.by} (${clip.overflow}): ${Object.entries(clip.cut).map(([side, by]) => `${side} ${by?.toFixed(1)} px`).join(", ")}`),
  );
}
