import type { Page } from "@playwright/test";

// The page audit of the dark glass interface (C2): on a page, no element of ours stands where glasscn has a component. Every visible
// control must be a glasscn component (it, or an ancestor within a few levels, carries a glasscn `data-slot`), and every row, surface and
// badge that glasscn has a counterpart for (`Item`, `Card`, `Badge`, `Progress`, …) must be that component. What is ours and has no
// counterpart (the colour wheels, the curve plot, the histogram, look tiles, the photo cell, the layout bands) is listed, so that the list
// of what is left is visible and a new element of ours with a counterpart is caught.

/** Elements of the app that are not glasscn components but have one: [selector, glasscn component]. Each must be empty on every page. */
export const HAS_COUNTERPART: ReadonlyArray<readonly [string, string]> = [
  [".ym-queue__row", "Item"], [".ym-about-row", "Item"], [".ym-step", "Item"], [".ym-step-rail__step", "Item"], [".ym-pref", "Item"],
  [".ym-cube-list > li", "Item"], [".ym-wizard__option", "Item"], [".ym-wizard__watch", "Item"], [".ym-wizard__callout", "Item"],
  [".ym-wizard__waiting", "Item"], [".ym-wizard__problem", "Item"], [".ym-wizard__received > li", "Item"], [".ym-error-bar", "Item"],
  [".ym-wizard__package", "Item"], [".ym-info-row", "Item"], [".ym-switch-row", "Item"], [".ym-auto-row", "Item"],
  [".ym-drop-zone__notice", "Item"], [".ym-develop__notice", "Item"], [".ym-about-dialog__adobe", "Item"],
  [".ym-drop-zone", "Card"], [".ym-wizard__tile", "Card"],
  [".ym-photo-cell__before", "Badge"], [".ym-photo-cell__chip", "Badge"], [".ym-photo-cell__readout", "Badge"],
  ["[role=progressbar]", "Progress"], ["hr, [role=separator]", "Separator"], ["kbd", "Kbd"], ["progress", "Progress"], ["select", "Select"],
  ["input, textarea", "Input"], ["button, [role=button]", "Button"], ["[role=switch]", "Switch"], ["[role=slider]", "Slider"], ["[role=tab]", "Tabs"],
  ["[role=radio]:not([role=radio][aria-checked]) ", "Segmented control"],
];

/** What is ours and has no counterpart in glasscn: [selector, name]. Listed when present. */
export const NO_COUNTERPART: ReadonlyArray<readonly [string, string]> = [
  [".ym-wheel__disc", "colour wheel (shadows, midtones, highlights)"], [".ym-curve__plot", "curve plot"], [".ym-curve__point", "curve point"],
  [".ym-histogram__box", "histogram"], [".ym-filmstrip__thumb", "filmstrip thumbnail"], [".ym-look-tile", "look tile (a picture button)"],
  [".ym-crop__frame", "crop frame"], [".ym-crop__handle", "crop handle"], [".ym-pick-white", "pick-white layer"],
  [".ym-photo-cell", "photo cell (the native viewport shows through)"], ["[role=table]", "wizard table (glasscn has no table)"],
  [".ym-moon", "moon mark"], [".ym-card", "tool-panel section (a ruled band of one glass pane, by the layout rule; glasscn's Card is a floating pane)"],
  [".ym-title-block", "title block (a tinted band of the pane)"], [".ym-field", "label + control row (glasscn has no Field)"],
  [".ym-photo-cell__camera", "camera line over the photo (text)"],
  [".ym-wizard__tile-picture img, .ym-look-tile__picture img", "look picture"],
];

/** The `data-slot` values of glasscn's components (`src/components/glass/`) that a control can be or sit in. */
export const GLASS_SLOT =
  /^(tabs-list|segmented-control|button|toggle|tabs-trigger|segmented-control-item|slider-thumb|slider|switch|select-trigger|select-item|input|search-field-input|search-field-clear|dropdown-menu-[a-z-]+|dialog-close|dialog-content|item|card|badge|progress|separator|kbd|collapsible-trigger|toggle-group-item|tooltip-trigger|popover-trigger|textarea|tabs|spinner)$/;

/** What the audit found on one page. */
export interface PageAudit {
  /** Visible controls that are glasscn components, by `data-slot`. */
  glass: Record<string, number>;
  /** Visible controls of ours that glasscn has a component for. Must be empty. */
  ours: string[];
  /** Rows, surfaces and badges of ours that glasscn has a counterpart for. Must be empty. */
  pending: string[];
  /** What is ours with no counterpart, by name, with how many are visible. */
  none: Record<string, number>;
}

/** Audits the page as it is now. */
export function auditPage(page: Page): Promise<PageAudit> {
  return page.evaluate(
    ({ HAS, NONE, slotSrc }) => {
      const SLOT = new RegExp(slotSrc);
      const visible = (el: Element): boolean => {
        if (el.closest('[aria-hidden="true"]') && !el.matches(".ym-wizard__box,.ym-step-rail__mark")) return false;
        const r = el.getBoundingClientRect();
        return !(r.width === 0 && r.height === 0) && getComputedStyle(el).visibility !== "hidden";
      };
      const slotOf = (el: Element): string | null => {
        let node: Element | null = el;
        for (let i = 0; node && i < 6; node = node.parentElement, i++) {
          const slot = node.getAttribute("data-slot");
          if (slot && SLOT.test(slot)) return slot;
        }
        return null;
      };
      const label = (el: Element): string => {
        const name = (el.getAttribute("aria-label") || el.textContent || el.getAttribute("title") || "").trim().replace(/\s+/g, " ").slice(0, 36);
        const raw = (el as SVGElement).className;
        const cls = String(typeof raw === "string" ? raw : (raw as unknown as { baseVal?: string }).baseVal ?? "")
          .split(" ")
          .filter((c) => c.startsWith("ym-"))
          .slice(0, 2)
          .join(".");
        return `${el.tagName.toLowerCase()}${cls ? "." + cls : ""} "${name}"`;
      };
      const out: PageAudit = { glass: {}, ours: [], pending: [], none: {} };
      // Every control.
      const CONTROLS =
        'button, a[href], input, select, textarea, [role="slider"], [role="switch"], [role="tab"], [role="radio"], [role="checkbox"], [role="combobox"], [role="menuitem"], [role="menuitemradio"], [role="option"]';
      for (const el of document.querySelectorAll(CONTROLS)) {
        if (el.closest("[hidden]") || !visible(el)) continue;
        const slot = slotOf(el);
        if (slot) out.glass[slot] = (out.glass[slot] ?? 0) + 1;
        else if (!NONE.some(([sel]) => el.matches(sel) || el.closest(sel))) out.ours.push(label(el));
      }
      // Everything that has a counterpart must have it.
      for (const [sel, comp] of HAS) {
        if (/^(button|input|\[role=)/.test(sel)) continue; // covered by the control scan above
        for (const el of document.querySelectorAll(sel)) {
          if (!visible(el)) continue;
          if (!slotOf(el)) out.pending.push(`${label(el)} -> ${comp}`);
        }
      }
      for (const [sel, name] of NONE) {
        const n = [...document.querySelectorAll(sel)].filter(visible).length;
        if (n) out.none[name] = n;
      }
      return out;
    },
    { HAS: HAS_COUNTERPART, NONE: NO_COUNTERPART, slotSrc: GLASS_SLOT.source },
  );
}
