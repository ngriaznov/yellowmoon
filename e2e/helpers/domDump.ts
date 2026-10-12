/**
 * Lists what a screen draws as plain data: every visible element that has its own text, a border or a background, with
 * its box and the style that matters for a visual comparison. It runs in the page. A dump of the app and a dump of the
 * mockup in the same state can then be compared box by box, text by text, which finds a one-pixel shift or a wrong
 * weight that a screenshot threshold lets through. `compareScreen` writes one when `YM_DOM_DUMP=1`.
 */
export interface DumpedBox {
  tag: string;
  text: string;
  x: number;
  y: number;
  w: number;
  h: number;
  /** Computed font size, weight and letter spacing, as the browser reports them. */
  fs: string;
  fw: string;
  ls: string;
  color: string;
  bg: string;
  /** Border widths, top, right, bottom, left. */
  bw: string;
  bc: string;
  op: string;
  ff: string;
  tt: string;
  td: string;
}

/** The function to pass to `page.evaluate`. */
export function dumpBoxes(): DumpedBox[] {
  const out: DumpedBox[] = [];
  for (const el of document.querySelectorAll("body *")) {
    const cs = getComputedStyle(el);
    if (cs.display === "none" || cs.visibility === "hidden") continue;
    const r = el.getBoundingClientRect();
    if (r.width === 0 && r.height === 0) continue;
    let text = "";
    for (const node of el.childNodes) if (node.nodeType === 3) text += node.textContent ?? "";
    // A text box shows its value as its text, so it compares with a mockup that draws the value in a <div>.
    if (el instanceof HTMLInputElement && el.type !== "range" && el.type !== "checkbox" && el.type !== "radio") {
      text = el.value || el.placeholder;
    }
    text = text.replace(/\s+/g, " ").trim();
    const widths = (["Top", "Right", "Bottom", "Left"] as const).map(
      (side) => parseFloat(cs[`border${side}Width` as "borderTopWidth"]) || 0,
    );
    const hasBorder = widths.some((width) => width > 0);
    const bg = cs.backgroundColor;
    const hasBg = bg !== "rgba(0, 0, 0, 0)" && bg !== "transparent";
    const tag = el.tagName.toLowerCase();
    if (!text && !hasBorder && !hasBg && tag !== "svg" && tag !== "img" && tag !== "input") continue;
    out.push({
      tag,
      text,
      x: +r.x.toFixed(2),
      y: +r.y.toFixed(2),
      w: +r.width.toFixed(2),
      h: +r.height.toFixed(2),
      fs: cs.fontSize,
      fw: cs.fontWeight,
      ls: cs.letterSpacing,
      color: cs.color,
      bg: hasBg ? bg : "",
      bw: widths.join(","),
      bc: hasBorder ? cs.borderTopColor : "",
      op: cs.opacity === "1" ? "" : cs.opacity,
      ff: text ? (cs.fontFamily.split(",")[0] ?? "") : "",
      tt: cs.textTransform === "none" ? "" : cs.textTransform,
      td: cs.textDecorationLine === "none" ? "" : cs.textDecorationLine,
    });
  }
  return out;
}
