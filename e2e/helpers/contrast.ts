import type { Page } from "@playwright/test";

// The measuring of `e2e/contrast.spec.ts` (`docs/28` A-05, M4-04, the method of its section 2.5): which texts are on the page and in what
// colour (read from the page), what is behind each of them (read from a screenshot, because the panes are glass and the page cannot say),
// and the WCAG 2.x contrast of the two. Nothing here asserts anything.

/** A rectangle in CSS pixels of the page. */
export interface Box {
  x: number;
  y: number;
  w: number;
  h: number;
}

/** One visible text element, as the page reports it. */
export interface TextItem extends Box {
  text: string;
  tag: string;
  /** All its classes, space separated. */
  cls: string;
  /** Its computed colour with the alpha recovered: `[r, g, b, a]`, rgb 0 to 255, a 0 to 1. */
  color: [number, number, number, number];
  /** The product of the `opacity` of the element and its ancestors. */
  opacity: number;
  size: number;
  weight: number;
  placeholder: boolean;
  /** The element or an ancestor is a disabled control. */
  disabled: boolean;
  /** The element is a keycap (`kbd`): the key hint of a button. */
  keycap: boolean;
}

/** The three label colours of the glass, as the page resolves them (rgb 0 to 255). */
export interface LabelColours {
  label: [number, number, number];
  secondary: [number, number, number];
  tertiary: [number, number, number];
}

/**
 * Every visible text element of the page: an element with a text node of its own (or an input with a value or a placeholder) that is
 * on screen and is the topmost thing at its centre. The colour is read through a 1 x 1 canvas drawn over black and over white, which
 * gives the colour whatever its syntax (`oklch()`, `color-mix()`, a variable) and its alpha: `alpha = 1 - (white - black) / 255`.
 */
export async function collectText(page: Page): Promise<{ items: TextItem[]; labels: LabelColours }> {
  return page.evaluate(() => {
    const canvas = document.createElement("canvas");
    canvas.width = 1;
    canvas.height = 1;
    const cx = canvas.getContext("2d", { willReadFrequently: true }) as CanvasRenderingContext2D;
    const rgba = (css: string): [number, number, number, number] => {
      const px = (ground: string): Uint8ClampedArray => {
        cx.fillStyle = ground;
        cx.fillRect(0, 0, 1, 1);
        cx.fillStyle = css;
        cx.fillRect(0, 0, 1, 1);
        return cx.getImageData(0, 0, 1, 1).data;
      };
      const b = px("#000");
      const w = px("#fff");
      const a = Math.min(1, Math.max(0, 1 - (w[0]! - b[0]! + w[1]! - b[1]! + w[2]! - b[2]!) / (3 * 255)));
      if (a < 0.02) return [0, 0, 0, 0];
      return [b[0]! / a, b[1]! / a, b[2]! / a, a];
    };
    const token = (name: string): [number, number, number] => {
      const probe = document.createElement("i");
      probe.style.color = `var(${name})`;
      document.body.append(probe);
      const colour = rgba(getComputedStyle(probe).color);
      probe.remove();
      return [colour[0], colour[1], colour[2]];
    };
    const labels = { label: token("--glass-label"), secondary: token("--glass-label-secondary"), tertiary: token("--glass-label-tertiary") };

    // `elementFromPoint` skips what the pointer cannot reach (`pointer-events: none`: the chips and the camera line over the photo, the
    // keycap of a disabled button), so for the covered check every element is made reachable while the page is read.
    const reach = document.createElement("style");
    reach.textContent = "* { pointer-events: auto !important; }";
    document.head.append(reach);
    const items: TextItem[] = [];
    const walker = document.createTreeWalker(document.body, NodeFilter.SHOW_ELEMENT);
    for (let node = walker.nextNode(); node; node = walker.nextNode()) {
      const el = node as HTMLElement;
      if (["SCRIPT", "STYLE", "svg", "path"].includes(el.tagName)) continue;
      let own = "";
      for (const child of el.childNodes) if (child.nodeType === 3) own += child.textContent ?? "";
      own = own.trim();
      const input = el as HTMLInputElement;
      const isField = (el.tagName === "INPUT" && !["range", "checkbox", "radio", "button"].includes(input.type)) || el.tagName === "TEXTAREA";
      if (!own && !(isField && (input.value || input.placeholder))) continue;
      const cs = getComputedStyle(el);
      if (cs.visibility === "hidden" || cs.display === "none") continue;
      const r = el.getBoundingClientRect();
      // 4 px: the visually hidden text (sr-only, 1 px) is not seen.
      if (r.width < 4 || r.height < 4 || r.bottom < 0 || r.top > innerHeight || r.right < 0 || r.left > innerWidth) continue;
      const top = document.elementFromPoint(Math.min(innerWidth - 1, Math.max(0, r.left + r.width / 2)), Math.min(innerHeight - 1, Math.max(0, r.top + r.height / 2)));
      if (!top || !(top === el || el.contains(top) || top.contains(el))) continue;
      let opacity = 1;
      for (let p: Element | null = el; p; p = p.parentElement) opacity *= parseFloat(getComputedStyle(p).opacity);
      const placeholder = !own && isField && !input.value;
      const color = rgba(placeholder ? getComputedStyle(el, "::placeholder").color : cs.color);
      const disabled = (el as HTMLButtonElement).disabled === true || !!el.closest("[aria-disabled=true], [disabled], fieldset:disabled");
      items.push({
        text: (own || input.value || input.placeholder).slice(0, 40),
        tag: el.tagName.toLowerCase(),
        cls: typeof el.className === "string" ? el.className : "",
        x: r.left,
        y: r.top,
        w: r.width,
        h: r.height,
        color,
        opacity,
        size: parseFloat(cs.fontSize),
        weight: parseInt(cs.fontWeight, 10),
        placeholder,
        disabled,
        keycap: el.tagName === "KBD",
      });
    }
    reach.remove();
    return { items, labels };
  });
}

/**
 * What is behind each box, read from a screenshot (PNG, one pixel per CSS pixel): the commonest colour of the box (the glyphs are a
 * minority of its pixels, so the commonest colour is the ground), the mean of the pixels of the commonest 6-bit colour. The PNG is decoded
 * in the page (`createImageBitmap`), so the suite needs no image library.
 */
export async function groundsOf(page: Page, png: Buffer, boxes: readonly Box[]): Promise<Array<[number, number, number]>> {
  return page.evaluate(
    async ({ b64, list }) => {
      const bitmap = await createImageBitmap(await (await fetch(`data:image/png;base64,${b64}`)).blob());
      const canvas = new OffscreenCanvas(bitmap.width, bitmap.height);
      const cx = canvas.getContext("2d", { willReadFrequently: true }) as OffscreenCanvasRenderingContext2D;
      cx.drawImage(bitmap, 0, 0);
      const out: Array<[number, number, number]> = [];
      for (const b of list) {
        const x0 = Math.max(0, Math.floor(b.x));
        const y0 = Math.max(0, Math.floor(b.y));
        const x1 = Math.min(bitmap.width, Math.ceil(b.x + b.w));
        const y1 = Math.min(bitmap.height, Math.ceil(b.y + b.h));
        if (x1 - x0 < 1 || y1 - y0 < 1) {
          out.push([0, 0, 0]);
          continue;
        }
        const { data } = cx.getImageData(x0, y0, x1 - x0, y1 - y0);
        const buckets = new Map<number, [number, number, number, number]>();
        for (let i = 0; i < data.length; i += 4) {
          const key = ((data[i]! >> 2) << 12) | ((data[i + 1]! >> 2) << 6) | (data[i + 2]! >> 2);
          const bucket = buckets.get(key) ?? [0, 0, 0, 0];
          bucket[0] += data[i]!;
          bucket[1] += data[i + 1]!;
          bucket[2] += data[i + 2]!;
          bucket[3] += 1;
          buckets.set(key, bucket);
        }
        let best: [number, number, number, number] = [0, 0, 0, 0];
        for (const bucket of buckets.values()) if (bucket[3] > best[3]) best = bucket;
        out.push([best[0] / best[3], best[1] / best[3], best[2] / best[3]]);
      }
      return out;
    },
    { b64: png.toString("base64"), list: boxes as Box[] },
  );
}

/** WCAG relative luminance of an sRGB colour (rgb 0 to 255). */
export function luminance(c: readonly number[]): number {
  const lin = (v: number): number => (v / 255 <= 0.03928 ? v / 255 / 12.92 : ((v / 255 + 0.055) / 1.055) ** 2.4);
  return 0.2126 * lin(c[0]!) + 0.7152 * lin(c[1]!) + 0.0722 * lin(c[2]!);
}

/** WCAG contrast ratio of two colours. */
export function contrastRatio(a: readonly number[], b: readonly number[]): number {
  const [hi, lo] = [luminance(a), luminance(b)].sort((x, y) => y - x) as [number, number];
  return (hi + 0.05) / (lo + 0.05);
}

/** One measured text: the contrast of its colour, over what is behind it, and what it needs. */
export interface Measured {
  text: string;
  cls: string;
  tag: string;
  size: number;
  disabled: boolean;
  keycap: boolean;
  placeholder: boolean;
  /** The colour of the text as drawn (its colour and opacity over the ground). */
  fg: [number, number, number];
  /** The ground behind it, from the screenshot. */
  bg: [number, number, number];
  ratio: number;
  /** 3 for large text (24 px, or 18.66 px and bold), else 4.5. */
  need: number;
  /** Which of the label colours the element has: the three tokens, or `other`. */
  token: "label" | "secondary" | "tertiary" | "other";
}

/** Measures texts against the grounds read from a screenshot. */
export function measureText(items: readonly TextItem[], grounds: ReadonlyArray<[number, number, number]>, labels: LabelColours): Measured[] {
  const near = (a: readonly number[], b: readonly number[]): boolean => Math.abs(a[0]! - b[0]!) <= 2 && Math.abs(a[1]! - b[1]!) <= 2 && Math.abs(a[2]! - b[2]!) <= 2;
  return items.map((it, i) => {
    const bg = grounds[i] as [number, number, number];
    const alpha = it.color[3] * it.opacity;
    const fg = [0, 1, 2].map((k) => it.color[k]! * alpha + bg[k]! * (1 - alpha)) as [number, number, number];
    const large = it.size >= 24 || (it.size >= 18.66 && it.weight >= 700);
    const token = near(it.color, labels.label) ? "label" : near(it.color, labels.secondary) ? "secondary" : near(it.color, labels.tertiary) ? "tertiary" : "other";
    return {
      text: it.text,
      cls: it.cls,
      tag: it.tag,
      size: it.size,
      disabled: it.disabled,
      keycap: it.keycap,
      placeholder: it.placeholder,
      fg,
      bg,
      ratio: contrastRatio(fg, bg),
      need: large ? 3 : 4.5,
      token,
    };
  });
}

/**
 * How well a focus ring shows: the contrast of what the focus changed, against what was there before. `before` and `after` are screenshots
 * of the same page (one pixel per CSS pixel) without and with the focus; `clip` is the part to compare. Every pixel that differs is a pixel
 * of the ring (an outline, and the dark line of a double ring); each is rated by its contrast with the pixel it replaced. The ring's rating is
 * the contrast that the best 15 % of its pixels reach: a ring of two lines, an accent and a dark line outside it, reads by the line that
 * contrasts, which is a quarter of its width, while one line of one colour has to do it all.
 */
export async function ringContrast(page: Page, before: Buffer, after: Buffer, clip: Box): Promise<{ ratio: number; pixels: number }> {
  return page.evaluate(
    async ({ a64, b64, c }) => {
      const load = async (b: string): Promise<ImageData> => {
        const bitmap = await createImageBitmap(await (await fetch(`data:image/png;base64,${b}`)).blob());
        const canvas = new OffscreenCanvas(bitmap.width, bitmap.height);
        const cx = canvas.getContext("2d", { willReadFrequently: true }) as OffscreenCanvasRenderingContext2D;
        cx.drawImage(bitmap, 0, 0);
        const x = Math.max(0, Math.floor(c.x));
        const y = Math.max(0, Math.floor(c.y));
        return cx.getImageData(x, y, Math.min(bitmap.width - x, Math.ceil(c.w)), Math.min(bitmap.height - y, Math.ceil(c.h)));
      };
      const [a, b] = [await load(a64), await load(b64)];
      const lin = (v: number): number => (v / 255 <= 0.03928 ? v / 255 / 12.92 : ((v / 255 + 0.055) / 1.055) ** 2.4);
      const lum = (d: Uint8ClampedArray, i: number): number => 0.2126 * lin(d[i]!) + 0.7152 * lin(d[i + 1]!) + 0.0722 * lin(d[i + 2]!);
      const ratios: number[] = [];
      for (let i = 0; i < a.data.length; i += 4) {
        if (a.data[i] === b.data[i] && a.data[i + 1] === b.data[i + 1] && a.data[i + 2] === b.data[i + 2]) continue;
        const [hi, lo] = [lum(a.data, i), lum(b.data, i)].sort((p, q) => q - p) as [number, number];
        ratios.push((hi + 0.05) / (lo + 0.05));
      }
      ratios.sort((p, q) => q - p);
      return { ratio: ratios.length === 0 ? 1 : ratios[Math.floor(ratios.length * 0.15)]!, pixels: ratios.length };
    },
    { a64: before.toString("base64"), b64: after.toString("base64"), c: clip },
  );
}
