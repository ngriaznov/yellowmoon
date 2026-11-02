/**
 * What a mask row says about itself (`docs/23` §7.6): its name, its subtitle under the name and the accessible name of the row. Pure, so the
 * Mask tab's rows (read only in N-09, editable from N-12) and their tests do not need a screen. The numbers are what the sliders show
 * (`docs/algorithms/06` §4.5): Feather and Softness 0 to 100 are 0 to 1, a Colour range's Width is `0.005 + 0.295 · (x / 100)²`, Low and
 * High are the code values times 100.
 */
import type { MaskItem } from "../api/types";
import { MASK_KIND_NAME } from "./graphOps";

/** The row's name in bold: "Object", "Radial", "Colour range". */
export const maskRowName = (row: MaskItem): string => MASK_KIND_NAME[row.kind] ?? row.kind;

const plural = (n: number, one: string, many: string): string => `${n} ${n === 1 ? one : many}`;

/** A Colour range's width as its slider shows it: the inverse of `0.005 + 0.295 · (x / 100)²`. */
export const widthSlider = (width: number): number => Math.round(100 * Math.sqrt(Math.max(0, (width - 0.005) / 0.295)));

/**
 * The subtitle of a row (`docs/23` §7.6): "2 points", "1 point", "Box", "Box and 2 points" and, with none, "Click the object on this photo"
 * (Object); "Foreground" (Subject); "People"; "Level 50" (Background); "Feather 20" (Linear, Radial); "3 strokes", "1 stroke" and "Paint on the
 * photo" (Brush); "Width 40" (Colour range); "35 to 100" (Brightness range). The states of a mask being made ("Finding…") and of a Mac that
 * cannot make one are the Mask tab's (N-12).
 */
export function maskSubtitle(row: MaskItem): string {
  switch (row.kind) {
    case "object": {
      const points = row.points.length;
      if (row.box && points > 0) return `Box and ${plural(points, "point", "points")}`;
      if (row.box) return "Box";
      return points > 0 ? plural(points, "point", "points") : "Click the object on this photo";
    }
    case "subject":
      return "Foreground";
    case "people":
      return "People";
    case "background":
      return `Level ${Math.round(row.level * 100)}`;
    case "linear":
    case "radial":
      return `Feather ${Math.round(row.feather * 100)}`;
    case "brush":
      return row.strokes.length > 0 ? plural(row.strokes.length, "stroke", "strokes") : "Paint on the photo";
    case "color_range":
      return `Width ${widthSlider(row.width)}`;
    case "brightness_range":
      return `${Math.round(row.low * 100)} to ${Math.round(row.high * 100)}`;
  }
}

/** The row's accessible name: the kind, "mask" and the subtitle ("Object mask, 2 points", "Radial mask, Feather 45"). */
export const maskRowLabel = (row: MaskItem): string => `${maskRowName(row)} mask, ${maskSubtitle(row)}`;
